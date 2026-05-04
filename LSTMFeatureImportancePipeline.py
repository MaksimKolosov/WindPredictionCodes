# Feature Importance

!pip install optuna

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# УСТРОЙСТВО И ФИКСАЦИЯ СЛУЧАЙНОСТИ
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {DEVICE}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# ЗАГРУЗКА ДАННЫХ ИЗ НЕСКОЛЬКИХ ФАЙЛОВ
# ============================================================================
class MultiFileDataLoader:
    def __init__(self, data_paths: list, target_column: str,
                 exclude_columns: list = None,
                 min_segment_minutes: int = 60, frequency_minutes: int = 1):
        self.data_paths = data_paths
        self.target_column = target_column
        self.exclude_columns = exclude_columns if exclude_columns is not None else ['date']
        self.min_segment_minutes = min_segment_minutes
        self.frequency_minutes = frequency_minutes
        self.expected_interval_seconds = frequency_minutes * 60
        self.continuous_segments = []
        self.feature_columns = None

    def load_and_extract_segments(self):
        print("--- Загрузка данных ---")
        all_segments = []
        feature_columns_set = None
        for file_path in self.data_paths:
            print(f"  Файл: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            if self.target_column not in df.columns:
                print(f"    Целевая колонка '{self.target_column}' не найдена, пропуск")
                continue
            if feature_columns_set is None:
                exclude = set(self.exclude_columns + [self.target_column])
                feature_columns_set = [c for c in df.columns if c not in exclude]
                self.feature_columns = feature_columns_set
            df = df[['date', self.target_column] + self.feature_columns]
            df['time_diff'] = df['date'].diff().dt.total_seconds()
            gap_mask = df['time_diff'] > (self.expected_interval_seconds + 10)
            df['segment_id'] = gap_mask.cumsum()
            for seg_id in df['segment_id'].unique():
                seg = df[df['segment_id'] == seg_id].copy()
                seg = seg.drop(columns=['time_diff', 'segment_id'])
                duration = (seg['date'].iloc[-1] - seg['date'].iloc[0]).total_seconds() / 60
                if duration >= self.min_segment_minutes and len(seg) >= self.min_segment_minutes:
                    all_segments.append(seg)
        self.continuous_segments = all_segments
        return self

    def get_all_data(self):
        if not self.continuous_segments:
            raise ValueError("Нет непрерывных отрезков")
        combined_list = []
        for seg in self.continuous_segments:
            target_data = seg[[self.target_column]].values
            feature_data = seg[self.feature_columns].values
            combined = np.hstack([target_data, feature_data])
            combined_list.append(combined)
        all_data = np.vstack(combined_list)
        return all_data

# ============================================================================
# ФОРМИРОВАНИЕ ДАТАСЕТОВ
# ============================================================================
class TimeSeriesDatasetBuilder:
    def __init__(self, lookback_minutes: int, forecast_minutes: int,
                 frequency_minutes: int = 1, stride_minutes: int = 60):
        self.lookback = lookback_minutes // frequency_minutes
        self.forecast = forecast_minutes // frequency_minutes
        self.stride = stride_minutes // frequency_minutes

    def create_sequences(self, data: np.ndarray, target_col_idx: int = 0):
        X, y = [], []
        for i in range(0, len(data) - self.lookback - self.forecast + 1, self.stride):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback + self.forecast - 1, target_col_idx])
        return np.array(X), np.array(y)

    def temporal_train_test_split(self, X, y, train_ratio=0.7, val_ratio=0.15):
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return (X[:train_end], X[train_end:val_end], X[val_end:],
                y[:train_end], y[train_end:val_end], y[val_end:])

# ============================================================================
# LSTM МОДЕЛЬ (С РЕГУЛЯРИЗАЦИЕЙ)
# ============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1,
                 dropout_rate: float = 0.3, output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout_rate if num_layers > 1 else 0,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out  # (batch, output_size)

class LSTMTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001, weight_decay: float = 1e-5):
        self.model = model
        self.model.to(DEVICE)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=16, patience=15,
              trial=None, verbose=True):
        X_train_t = torch.FloatTensor(X_train).to(DEVICE)
        y_train_t = torch.FloatTensor(y_train).to(DEVICE)
        X_val_t = torch.FloatTensor(X_val).to(DEVICE)
        y_val_t = torch.FloatTensor(y_val).to(DEVICE)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}

        for epoch in range(epochs):
            # --- Обучение ---
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_dataset)
            history['train_loss'].append(train_loss)

            # --- Валидация ---
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t).squeeze()
                val_loss = self.criterion(val_outputs, y_val_t).item()
                train_pred = self.model(X_train_t).squeeze()
                train_mae = torch.mean(torch.abs(train_pred - y_train_t)).item()
                val_mae = torch.mean(torch.abs(val_outputs - y_val_t)).item()
                history['train_mae'].append(train_mae)
                history['val_mae'].append(val_mae)
                history['val_loss'].append(val_loss)

            self.scheduler.step(val_loss)

            # --- Вывод каждые 10 эпох ---
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}/{epochs}: train_MAE={train_mae:.6f}, val_MAE={val_mae:.6f}, val_loss={val_loss:.6f}")

            # --- Отчёт для Optuna (pruning) ---
            if trial is not None:
                trial.report(val_mae, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def predict(self, X):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(DEVICE)
        with torch.no_grad():
            return self.model(X_t).squeeze().cpu().numpy()

# ============================================================================
# ПАЙПЛАЙН АНАЛИЗА ВАЖНОСТИ ПРИЗНАКОВ
# ============================================================================
class FeatureImportancePipeline:
    def __init__(self, data_paths, target_column, exclude_columns,
                 lookback_minutes, forecast_horizons, stride_minutes=60,
                 hidden_size=64, num_layers=1, dropout_rate=0.3,
                 learning_rate=0.001, weight_decay=1e-4, batch_size=16,
                 epochs=50, patience=10):
        # базовые гиперпараметры (по умолчанию для всех горизонтов)
        self.data_paths = data_paths
        self.target_column = target_column
        self.exclude_columns = exclude_columns
        self.lookback_minutes = lookback_minutes
        self.forecast_horizons = forecast_horizons
        self.stride_minutes = stride_minutes
        self.default_hparams = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'epochs': epochs,
            'patience': patience,
            'lookback': lookback_minutes
        }
        self.scaled_data = None
        self.feature_columns = None
        self.feature_names = None
        self.models = {}
        self.shap_results = {}
        self.perm_importance = {}
        self.metrics = {}    # {horizon: {'MAE':..., 'RMSE':..., 'R2':..., 'Bias':..., 'StdRatio':...}}

    def load_data(self):
        loader = MultiFileDataLoader(
            self.data_paths, self.target_column, self.exclude_columns,
            min_segment_minutes=self.default_hparams['lookback'] + 100
        )
        loader.load_and_extract_segments()
        self.scaled_data = loader.get_all_data()
        self.feature_columns = loader.feature_columns
        self.feature_names = [self.target_column] + self.feature_columns
        print(f"Данные загружены. Форма: {self.scaled_data.shape}")
        return self

    def _calculate_metrics(self, y_true, y_pred):
        """Вычисляет MAE, RMSE, R², Bias, отношение стандартных отклонений."""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        bias = np.mean(y_pred - y_true)
        std_ratio = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else np.nan
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Bias': bias, 'StdRatio': std_ratio}

    def train_base_models(self, horizon_params=None):
        """Обучает LSTM для каждого горизонта. horizon_params: dict {forecast: {param: value}}."""
        for forecast in self.forecast_horizons:
            print(f"\n--- Обучение модели для горизонта {forecast} мин ---")
            # Берём гиперпараметры для данного горизонта (если заданы), иначе default
            hp = self.default_hparams.copy()
            if horizon_params is not None and forecast in horizon_params:
                hp.update(horizon_params[forecast])

            lookback = hp['lookback']
            builder = TimeSeriesDatasetBuilder(lookback, forecast, stride_minutes=self.stride_minutes)
            X, y = builder.create_sequences(self.scaled_data, target_col_idx=0)
            print(f"  Создано последовательностей: {len(X)} (lookback={lookback} мин)")
            if len(X) < 200:
                raise ValueError(f"Недостаточно данных для горизонта {forecast}")
            X_train, X_val, X_test, y_train, y_val, y_test = builder.temporal_train_test_split(X, y)

            model = LSTMModel(
                input_size=X_train.shape[2],
                hidden_size=hp['hidden_size'],
                num_layers=hp['num_layers'],
                dropout_rate=hp['dropout_rate']
            )
            trainer = LSTMTrainer(model, learning_rate=hp['learning_rate'], weight_decay=hp['weight_decay'])
            trainer.train(X_train, y_train, X_val, y_val,
                          epochs=hp['epochs'], batch_size=hp['batch_size'], patience=hp['patience'], verbose=True)

            y_pred = trainer.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            self.metrics[forecast] = metrics

            # Вычисляем коэффициент пересчёта в зависимости от целевой переменной
            if self.target_column == 'wind_speed_horizontal':
                koef = 12.5
            elif self.target_column == 'wind_speed_vertical':
                koef = 25
            else:
                koef = 1.0

            print(f"  Горизонт {forecast}: MAE={metrics['MAE']:.6f} ({metrics['MAE']*koef:.2f} м/с), "
                  f"RMSE={metrics['RMSE']:.6f} ({metrics['RMSE']*koef:.2f} м/с), "
                  f"R2={metrics['R2']:.4f}, "
                  f"Bias={metrics['Bias']:.6f} ({metrics['Bias']*koef:.2f} м/с), "
                  f"StdRatio={metrics['StdRatio']:.4f}")

            # Сохраняем для дальнейшего SHAP и Permutation
            self.models[forecast] = trainer
        return self

    def compute_shap_importance(self, n_samples=500):
        for forecast, trainer in self.models.items():
            print(f"\n--- Вычисление SHAP для горизонта {forecast} мин ---")
            builder = TimeSeriesDatasetBuilder(
                self.lookback_minutes, forecast,
                stride_minutes=self.stride_minutes
            )
            X, y = builder.create_sequences(self.scaled_data, target_col_idx=0)
            X_train, X_val, X_test, _, _, _ = builder.temporal_train_test_split(X, y)

            n_val = min(n_samples, X_val.shape[0])
            n_train = min(100, X_train.shape[0])
            X_background = X_train[:n_train]
            X_explain = X_val[:n_val]

            X_background_t = torch.FloatTensor(X_background)
            X_explain_t = torch.FloatTensor(X_explain)

            explainer = shap.GradientExplainer(trainer.model, X_background_t)
            shap_values = explainer.shap_values(X_explain_t)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = shap_values.detach().cpu().numpy() if hasattr(shap_values, 'detach') else np.array(shap_values)

            # Усреднение по примерам и временным шагам -> важность признаков
            feature_importance = np.mean(np.abs(shap_values), axis=(0, 1)).flatten()
            # Обрезаем до числа признаков (на всякий случай)
            if len(feature_importance) > len(self.feature_names):
                feature_importance = feature_importance[:len(self.feature_names)]
            elif len(feature_importance) < len(self.feature_names):
                # Дополняем нулями (не должно случиться, но на всякий случай)
                feature_importance = np.pad(feature_importance, (0, len(self.feature_names)-len(feature_importance)))

            imp_df = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': feature_importance
            }).sort_values('shap_importance', ascending=False)
            self.shap_results[forecast] = imp_df
            print(imp_df)

            # Направление влияния
            direction = np.mean(shap_values, axis=(0, 1)).flatten()[:len(self.feature_names)]
            imp_df['direction'] = ['positive' if d > 0 else 'negative' for d in direction]
        return self

    def compute_permutation_importance(self, n_samples=300, n_repeats=3):
        for forecast, trainer in self.models.items():
            print(f"\n--- Permutation Importance для горизонта {forecast} мин ---")
            builder = TimeSeriesDatasetBuilder(
                self.lookback_minutes, forecast,
                stride_minutes=self.stride_minutes
            )
            X, y = builder.create_sequences(self.scaled_data, target_col_idx=0)
            n = len(X)
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            X_test = X[val_end:]
            y_test = y[val_end:]
            if len(X_test) > n_samples:
                idx = np.random.choice(len(X_test), n_samples, replace=False)
                X_test = X_test[idx]
                y_test = y_test[idx]

            baseline_mae = mean_absolute_error(y_test, trainer.predict(X_test))

            perm_importances = []
            for i in range(X_test.shape[2]):  # по каждому признаку
                scores = []
                for _ in range(n_repeats):
                    X_permuted = X_test.copy()
                    X_permuted[:, :, i] = np.random.permutation(X_permuted[:, :, i])
                    mae = mean_absolute_error(y_test, trainer.predict(X_permuted))
                    scores.append(mae - baseline_mae)
                perm_importances.append(np.mean(scores))

            perm_df = pd.DataFrame({
                'feature': self.feature_names,
                'perm_importance': perm_importances
            }).sort_values('perm_importance', ascending=False)
            self.perm_importance[forecast] = perm_df
            print(perm_df)
        return self

    def plot_importance(self):
        # SHAP plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, forecast in enumerate(self.forecast_horizons):
            ax = axes[i]
            imp_df = self.shap_results[forecast]  # все признаки
            ax.barh(imp_df['feature'], imp_df['shap_importance'], color='skyblue')
            ax.set_title(f'SHAP важность (горизонт {forecast} мин)')
            ax.set_xlabel('Среднее абсолютное SHAP-значение')
            ax.grid(axis='x')
        plt.tight_layout()
        plt.show()

        # Permutation Importance plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, forecast in enumerate(self.forecast_horizons):
            ax = axes[i]
            perm_df = self.perm_importance[forecast]   # все признаки
            ax.barh(perm_df['feature'], perm_df['perm_importance'], color='lightcoral')
            ax.set_title(f'Permutation Importance (горизонт {forecast} мин)')
            ax.set_xlabel('Увеличение MAE при перемешивании')
            ax.grid(axis='x')
        plt.tight_layout()
        plt.show()

    def optimize_features(self, threshold_ratio=0.05):
        recommended = {}
        for forecast, imp_df in self.shap_results.items():
            max_imp = imp_df['shap_importance'].max()
            threshold = max_imp * threshold_ratio
            to_keep = imp_df[imp_df['shap_importance'] >= threshold]['feature'].tolist()
            to_drop = imp_df[imp_df['shap_importance'] < threshold]['feature'].tolist()
            recommended[forecast] = to_keep
            print(f"\nГоризонт {forecast} мин: следует сохранить {len(to_keep)} признаков, удалить {len(to_drop)}")
            if to_drop:
                print(f"  Удаляемые признаки: {to_drop}")
        return recommended

    def run(self, horizon_params=None):
        self.load_data()
        self.train_base_models(horizon_params)
        self.compute_shap_importance()
        self.compute_permutation_importance()
        self.plot_importance()
        return self.optimize_features()

# ============================================================================
# ЗАПУСК АНАЛИЗА ВАЖНОСТИ ПРИЗНАКОВ
# ============================================================================
if __name__ == "__main__":
    DATA_PATHS = [
        "/content/winter_oblkom_27m.csv",
        "/content/spring_oblkom_27m.csv",
        "/content/summer_oblkom_26m.csv",
        "/content/autumn_oblkom_27m.csv"
    ]
    EXCLUDE = ['date']
    FORECASTS = [30, 60, 120, 180]

    # Гиперпараметры для горизонтальной скорости ветра
    horizon_params_horizontal = {
        30: {
            'lookback': 600,
            'learning_rate': 0.00064,
            'batch_size': 8,
            'weight_decay': 0.00031,
            'hidden_size': 64,
            'dropout_rate': 0.25
        },
        60: {
            'lookback': 360,
            'learning_rate': 0.00187,
            'batch_size': 8,
            'weight_decay': 0.000035,
            'hidden_size': 64,
            'dropout_rate': 0.35
        },
        120: {
            'lookback': 240,
            'learning_rate': 0.00846,
            'batch_size': 16,
            'weight_decay': 0.000994,
            'hidden_size': 64,
            'dropout_rate': 0.35
        },
        180: {
            'lookback': 240,
            'learning_rate': 0.00311,
            'batch_size': 16,
            'weight_decay': 0.0000567,
            'hidden_size': 64,
            'dropout_rate': 0.3
        }
    }

    # Гиперпараметры для вертикальной скорости ветра
    horizon_params_vertical = {
        30: {
            'lookback': 360,
            'learning_rate': 0.000225,
            'batch_size': 8,
            'weight_decay': 0.0000134,
            'hidden_size': 64,
            'dropout_rate': 0.25
        },
        60: {
            'lookback': 600,
            'learning_rate': 0.00259,
            'batch_size': 16,
            'weight_decay': 0.0000141,
            'hidden_size': 32,
            'dropout_rate': 0.4
        },
        120: {
            'lookback': 720,
            'learning_rate': 0.0011,
            'batch_size': 8,
            'weight_decay': 0.0000189,
            'hidden_size': 64,
            'dropout_rate': 0.3
        },
        180: {
            'lookback': 480,
            'learning_rate': 0.000457,
            'batch_size': 16,
            'weight_decay': 0.0000188,
            'hidden_size': 64,
            'dropout_rate': 0.25
        }
    }

    targets = [
        {"name": "wind_speed_horizontal", "params": horizon_params_horizontal, "scale": 12.5},
        {"name": "wind_speed_vertical", "params": horizon_params_vertical, "scale": 25}
    ]

    for target in targets:
        print(f"\n{'='*70}")
        print(f"ЗАПУСК АНАЛИЗА ВАЖНОСТИ ПРИЗНАКОВ ДЛЯ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ: {target['name']}")
        print(f"{'='*70}")

        pipeline = FeatureImportancePipeline(
            data_paths=DATA_PATHS,
            target_column=target['name'],
            exclude_columns=EXCLUDE,
            lookback_minutes=360,
            forecast_horizons=FORECASTS,
            stride_minutes=270,
            hidden_size=64,
            num_layers=1,
            dropout_rate=0.3,
            learning_rate=0.001,
            weight_decay=0.0001,
            batch_size=16,
            epochs=50,
            patience=10
        )
        optimal_features = pipeline.run(horizon_params=target['params'])
        print("\n=== Метрики по горизонтам ===")
        for h, metrics in pipeline.metrics.items():
            print(f"{h} мин: MAE={metrics['MAE']:.6f} ({metrics['MAE']*target['scale']:.2f} м/с), "
                  f"RMSE={metrics['RMSE']:.6f} ({metrics['RMSE']*target['scale']:.2f} м/с), "
                  f"R2={metrics['R2']:.4f}, "
                  f"Bias={metrics['Bias']:.6f} ({metrics['Bias']*target['scale']:.2f} м/с), "
                  f"StdRatio={metrics['StdRatio']:.4f}")