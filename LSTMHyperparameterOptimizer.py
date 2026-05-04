# LSTMHyperparameterOptimizer

!pip install optuna

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import gc
from optuna.trial import TrialState
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
# ПАЙПЛАЙН НАСТРОЙКИ ГИПЕРПАРАМЕТРОВ С OPTUNA
# ============================================================================
class LSTMHyperparameterOptimizer:
    def __init__(self, data_paths, target_column, exclude_columns,
                 forecast_horizons, stride_minutes=60,
                 n_trials=50, n_repeats=2, min_segment_minutes=480):
        self.data_paths = data_paths
        self.target_column = target_column
        self.exclude_columns = exclude_columns
        self.forecast_horizons = forecast_horizons
        self.stride_minutes = stride_minutes
        self.n_trials = n_trials
        self.n_repeats = n_repeats
        self.min_segment_minutes = min_segment_minutes
        self.scaled_data = None
        self.feature_columns = None

    def load_data(self):
        loader = MultiFileDataLoader(
            self.data_paths, self.target_column, self.exclude_columns,
            min_segment_minutes=self.min_segment_minutes
        )
        loader.load_and_extract_segments()
        self.scaled_data = loader.get_all_data()
        self.feature_columns = loader.feature_columns
        print(f"Данные загружены. Форма: {self.scaled_data.shape}")
        return self

    def objective(self, trial, forecast_min):
        # ДИАПАЗОНЫ ГИПЕРПАРАМЕТРОВ
        lookback = trial.suggest_categorical('lookback', list(range(120, 721, 120)))
        units = trial.suggest_categorical('units', [32, 64])
        num_layers = 1
        dropout = trial.suggest_float('dropout', 0.25, 0.4, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16])
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        builder = TimeSeriesDatasetBuilder(lookback, forecast_min, stride_minutes=self.stride_minutes)
        X, y = builder.create_sequences(self.scaled_data, target_col_idx=0)
        if len(X) < 200:
            return float('inf')

        print(f"  [Trial] num_of_seq={len(X)}")

        X_train, X_val, X_test, y_train, y_val, y_test = builder.temporal_train_test_split(X, y)

        val_maes = []
        val_rmses = []
        val_biases = []
        val_std_ratios = []
        for repeat in range(self.n_repeats):
            model = LSTMModel(
                input_size=X_train.shape[2],
                hidden_size=units,
                num_layers=num_layers,
                dropout_rate=dropout
            )
            trainer = LSTMTrainer(model, learning_rate=lr, weight_decay=weight_decay)
            history = trainer.train(X_train, y_train, X_val, y_val,
                                    epochs=100, batch_size=batch_size, patience=15,
                                    trial=trial, verbose=False)
            y_pred_val = trainer.predict(X_val)

            mae = mean_absolute_error(y_val, y_pred_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            bias = np.mean(y_pred_val - y_val)
            std_ratio = np.std(y_pred_val) / np.std(y_val) if np.std(y_val) > 0 else np.nan

            val_maes.append(mae)
            val_rmses.append(rmse)
            val_biases.append(bias)
            val_std_ratios.append(std_ratio)

            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_mae = np.mean(val_maes)
        mean_rmse = np.mean(val_rmses)
        mean_bias = np.mean(val_biases)
        mean_std_ratio = np.mean(val_std_ratios)

        scale = self.get_descale_koef_for_error()
        print(f"  [Trial] MAE={mean_mae:.6f} ({mean_mae*scale:.2f} м/с), "
              f"RMSE={mean_rmse:.6f} ({mean_rmse*scale:.2f} м/с), "
              f"Bias={mean_bias:.6f} ({mean_bias*scale:.2f} м/с), "
              f"StdRatio={mean_std_ratio:.4f}")

        return mean_mae

    def optimize_for_horizon(self, forecast_min):
        print(f"\n=== Оптимизация гиперпараметров для горизонта {forecast_min} мин ===")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: self.objective(trial, forecast_min), n_trials=self.n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_value = study.best_value
        print(f"Лучшие параметры: {best_params}")
        descale_koef = self.get_descale_koef_for_error()
        best_value_descaled = best_value * descale_koef
        print(f"Лучшее среднее val MAE: {best_value:.6f} ({best_value_descaled:.2f} м/с)")
        return best_params, best_value

    def run(self):
        self.load_data()
        results = {}
        for forecast in self.forecast_horizons:
            best_params, best_mae = self.optimize_for_horizon(forecast)
            results[forecast] = {
                'best_params': best_params,
                'best_val_mae': best_mae,
                'best_lookback': best_params['lookback']
            }
        return results

    def get_descale_koef_for_error(self):
        if self.target_column == 'wind_speed_horizontal':
            return 12.5
        elif self.target_column == 'wind_speed_vertical':
            return 25
        else:
            return 1

# ============================================================================
# ФИНАЛЬНОЕ ОБУЧЕНИЕ И ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ (С НАИВНЫМ ПРОГНОЗОМ)
# ============================================================================
def train_final_model(data_paths, target_column, exclude_columns,
                      forecast_min, best_params, stride_minutes=120):
    # Очистка памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    loader = MultiFileDataLoader(data_paths, target_column, exclude_columns,
                                 min_segment_minutes=360)
    loader.load_and_extract_segments()
    scaled_data = loader.get_all_data()

    lookback = best_params['lookback']
    builder = TimeSeriesDatasetBuilder(lookback, forecast_min, stride_minutes=stride_minutes)
    X, y = builder.create_sequences(scaled_data, target_col_idx=0)
    if len(X) == 0:
        raise ValueError(f"Нет данных для lookback={lookback}, forecast={forecast_min}")

    X_train, X_val, X_test, y_train, y_val, y_test = builder.temporal_train_test_split(X, y)

    # ВЫВОД СТАТИСТИКИ ПОСЛЕДОВАТЕЛЬНОСТЕЙ
    print(f"\n=== СТАТИСТИКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ ===")
    print(f"\nГоризонт: {forecast_min} мин")
    print(f"Всего последовательностей: {len(X)}")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Размер окна: {lookback} мин = {X_train.shape[1]} временных шагов")
    print(f"Количество признаков: {X_train.shape[2]}")
    print(f"====================================\n")

    # --- Обучение LSTM ---
    model = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=best_params['units'],
        num_layers=1,
        dropout_rate=best_params['dropout']
    )
    trainer = LSTMTrainer(model, learning_rate=best_params['lr'], weight_decay=best_params['weight_decay'])
    trainer.train(X_train, y_train, X_val, y_val,
                  epochs=100, batch_size=best_params.get('batch_size', 16), patience=15, verbose=True)

    y_pred_lstm = trainer.predict(X_test)

    # --- НАИВНЫЙ ПРОГНОЗ (PERSISTENCE) ---
    # Берём последнее значение целевой переменной из каждого окна
    y_pred_persistence = X_test[:, -1, 0]  # индекс 0 – целевая переменная

    # --- РАСЧЁТ МЕТРИК ---
    scale = 12.5 if target_column == 'wind_speed_horizontal' else 25

    # LSTM метрики
    mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
    bias_lstm = np.mean(y_pred_lstm - y_test)
    std_ratio_lstm = np.std(y_pred_lstm) / np.std(y_test) if np.std(y_test) > 0 else np.nan

    # Наивный прогноз метрики
    mae_pers = mean_absolute_error(y_test, y_pred_persistence)
    rmse_pers = np.sqrt(mean_squared_error(y_test, y_pred_persistence))
    bias_pers = np.mean(y_pred_persistence - y_test)
    std_ratio_pers = np.std(y_pred_persistence) / np.std(y_test) if np.std(y_test) > 0 else np.nan

    # Улучшение
    improvement_mae = (mae_pers - mae_lstm) / mae_pers * 100

    print(f"\n--- РЕЗУЛЬТАТЫ ДЛЯ ГОРИЗОНТА {forecast_min} мин ---")
    print("\nLSTM прогноз:")
    print(f"  MAE  : {mae_lstm:.6f} ({mae_lstm*scale:.2f} м/с)")
    print(f"  RMSE : {rmse_lstm:.6f} ({rmse_lstm*scale:.2f} м/с)")
    print(f"  Bias : {bias_lstm:.6f} ({bias_lstm*scale:.2f} м/с)")
    print(f"  StdRatio : {std_ratio_lstm:.4f}")

    print("\nНаивный прогноз (persistence):")
    print(f"  MAE  : {mae_pers:.6f} ({mae_pers*scale:.2f} м/с)")
    print(f"  RMSE : {rmse_pers:.6f} ({rmse_pers*scale:.2f} м/с)")
    print(f"  Bias : {bias_pers:.6f} ({bias_pers*scale:.2f} м/с)")
    print(f"  StdRatio : {std_ratio_pers:.4f}")

    print(f"\nУлучшение MAE по сравнению с наивным прогнозом: {improvement_mae:.2f}%")

    return trainer, mae_lstm

# ============================================================================
# ЗАПУСК ПАЙПЛАЙНА ДЛЯ НЕСКОЛЬКИХ ЦЕЛЕВЫХ ПЕРЕМЕННЫХ
# ============================================================================
if __name__ == "__main__":
    DATA_PATHS = [
        "/content/winter_oblkom_27m.csv",
        "/content/spring_oblkom_27m.csv",
        "/content/summer_oblkom_26m.csv",
        "/content/autumn_oblkom_27m.csv"
    ]
    EXCLUDE = ['date']
    FORECAST_HORIZONS = [30, 60, 120, 180]

    # Список целевых переменных
    TARGETS = ["wind_speed_horizontal", "wind_speed_vertical"]

    # Общие параметры оптимизации
    OPTIMIZER_PARAMS = {
        'stride_minutes': 270,
        'n_trials': 50,
        'n_repeats': 2,
        'min_segment_minutes': 150
    }

    for target in TARGETS:
        print(f"\n***** ЗАПУСК ПОДБОРА ГИПЕРПАРАМЕТРОВ МОДЕЛИ LSTM ДЛЯ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ: {target} *****")
        print("="*70)

        # Создаём оптимизатор
        optimizer = LSTMHyperparameterOptimizer(
            data_paths=DATA_PATHS,
            target_column=target,
            exclude_columns=EXCLUDE,
            forecast_horizons=FORECAST_HORIZONS,
            **OPTIMIZER_PARAMS
        )

        # Запускаем оптимизацию
        results = optimizer.run()

        # Вывод результатов
        print(f"\nРЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ДЛЯ {target}")
        print("="*70)
        for h, res in results.items():
            print(f"\nГоризонт {h} мин:")
            print(f"  Оптимальный lookback: {res['best_lookback']} мин")
            print(f"  Параметры: {res['best_params']}")
            best_value = res['best_val_mae']
            koef = optimizer.get_descale_koef_for_error()
            print(f"  Лучший val MAE: {best_value:.6f} ({best_value*koef:.2f} м/с)")

        # Финальное обучение для каждого горизонта и сохранение моделей
        print("\n" + "="*70)
        print(f"ФИНАЛЬНОЕ ОБУЧЕНИЕ ДЛЯ {target}")
        print("="*70)
        for horizon in FORECAST_HORIZONS:
            # Получаем лучшие параметры для данного горизонта
            best_params = results[horizon]['best_params']

            # Обучаем финальную модель
            trainer, test_mae = train_final_model(
                data_paths=DATA_PATHS,
                target_column=target,
                exclude_columns=EXCLUDE,
                forecast_min=horizon,
                best_params=best_params,
                stride_minutes=OPTIMIZER_PARAMS['stride_minutes']
            )
        print(f"\n***** ЗАВЕРШЕНИЕ ДЛЯ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ: {target} *****\n\n")