# LSTM Test and Stability Check

!pip uninstall sympy -y
!pip install sympy==1.12

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_DYNAMO_DISABLE"] = "1"  # Отключаем torch._dynamo

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, wilcoxon
import gc
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
# ЗАГРУЗКА ДАННЫХ
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
        all_segments = []
        feature_columns_set = None
        for file_path in self.data_paths:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            if self.target_column not in df.columns:
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
# LSTM МОДЕЛЬ
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
        return out

class LSTMTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001, weight_decay: float = 1e-5):
        self.model = model
        self.model.to(DEVICE)
        self.criterion = nn.MSELoss()
        # Используем AdamW вместо Adam для лучшей стабильности
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=16, patience=15, verbose=False):
        X_train_t = torch.FloatTensor(X_train).to(DEVICE)
        y_train_t = torch.FloatTensor(y_train).to(DEVICE)
        X_val_t = torch.FloatTensor(X_val).to(DEVICE)
        y_val_t = torch.FloatTensor(y_val).to(DEVICE)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                # Добавляем clip_grad_norm для стабильности
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item() * batch_X.size(0)

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t).squeeze()
                val_loss = self.criterion(val_outputs, y_val_t).item()

            self.scheduler.step(val_loss)

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
        return self.model

    def predict(self, X):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(DEVICE)
        with torch.no_grad():
            return self.model(X_t).squeeze().cpu().numpy()

    def predict_with_uncertainty(self, X, n_dropout=30):
        self.model.train()
        predictions = []
        for _ in range(n_dropout):
            X_t = torch.FloatTensor(X).to(DEVICE)
            with torch.no_grad():
                pred = self.model(X_t).squeeze().cpu().numpy()
            predictions.append(pred)
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        return mean_pred, std_pred, predictions

# ============================================================================
# ОСНОВНЫЕ МЕТРИКИ
# ============================================================================
class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_persistence=None):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan

        r2 = r2_score(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        corr, _ = pearsonr(y_true, y_pred)
        std_ratio = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else np.nan

        metrics = {
            'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2,
            'Bias': bias, 'Pearson_Corr': corr, 'StdRatio': std_ratio
        }

        if y_pred_persistence is not None:
            mae_pers = mean_absolute_error(y_true, y_pred_persistence)
            metrics['Improvement_vs_Persistence_%'] = (mae_pers - mae) / mae_pers * 100

        return metrics

    @staticmethod
    def calculate_all_metrics(y_true, y_pred=None, ensemble_preds=None, persistence_pred=None):
        deterministic_metrics = {}
        if y_pred is not None:
            deterministic_metrics = ModelEvaluator.calculate_metrics(y_true, y_pred, persistence_pred)

        probabilistic_metrics = {}
        if ensemble_preds is not None:
            mean_pred = np.mean(ensemble_preds, axis=0)
            std_pred = np.std(ensemble_preds, axis=0)

            intervals_95 = {'lower': mean_pred - 1.96 * std_pred, 'upper': mean_pred + 1.96 * std_pred}

            widths_95 = intervals_95['upper'] - intervals_95['lower']

            probabilistic_metrics['Interval_Width_95%'] = np.mean(widths_95)
            probabilistic_metrics['Coverage_95%'] = np.mean((y_true >= intervals_95['lower']) & (y_true <= intervals_95['upper']))
            probabilistic_metrics['Sharpness'] = np.mean(std_pred)

        return {**deterministic_metrics, **probabilistic_metrics}

# ============================================================================
# АНАЛИЗ УСТОЙЧИВОСТИ
# ============================================================================
class RobustnessAnalyzer:
    @staticmethod
    def temporal_cross_validation(loader, horizon, best_params, n_splits=3):
        """Быстрая временная кросс-валидация"""
        scaled_data = loader.get_all_data()
        builder = TimeSeriesDatasetBuilder(best_params['lookback'], horizon, stride_minutes=60)
        X, y = builder.create_sequences(scaled_data, target_col_idx=0)

        if len(X) < 100:
            return np.nan, np.nan

        n = len(X)
        fold_size = n // n_splits
        cv_results = []

        for i in range(n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_splits - 1 else n

            X_test_fold = X[test_start:test_end]
            y_test_fold = y[test_start:test_end]
            X_train_fold = np.vstack([X[:test_start], X[test_end:]])
            y_train_fold = np.concatenate([y[:test_start], y[test_end:]])

            val_size = min(len(X_train_fold) // 5, 500)
            if val_size <= 0:
                val_size = max(1, len(X_train_fold) // 10)
            X_val_fold = X_train_fold[-val_size:]
            y_val_fold = y_train_fold[-val_size:]
            X_train_fold = X_train_fold[:-val_size] if len(X_train_fold) > val_size else X_train_fold
            y_train_fold = y_train_fold[:-val_size] if len(y_train_fold) > val_size else y_train_fold

            try:
                model = LSTMModel(
                    input_size=X_train_fold.shape[2],
                    hidden_size=best_params['units'],
                    num_layers=1,
                    dropout_rate=best_params['dropout']
                )
                trainer = LSTMTrainer(model, learning_rate=best_params['lr'],
                                      weight_decay=best_params['weight_decay'])
                trainer.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                             epochs=30, batch_size=best_params.get('batch_size', 16),
                             patience=10, verbose=False)

                y_pred = trainer.predict(X_test_fold)
                mae = mean_absolute_error(y_test_fold, y_pred)
                cv_results.append(mae)
            except:
                continue

        if len(cv_results) == 0:
            return np.nan, np.nan
        return np.mean(cv_results), np.std(cv_results)

# ============================================================================
# ТЕСТИРОВАНИЕ НА НОВЫХ ДАННЫХ
# ============================================================================
def test_on_new_data(trainer, new_data_paths, target_column, exclude_columns,
                     horizon, lookback, stride_minutes=60, scale=1.0):
    """Тестирование модели на новых данных"""
    print(f"\n  ТЕСТИРОВАНИЕ НА НОВЫХ ДАННЫХ (горизонт {horizon} мин)")

    loader = MultiFileDataLoader(
        new_data_paths, target_column, exclude_columns,
        min_segment_minutes=120, frequency_minutes=1
    )

    try:
        loader.load_and_extract_segments()
    except Exception as e:
        print(f"    Ошибка загрузки: {e}")
        return None

    if len(loader.continuous_segments) == 0:
        print("    Нет непрерывных сегментов!")
        return None

    scaled_data = loader.get_all_data()
    builder = TimeSeriesDatasetBuilder(lookback, horizon, stride_minutes=stride_minutes)
    X_new, y_new = builder.create_sequences(scaled_data, target_col_idx=0)

    if len(X_new) == 0:
        print("    Недостаточно данных!")
        return None

    print(f"    Последовательностей: {len(X_new)}")

    # ========== ПРОРЕЖИВАНИЕ ДО 3500 ==========
    MAX_SEQUENCES = 3500
    if len(X_new) > MAX_SEQUENCES:
        np.random.seed(42)  # <-- ДОБАВЬТЕ ЭТУ СТРОКУ для воспроизводимости
        indices = np.random.choice(len(X_new), MAX_SEQUENCES, replace=False)
        indices.sort()
        X_new = X_new[indices]
        y_new = y_new[indices]
        print(f"    Прорежено до {len(X_new)} последовательностей")
    # ==========================================

    mean_pred, std_pred, ensemble = trainer.predict_with_uncertainty(X_new, n_dropout=20)
    y_pred_persistence = X_new[:, -1, 0]

    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_all_metrics(
        y_true=y_new, y_pred=mean_pred,
        ensemble_preds=ensemble, persistence_pred=y_pred_persistence
    )

    print(f"    MAE: {metrics['MAE']*scale:.2f} м/с, R2: {metrics['R2']:.3f}")
    print(f"    Улучшение vs Persistence: {metrics.get('Improvement_vs_Persistence_%', 0):.1f}%")

    return {'metrics': metrics}

# ============================================================================
# ФИНАЛЬНОЕ ОБУЧЕНИЕ
# ============================================================================
def train_and_evaluate(data_paths, target_column, exclude_columns,
                       forecast_min, best_params, stride_minutes=120):
    """Обучение и оценка модели"""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    loader = MultiFileDataLoader(data_paths, target_column, exclude_columns, min_segment_minutes=360)
    loader.load_and_extract_segments()
    scaled_data = loader.get_all_data()

    lookback = best_params['lookback']
    builder = TimeSeriesDatasetBuilder(lookback, forecast_min, stride_minutes=stride_minutes)
    X, y = builder.create_sequences(scaled_data, target_col_idx=0)

    if len(X) == 0:
        raise ValueError(f"Нет данных для lookback={lookback}")

        # ========== ПРОРЕЖИВАНИЕ ДО 3500 ==========
    MAX_SEQUENCES = 3500
    if len(X) > MAX_SEQUENCES:
        indices = np.random.choice(len(X), MAX_SEQUENCES, replace=False)
        indices.sort()
        X = X[indices]
        y = y[indices]
        print(f"    Прорежено до {len(X)} последовательностей (было больше)")
    # ==========================================

    X_train, X_val, X_test, y_train, y_val, y_test = builder.temporal_train_test_split(X, y)

    if len(X_train) == 0:
        raise ValueError("Нет обучающих данных")

    print(f"\n  Данные: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    model = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=best_params['units'],
        num_layers=1,
        dropout_rate=best_params['dropout']
    )
    trainer = LSTMTrainer(model, learning_rate=best_params['lr'],
                          weight_decay=best_params['weight_decay'])
    trainer.train(X_train, y_train, X_val, y_val,
                  epochs=60, batch_size=best_params.get('batch_size', 16),
                  patience=10, verbose=False)

    scale = 12.5 if target_column == 'wind_speed_horizontal' else 25

    print("  Вычисление вероятностного прогноза...")
    mean_pred, std_pred, ensemble = trainer.predict_with_uncertainty(X_test, n_dropout=20)
    y_pred_persistence = X_test[:, -1, 0]

    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_all_metrics(
        y_true=y_test, y_pred=mean_pred,
        ensemble_preds=ensemble, persistence_pred=y_pred_persistence
    )

    print(f"\n  РЕЗУЛЬТАТЫ (горизонт {forecast_min} мин):")
    print(f"    MAE: {metrics['MAE']*scale:.2f} м/с")
    print(f"    R2: {metrics['R2']:.4f}")
    print(f"    Улучшение vs Persistence: {metrics.get('Improvement_vs_Persistence_%', 0):.1f}%")
    print(f"    Покрытие 95%: {metrics.get('Coverage_95%', 0)*100:.1f}%")

    # Анализ устойчивости
    print("\n  Анализ устойчивости (CV)...")
    robustness = RobustnessAnalyzer()
    cv_mean, cv_std = robustness.temporal_cross_validation(loader, forecast_min, best_params, n_splits=3)
    if not np.isnan(cv_mean):
        print(f"    CV MAE: {cv_mean*scale:.2f} ± {cv_std*scale:.2f} м/с")
    else:
        print(f"    CV MAE: N/A")

    return trainer, metrics, {'cv_mean': cv_mean, 'cv_std': cv_std}

# ============================================================================
# ОСНОВНОЙ ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    # Пути к данным
    TRAIN_DATA_PATHS = [
        "/content/winter_oblkom_27m.csv",
        "/content/spring_oblkom_27m.csv",
        "/content/summer_oblkom_26m.csv",
        "/content/autumn_oblkom_27m.csv"
    ]

    NEW_DATA_PATHS = [
        "/content/winter_imces_27m.csv",
        "/content/summer_imces_30m.csv",
        "/content/autumn_imces_27m.csv"
    ]

    # Исключаемые колонки
    EXCLUDE_COLUMNS = {
        "wind_speed_horizontal": ['date', 'air_temperature', 'dew_point_temperature', 'relative_humidity',
                       'pressure_derivative', 'atmospheric_pressure', 'wind_speed_min',
                       'wind_speed_vertical', 'sin_time_of_day', 'cos_time_of_day'],
        "wind_speed_vertical": ['date', 'wind_forecast_three_hours', 'atmospheric_pressure',
                       'relative_humidity', 'wind_speed_horizontal', 'wind_speed_min',
                       'wind_speed_max', 'cos_time_of_day', 'sin_day_of_year', 'cos_day_of_year', 'air_temperature']
    }

    FORECAST_HORIZONS = [30, 60, 120, 180]
    TARGETS = ["wind_speed_horizontal", "wind_speed_vertical"]

    # Гиперпараметры
    OPTIMAL_PARAMS = {
        "wind_speed_horizontal": {
            30: {'lookback': 600, 'units': 64, 'dropout': 0.25, 'lr': 0.00073365, 'batch_size': 8, 'weight_decay': 0.00029072},
            60: {'lookback': 240, 'units': 64, 'dropout': 0.35, 'lr': 0.00154788, 'batch_size': 16, 'weight_decay': 0.00054507},
            120: {'lookback': 240, 'units': 64, 'dropout': 0.3, 'lr': 0.00769386, 'batch_size': 16, 'weight_decay': 0.00055186},
            180: {'lookback': 240, 'units': 64, 'dropout': 0.35, 'lr': 0.00068053, 'batch_size': 8, 'weight_decay': 0.00002280}
        },
        "wind_speed_vertical": {
            30: {'lookback': 360, 'units': 64, 'dropout': 0.35, 'lr': 0.00104418, 'batch_size': 8, 'weight_decay': 0.00001195},
            60: {'lookback': 600, 'units': 32, 'dropout': 0.4, 'lr': 0.00259248, 'batch_size': 16, 'weight_decay': 0.00001406},
            120: {'lookback': 600, 'units': 32, 'dropout': 0.4, 'lr': 0.00259248, 'batch_size': 16, 'weight_decay': 0.00001406},
            180: {'lookback': 120, 'units': 64, 'dropout': 0.25, 'lr': 0.00027692, 'batch_size': 8, 'weight_decay': 0.00001292}
        }
    }

    RUN_TRAINING = True
    RUN_NEW_DATA_TEST = True

    all_results = {}

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"ЦЕЛЕВАЯ ПЕРЕМЕННАЯ: {target}")
        print(f"{'='*60}")

        exclude_for_target = EXCLUDE_COLUMNS.get(target, ['date'])

        for horizon in FORECAST_HORIZONS:
            print(f"\n--- Горизонт {horizon} мин ---")
            try:
                best_params = OPTIMAL_PARAMS[target][horizon]
                print(f"  Параметры: lookback={best_params['lookback']}, units={best_params['units']}, lr={best_params['lr']:.6f}")
            except KeyError:
                print(f"  Нет параметров для {target} {horizon} мин")
                continue

            if RUN_TRAINING:
                try:
                    trainer, metrics, cv_results = train_and_evaluate(
                        data_paths=TRAIN_DATA_PATHS,
                        target_column=target,
                        exclude_columns=exclude_for_target,
                        forecast_min=horizon,
                        best_params=best_params,
                        stride_minutes=120
                    )

                    result = {
                        'target': target,
                        'horizon': horizon,
                        'train_metrics': metrics,
                        'cv_mae_mean': cv_results['cv_mean'],
                        'cv_mae_std': cv_results['cv_std']
                    }

                    # Тестирование на новых данных (если включено)
                    if RUN_NEW_DATA_TEST and NEW_DATA_PATHS:
                        scale = 12.5 if target == 'wind_speed_horizontal' else 25
                        new_results = test_on_new_data(
                            trainer=trainer,
                            new_data_paths=NEW_DATA_PATHS,
                            target_column=target,
                            exclude_columns=exclude_for_target,
                            horizon=horizon,
                            lookback=best_params['lookback'],
                            stride_minutes=120,
                            scale=scale
                        )
                        if new_results:
                            result['new_data_metrics'] = new_results['metrics']

                    all_results[f"{target}_{horizon}"] = result

                except Exception as e:
                    print(f"  Ошибка: {e}")
                    continue

    # Сводная таблица
    print("\n\n" + "="*80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)

    if all_results:
        summary_data = []
        for key, result in all_results.items():
            target = result['target']
            horizon = result['horizon']
            scale = 12.5 if 'horizontal' in target else 25
            m = result['train_metrics']

            row = {
                'Target': target.replace('wind_speed_', '')[:15],
                'Hor': horizon,
                'MAE': f"{m['MAE']*scale:.2f}",
                'R2': f"{m['R2']:.3f}",
                'Corr': f"{m['Pearson_Corr']:.3f}",
                'Cov_95': f"{m.get('Coverage_95%', 0)*100:.0f}%",
                'Width': f"{m.get('Interval_Width_95%', 0)*scale:.2f}",
                'Improve': f"{m.get('Improvement_vs_Persistence_%', 0):.0f}%"
            }

            if not np.isnan(result['cv_mae_mean']):
                row['CV_MAE'] = f"{result['cv_mae_mean']*scale:.2f}±{result['cv_mae_std']*scale:.2f}"
            else:
                row['CV_MAE'] = "N/A"

            if 'new_data_metrics' in result:
                nd = result['new_data_metrics']
                row['New_MAE'] = f"{nd['MAE']*scale:.2f}"
                row['New_R2'] = f"{nd['R2']:.3f}"
                degrad = (nd['MAE'] - m['MAE']) / m['MAE'] * 100
                row['Degrad'] = f"{degrad:.0f}%"
            else:
                row['New_MAE'] = "N/A"
                row['New_R2'] = "N/A"
                row['Degrad'] = "N/A"

            summary_data.append(row)

        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values(['Target', 'Hor'])
        print("\n", df_summary.to_string(index=False))

        print("\n" + "="*80)
        print("ГОТОВО")
        print("="*80)
    else:
        print("\nНет результатов для отображения!")