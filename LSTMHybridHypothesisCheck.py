# LSTM Hypothesis Check

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
# КЛАСС ДЛЯ ОБРАТНОГО МАСШТАБИРОВАНИЯ
# ============================================================================

class DataScaler:
    """Класс для обратного масштабирования данных из [-1, 1] в физические значения"""

    def __init__(self):
        # Диапазоны для обратного масштабирования
        self.ranges = {
            "air_temperature": [-30, 30],
            "wind_speed_horizontal": [0, 25],
            "wind_speed_min": [0, 25],
            "wind_speed_max": [0, 25],
            "wind_speed_vertical": [-25, 25],
            "atmospheric_pressure": [730, 780],
            "relative_humidity": [30, 100],
            "dew_point_temperature": [-30, 30],
            "pressure_derivative": [-0.00026, 0.00026],
            "wind_forecast_three_hours": [0, 25]
        }

    def inverse_scale(self, scaled_values, column_name):
        """
        Обратное масштабирование из [-1, 1] в физические значения

        Parameters:
        -----------
        scaled_values : numpy array
            Масштабированные значения в диапазоне [-1, 1]
        column_name : str
            Название колонки для определения диапазона

        Returns:
        --------
        numpy array
            Физические значения в исходных единицах
        """
        if column_name not in self.ranges:
            raise ValueError(f"Неизвестная колонка: {column_name}. Доступны: {list(self.ranges.keys())}")

        min_val, max_val = self.ranges[column_name]
        # Обратное преобразование: value_physical = min_val + (scaled + 1) / 2 * (max_val - min_val)
        physical_values = min_val + (scaled_values + 1) / 2 * (max_val - min_val)

        return physical_values

    def inverse_scale_batch(self, scaled_values, column_names):
        """
        Обратное масштабирование для батча значений
        """
        if scaled_values.ndim == 1:
            physical_values = np.zeros_like(scaled_values)
            for i, col_name in enumerate(column_names):
                if i < len(scaled_values):
                    physical_values[i] = self.inverse_scale(scaled_values[i], col_name)
        else:
            physical_values = np.zeros_like(scaled_values)
            for i in range(scaled_values.shape[1]):
                if i < len(column_names):
                    physical_values[:, i] = self.inverse_scale(scaled_values[:, i], column_names[i])
                else:
                    physical_values[:, i] = scaled_values[:, i]

        return physical_values

    def compute_full_speed_physical(self, scaled_horizontal, scaled_vertical):
        """
        Вычисление полной скорости в физических единицах (м/с)
        """
        # Обратное масштабирование в физические единицы
        horizontal_phys = self.inverse_scale(scaled_horizontal, "wind_speed_horizontal")
        vertical_phys = self.inverse_scale(scaled_vertical, "wind_speed_vertical")

        # Вычисление полной скорости
        full_speed = np.sqrt(horizontal_phys**2 + vertical_phys**2)

        return full_speed

    def get_physical_range(self, column_name):
        """Возвращает физический диапазон для колонки"""
        return self.ranges.get(column_name, [0, 1])


# ============================================================================
# КЛАСС WINDSPEEDPREDICTOR
# ============================================================================

class WindSpeedPredictor:
    """
    Прогнозирование полной скорости ветра с учётом компонент и масштабирования

    Важно: wind_forecast_three_hours - это прогноз на 180 минут вперёд
    (downsampled глобальный прогноз)
    """

    def __init__(self, horizontal_col='wind_speed_horizontal', vertical_col='wind_speed_vertical',
                 global_forecast_col='wind_forecast_three_hours', forecast_horizon_minutes=180):
        self.horizontal_col = horizontal_col
        self.vertical_col = vertical_col
        self.global_col = global_forecast_col
        self.forecast_horizon_minutes = forecast_horizon_minutes  # 180 минут
        self.scaler = DataScaler()
        self.models = {}
        self.forecast_steps = None

    def prepare_data_for_components(self, all_data, feature_columns, frequency_minutes=1):
        """
        Подготовка данных для разных стратегий

        Важно: wind_forecast_three_hours в момент времени t - это прогноз на t+forecast_horizon_minutes
        """
        # Извлекаем компоненты (первые две колонки)
        horizontal_data = all_data[:, 0]  # wind_speed_horizontal
        vertical_data = all_data[:, 1]     # wind_speed_vertical

        # Другие локальные признаки (все кроме горизонтальной, вертикальной и глобального прогноза)
        other_features = all_data[:, 2:-1] if all_data.shape[1] > 3 else np.empty((all_data.shape[0], 0))

        # Глобальный прогноз полной скорости (последняя колонка) - прогноз на будущее
        global_forecast = all_data[:, -1:]

        # Для валидации вычисляем полную скорость в физических единицах
        full_speed_physical = self.scaler.compute_full_speed_physical(horizontal_data, vertical_data)

        # Количество шагов для горизонта прогноза
        self.forecast_steps = self.forecast_horizon_minutes // frequency_minutes

        print(f"\n" + "="*60)
        print("ИНФОРМАЦИЯ О ДАННЫХ:")
        print(f"  Горизонтальная компонента (u): {horizontal_data.shape}, диапазон [{horizontal_data.min():.3f}, {horizontal_data.max():.3f}]")
        print(f"  Вертикальная компонента (v): {vertical_data.shape}, диапазон [{vertical_data.min():.3f}, {vertical_data.max():.3f}]")
        print(f"  Другие локальные признаки: {other_features.shape[1] if other_features.size > 0 else 0}")
        print(f"  Глобальный прогноз: {global_forecast.shape[1]}, диапазон [{global_forecast.min():.3f}, {global_forecast.max():.3f}]")
        print(f"\n  ВАЖНО: wind_forecast_three_hours - это прогноз на {self.forecast_horizon_minutes} минут вперёд")
        print(f"  Это соответствует {self.forecast_steps} шагам при частоте {frequency_minutes} мин")
        print("="*60)

        return {
            'horizontal': horizontal_data,
            'vertical': vertical_data,
            'full_speed_physical': full_speed_physical,
            'local_features': other_features,
            'global_forecast': global_forecast  # прогноз на будущее
        }

    def create_sequences_for_components(self, data_dict, lookback_steps, forecast_steps, stride):
        """
        Создание последовательностей для прогнозирования компонент (только локальные данные)

        Вход: исторические значения горизонтальной и вертикальной компонент + другие признаки
        Выход: прогноз компонент через forecast_steps шагов
        """
        horizontal = data_dict['horizontal']
        vertical = data_dict['vertical']
        local_features = data_dict['local_features']

        # Объединяем входные признаки
        if local_features.shape[1] > 0:
            input_data = np.column_stack([horizontal, vertical, local_features])
        else:
            input_data = np.column_stack([horizontal, vertical])

        X, y_hor, y_ver = [], [], []

        for i in range(0, len(input_data) - lookback_steps - forecast_steps + 1, stride):
            # Вход: lookback_steps шагов истории
            X.append(input_data[i:i + lookback_steps])

            # Цель: прогноз на шаг forecast_steps вперёд
            target_idx = i + lookback_steps + forecast_steps - 1
            y_hor.append(horizontal[target_idx])
            y_ver.append(vertical[target_idx])

        print(f"  Создано последовательностей (локальные): {len(X)}")
        print(f"    Форма X: {np.array(X).shape if X else (0,)}")

        return np.array(X), np.array(y_hor), np.array(y_ver)

    def create_sequences_hybrid(self, data_dict, lookback_steps, forecast_steps, stride):
        """
        Создание последовательностей для гибридной модели

        Вход:
          - исторические значения горизонтальной и вертикальной компонент
          - другие локальные признаки
          - глобальный прогноз (который уже предсказывает через forecast_steps шагов)

        Выход: прогноз компонент через forecast_steps шагов

        Важно: global_forecast[t] предсказывает значение на момент t+forecast_steps
        """
        horizontal = data_dict['horizontal']
        vertical = data_dict['vertical']
        local_features = data_dict['local_features']
        global_forecast = data_dict['global_forecast'].flatten()

        X, y_hor, y_ver = [], [], []

        # Используем правильную индексацию с учётом горизонта прогноза
        for i in range(lookback_steps, len(horizontal) - forecast_steps, stride):
            # Исторические данные: от i-lookback_steps до i
            hist_start = i - lookback_steps
            hist_end = i

            # Собираем исторические данные
            if local_features.shape[1] > 0:
                hist_data = np.column_stack([
                    horizontal[hist_start:hist_end],
                    vertical[hist_start:hist_end],
                    local_features[hist_start:hist_end]
                ])
            else:
                hist_data = np.column_stack([
                    horizontal[hist_start:hist_end],
                    vertical[hist_start:hist_end]
                ])

            # Добавляем глобальный прогноз как дополнительный признак
            # В момент i у нас есть прогноз на i+forecast_steps
            # Добавляем его как константу на всех шагах истории
            global_forecast_value = global_forecast[i]
            global_feature = np.full((lookback_steps, 1), global_forecast_value)
            hist_data_with_global = np.column_stack([hist_data, global_feature])

            X.append(hist_data_with_global)

            # Цель: значение на момент i + forecast_steps
            target_idx = i + forecast_steps
            if target_idx < len(horizontal):
                y_hor.append(horizontal[target_idx])
                y_ver.append(vertical[target_idx])

        print(f"  Создано последовательностей (гибридные): {len(X)}")
        print(f"    Форма X: {np.array(X).shape if X else (0,)}")

        return np.array(X), np.array(y_hor), np.array(y_ver)

    def train_component_models(self, X_train, y_hor_train, y_ver_train,
                                X_val, y_hor_val, y_ver_val,
                                input_size, hidden_size=64, num_layers=2,
                                dropout_rate=0.3, learning_rate=0.001,
                                epochs=100, batch_size=32, verbose=True):
        """
        Обучение двух LSTM моделей для прогноза горизонтальной и вертикальной компонент
        """
        model_hor = LSTMModel(input_size, hidden_size, num_layers, dropout_rate, output_size=1)
        model_ver = LSTMModel(input_size, hidden_size, num_layers, dropout_rate, output_size=1)

        trainer_hor = LSTMTrainer(model_hor, learning_rate, weight_decay=1e-5)
        trainer_ver = LSTMTrainer(model_ver, learning_rate, weight_decay=1e-5)

        if verbose:
            print("\n  Обучение модели для горизонтальной компоненты...")
        history_hor = trainer_hor.train(
            X_train, y_hor_train, X_val, y_hor_val,
            epochs, batch_size, patience=15, verbose=verbose
        )

        if verbose:
            print("\n  Обучение модели для вертикальной компоненты...")
        history_ver = trainer_ver.train(
            X_train, y_ver_train, X_val, y_ver_val,
            epochs, batch_size, patience=15, verbose=verbose
        )

        return {
            'model_horizontal': model_hor,
            'model_vertical': model_ver,
            'trainer_horizontal': trainer_hor,
            'trainer_vertical': trainer_ver,
            'history_horizontal': history_hor,
            'history_vertical': history_ver
        }

    def predict_components(self, models_dict, X_test):
        """
        Прогнозирование компонент (возвращает масштабированные значения)
        """
        pred_horizontal = models_dict['trainer_horizontal'].predict(X_test)
        pred_vertical = models_dict['trainer_vertical'].predict(X_test)
        return pred_horizontal, pred_vertical

    def compute_physical_metrics(self, y_true_scaled_horizontal, y_true_scaled_vertical,
                                  y_pred_scaled_horizontal, y_pred_scaled_vertical):
        """
        Вычисление метрик в физических единицах (м/с)
        """
        # Обратное масштабирование компонент
        true_horizontal_phys = self.scaler.inverse_scale(y_true_scaled_horizontal, "wind_speed_horizontal")
        true_vertical_phys = self.scaler.inverse_scale(y_true_scaled_vertical, "wind_speed_vertical")
        pred_horizontal_phys = self.scaler.inverse_scale(y_pred_scaled_horizontal, "wind_speed_horizontal")
        pred_vertical_phys = self.scaler.inverse_scale(y_pred_scaled_vertical, "wind_speed_vertical")

        # Вычисление полной скорости
        true_full_phys = self.compute_full_speed_from_physical(true_horizontal_phys, true_vertical_phys)
        pred_full_phys = self.compute_full_speed_from_physical(pred_horizontal_phys, pred_vertical_phys)

        # Метрики
        mae = mean_absolute_error(true_full_phys, pred_full_phys)
        rmse = np.sqrt(mean_squared_error(true_full_phys, pred_full_phys))

        return {
            'mae': mae,
            'rmse': rmse,
            'true_full_physical': true_full_phys,
            'pred_full_physical': pred_full_phys,
            'true_horizontal_physical': true_horizontal_phys,
            'true_vertical_physical': true_vertical_phys,
            'pred_horizontal_physical': pred_horizontal_phys,
            'pred_vertical_physical': pred_vertical_phys
        }

    @staticmethod
    def compute_full_speed_from_physical(horizontal_phys, vertical_phys):
        """Вычисление полной скорости из физических значений компонент"""
        return np.sqrt(horizontal_phys**2 + vertical_phys**2)


# ============================================================================
# КЛАСС HYPOTHESISTESTERWITHPHYSICS
# ============================================================================

class HypothesisTesterWithPhysics:
    """Проверка гипотезы с учётом физической природы данных и глобального прогноза"""

    def __init__(self):
        self.predictor = WindSpeedPredictor()
        self.scaler = DataScaler()
        self.results = {}

    def run_experiment(self, all_data, feature_columns, lookback_minutes=60,
                      forecast_minutes=180, frequency_minutes=1, stride_minutes=60):
        """
        Запуск эксперимента для всех трёх стратегий

        Параметры:
        -----------
        all_data : numpy array
            Данные в порядке: [wind_speed_horizontal, wind_speed_vertical, другие_признаки..., wind_forecast_three_hours]
        feature_columns : list
            Список названий колонок
        lookback_minutes : int
            Размер истории в минутах (по умолчанию 60)
        forecast_minutes : int
            Горизонт прогноза в минутах (по умолчанию 180)
        frequency_minutes : int
            Частота данных в минутах (по умолчанию 1)
        stride_minutes : int
            Шаг сдвига окна в минутах (по умолчанию 60)
        """

        lookback_steps = lookback_minutes // frequency_minutes
        forecast_steps = forecast_minutes // frequency_minutes
        stride_steps = stride_minutes // frequency_minutes

        print("\n" + "="*80)
        print("ПРОВЕРКА ГИПОТЕЗЫ: Гибридный прогноз полной скорости ветра")
        print("="*80)
        print(f"\nПараметры эксперимента:")
        print(f"  История для прогноза: {lookback_minutes} минут ({lookback_steps} шагов)")
        print(f"  Горизонт прогноза: {forecast_minutes} минут ({forecast_steps} шагов)")
        print(f"  Частота данных: {frequency_minutes} минута")
        print(f"  Шаг сдвига окна: {stride_minutes} минут ({stride_steps} шагов)")

        # Подготовка данных
        data_dict = self.predictor.prepare_data_for_components(all_data, feature_columns, frequency_minutes)

        # ======================================================================
        # СТРАТЕГИЯ 1: ТОЛЬКО ЛОКАЛЬНЫЕ ДАННЫЕ
        # ======================================================================
        print("\n" + "="*70)
        print("СТРАТЕГИЯ 1: ТОЛЬКО ЛОКАЛЬНЫЕ ДАННЫЕ")
        print("Прогноз горизонтальной и вертикальной компонент → вычисление полной скорости")
        print("="*70)

        # Создание последовательностей
        X_local, y_hor_local, y_ver_local = self.predictor.create_sequences_for_components(
            data_dict, lookback_steps, forecast_steps, stride_steps
        )

        if len(X_local) == 0:
            raise ValueError("Недостаточно данных для создания последовательностей! Уменьшите lookback или forecast.")

        # Разделение на train/val/test с сохранением временного порядка
        n_samples = len(X_local)
        train_end = int(n_samples * 0.6)
        val_end = int(n_samples * 0.8)

        X_train_l, X_val_l, X_test_l = X_local[:train_end], X_local[train_end:val_end], X_local[val_end:]
        y_hor_train_l, y_hor_val_l, y_hor_test_l = y_hor_local[:train_end], y_hor_local[train_end:val_end], y_hor_local[val_end:]
        y_ver_train_l, y_ver_val_l, y_ver_test_l = y_ver_local[:train_end], y_ver_local[train_end:val_end], y_ver_local[val_end:]

        print(f"\nРазделение данных (локальная модель):")
        print(f"  Обучающая выборка: {len(X_train_l)}")
        print(f"  Валидационная выборка: {len(X_val_l)}")
        print(f"  Тестовая выборка: {len(X_test_l)}")

        # Нормализация входных данных (для улучшения сходимости LSTM)
        X_mean_l, X_std_l = X_train_l.mean(axis=(0, 1), keepdims=True), X_train_l.std(axis=(0, 1), keepdims=True)
        X_std_l = np.where(X_std_l < 1e-8, 1.0, X_std_l)

        X_train_l_norm = (X_train_l - X_mean_l) / X_std_l
        X_val_l_norm = (X_val_l - X_mean_l) / X_std_l
        X_test_l_norm = (X_test_l - X_mean_l) / X_std_l

        # Обучение локальной модели
        input_size = X_local.shape[2]
        local_models = self.predictor.train_component_models(
            X_train_l_norm, y_hor_train_l, y_ver_train_l,
            X_val_l_norm, y_hor_val_l, y_ver_val_l,
            input_size=input_size,
            verbose=True
        )

        # Прогноз (в масштабированных значениях)
        pred_hor_scaled_l, pred_ver_scaled_l = self.predictor.predict_components(local_models, X_test_l_norm)

        # Вычисление метрик в физических единицах
        metrics_local = self.predictor.compute_physical_metrics(
            y_hor_test_l, y_ver_test_l,
            pred_hor_scaled_l, pred_ver_scaled_l
        )

        self.results['local_only'] = {
            'mae': metrics_local['mae'],
            'rmse': metrics_local['rmse'],
            'y_true': metrics_local['true_full_physical'],
            'y_pred': metrics_local['pred_full_physical'],
            'pred_horizontal': metrics_local['pred_horizontal_physical'],
            'pred_vertical': metrics_local['pred_vertical_physical'],
            'true_horizontal': metrics_local['true_horizontal_physical'],
            'true_vertical': metrics_local['true_vertical_physical']
        }

        print(f"\n{'='*50}")
        print(f"РЕЗУЛЬТАТЫ ЛОКАЛЬНОЙ МОДЕЛИ (в м/с):")
        print(f"  MAE полной скорости: {metrics_local['mae']:.4f} м/с")
        print(f"  RMSE полной скорости: {metrics_local['rmse']:.4f} м/с")
        print(f"{'='*50}")

        # ======================================================================
        # СТРАТЕГИЯ 2: ТОЛЬКО ГЛОБАЛЬНЫЙ ПРОГНОЗ
        # ======================================================================
        print("\n" + "="*70)
        print("СТРАТЕГИЯ 2: ТОЛЬКО ГЛОБАЛЬНЫЙ ПРОГНОЗ")
        print("Использование wind_forecast_three_hours как готовый прогноз полной скорости")
        print(f"ВАЖНО: глобальный прогноз уже предсказывает на {forecast_minutes} минут вперёд")
        print("="*70)

        # Глобальный прогноз (масштабированный)
        global_forecast_scaled = data_dict['global_forecast'].flatten()
        full_speed_true_physical = data_dict['full_speed_physical']

        # Правильное сравнение: глобальный прогноз в момент t сравниваем с фактом в момент t+forecast_steps
        if len(global_forecast_scaled) > forecast_steps and len(full_speed_true_physical) > forecast_steps:
            y_pred_global_scaled = global_forecast_scaled[:-forecast_steps]
            y_true_global_physical = full_speed_true_physical[forecast_steps:]

            # Обратное масштабирование глобального прогноза
            y_pred_global_physical = self.scaler.inverse_scale(y_pred_global_scaled, "wind_forecast_three_hours")

            # Обрезаем до одинаковой длины
            min_len = min(len(y_true_global_physical), len(y_pred_global_physical))
            y_true_global_physical = y_true_global_physical[:min_len]
            y_pred_global_physical = y_pred_global_physical[:min_len]

            mae_global = mean_absolute_error(y_true_global_physical, y_pred_global_physical)
            rmse_global = np.sqrt(mean_squared_error(y_true_global_physical, y_pred_global_physical))

            self.results['global_only'] = {
                'mae': mae_global,
                'rmse': rmse_global,
                'y_true': y_true_global_physical,
                'y_pred': y_pred_global_physical
            }

            print(f"\n{'='*50}")
            print(f"РЕЗУЛЬТАТЫ ГЛОБАЛЬНОЙ МОДЕЛИ (в м/с):")
            print(f"  MAE полной скорости: {mae_global:.4f} м/с")
            print(f"  RMSE полной скорости: {rmse_global:.4f} м/с")
            print(f"  Сравнено точек: {len(y_true_global_physical)}")
            print(f"{'='*50}")
        else:
            raise ValueError(f"Недостаточно данных для глобальной модели. Нужно > {forecast_steps} точек")

        # ======================================================================
        # СТРАТЕГИЯ 3: ГИБРИДНЫЙ ПОДХОД
        # ======================================================================
        print("\n" + "="*70)
        print("СТРАТЕГИЯ 3: ГИБРИДНЫЙ ПОДХОД")
        print("Локальные данные + глобальный прогноз как дополнительный признак")
        print("Глобальный прогноз используется как информация о будущем")
        print("="*70)

        # Создаём последовательности с глобальным прогнозом как признаком
        X_hybrid, y_hor_hybrid, y_ver_hybrid = self.predictor.create_sequences_hybrid(
            data_dict, lookback_steps, forecast_steps, stride_steps
        )

        if len(X_hybrid) == 0:
            raise ValueError("Недостаточно данных для гибридной модели!")

        # Разделение
        n_samples_h = len(X_hybrid)
        train_end_h = int(n_samples_h * 0.6)
        val_end_h = int(n_samples_h * 0.8)

        X_train_h, X_val_h, X_test_h = X_hybrid[:train_end_h], X_hybrid[train_end_h:val_end_h], X_hybrid[val_end_h:]
        y_hor_train_h, y_hor_val_h, y_hor_test_h = y_hor_hybrid[:train_end_h], y_hor_hybrid[train_end_h:val_end_h], y_hor_hybrid[val_end_h:]
        y_ver_train_h, y_ver_val_h, y_ver_test_h = y_ver_hybrid[:train_end_h], y_ver_hybrid[train_end_h:val_end_h], y_ver_hybrid[val_end_h:]

        print(f"\nРазделение данных (гибридная модель):")
        print(f"  Обучающая выборка: {len(X_train_h)}")
        print(f"  Валидационная выборка: {len(X_val_h)}")
        print(f"  Тестовая выборка: {len(X_test_h)}")

        # Нормализация
        X_mean_h, X_std_h = X_train_h.mean(axis=(0, 1), keepdims=True), X_train_h.std(axis=(0, 1), keepdims=True)
        X_std_h = np.where(X_std_h < 1e-8, 1.0, X_std_h)

        X_train_h_norm = (X_train_h - X_mean_h) / X_std_h
        X_val_h_norm = (X_val_h - X_mean_h) / X_std_h
        X_test_h_norm = (X_test_h - X_mean_h) / X_std_h

        # Обучение гибридной модели
        input_size_hybrid = X_hybrid.shape[2]
        hybrid_models = self.predictor.train_component_models(
            X_train_h_norm, y_hor_train_h, y_ver_train_h,
            X_val_h_norm, y_hor_val_h, y_ver_val_h,
            input_size=input_size_hybrid,
            verbose=True
        )

        # Прогноз
        pred_hor_scaled_h, pred_ver_scaled_h = self.predictor.predict_components(hybrid_models, X_test_h_norm)

        # Вычисление метрик в физических единицах
        metrics_hybrid = self.predictor.compute_physical_metrics(
            y_hor_test_h, y_ver_test_h,
            pred_hor_scaled_h, pred_ver_scaled_h
        )

        self.results['hybrid'] = {
            'mae': metrics_hybrid['mae'],
            'rmse': metrics_hybrid['rmse'],
            'y_true': metrics_hybrid['true_full_physical'],
            'y_pred': metrics_hybrid['pred_full_physical'],
            'pred_horizontal': metrics_hybrid['pred_horizontal_physical'],
            'pred_vertical': metrics_hybrid['pred_vertical_physical'],
            'true_horizontal': metrics_hybrid['true_horizontal_physical'],
            'true_vertical': metrics_hybrid['true_vertical_physical']
        }

        print(f"\n{'='*50}")
        print(f"РЕЗУЛЬТАТЫ ГИБРИДНОЙ МОДЕЛИ (в м/с):")
        print(f"  MAE полной скорости: {metrics_hybrid['mae']:.4f} м/с")
        print(f"  RMSE полной скорости: {metrics_hybrid['rmse']:.4f} м/с")
        print(f"{'='*50}")

        return self.results

    def statistical_test(self):
        """Статистическая проверка гипотез на физических значениях"""

        # Получаем абсолютные ошибки для каждой модели в м/с
        errors = {
            'local': np.abs(self.results['local_only']['y_true'] - self.results['local_only']['y_pred']),
            'global': np.abs(self.results['global_only']['y_true'] - self.results['global_only']['y_pred']),
            'hybrid': np.abs(self.results['hybrid']['y_true'] - self.results['hybrid']['y_pred'])
        }

        # Убеждаемся, что все массивы одинаковой длины
        min_len = min(len(errors['local']), len(errors['global']), len(errors['hybrid']))
        for key in errors:
            errors[key] = errors[key][:min_len]

        print(f"\n  Сравнивается {min_len} точек для статистических тестов")

        # Bootstrap тест
        n_bootstrap = 5000
        n_samples = min_len

        bootstrap_diffs = {
            'hybrid_vs_local': [],
            'hybrid_vs_global': []
        }

        np.random.seed(42)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n_samples, n_samples, replace=True)

            diff_local = np.mean(errors['hybrid'][idx]) - np.mean(errors['local'][idx])
            diff_global = np.mean(errors['hybrid'][idx]) - np.mean(errors['global'][idx])

            bootstrap_diffs['hybrid_vs_local'].append(diff_local)
            bootstrap_diffs['hybrid_vs_global'].append(diff_global)

        # p-values (чем меньше, тем лучше для H1)
        p_local = np.mean(np.array(bootstrap_diffs['hybrid_vs_local']) >= 0)
        p_global = np.mean(np.array(bootstrap_diffs['hybrid_vs_global']) >= 0)

        # Доверительные интервалы
        ci_local = np.percentile(bootstrap_diffs['hybrid_vs_local'], [2.5, 97.5])
        ci_global = np.percentile(bootstrap_diffs['hybrid_vs_global'], [2.5, 97.5])

        # Парный t-тест
        t_local, p_ttest_local = stats.ttest_rel(errors['hybrid'], errors['local'], alternative='less')
        t_global, p_ttest_global = stats.ttest_rel(errors['hybrid'], errors['global'], alternative='less')

        # Тест Уилкоксона
        wilcoxon_local = stats.wilcoxon(errors['hybrid'], errors['local'], alternative='less')
        wilcoxon_global = stats.wilcoxon(errors['hybrid'], errors['global'], alternative='less')

        return {
            'bootstrap': {
                'hybrid_vs_local': {
                    'mean_diff': np.mean(bootstrap_diffs['hybrid_vs_local']),
                    'ci_95': ci_local,
                    'p_value': p_local
                },
                'hybrid_vs_global': {
                    'mean_diff': np.mean(bootstrap_diffs['hybrid_vs_global']),
                    'ci_95': ci_global,
                    'p_value': p_global
                }
            },
            'paired_ttest': {
                'hybrid_vs_local': {'t_stat': t_local, 'p_value': p_ttest_local},
                'hybrid_vs_global': {'t_stat': t_global, 'p_value': p_ttest_global}
            },
            'wilcoxon': {
                'hybrid_vs_local': {'statistic': wilcoxon_local.statistic, 'p_value': wilcoxon_local.pvalue},
                'hybrid_vs_global': {'statistic': wilcoxon_global.statistic, 'p_value': wilcoxon_global.pvalue}
            }
        }

    def plot_results(self):
        """Визуализация результатов в физических единицах (м/с)"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        configs = ['local_only', 'global_only', 'hybrid']
        names = ['Локальная\n(u→полная)', 'Глобальная\n(готовый прогноз)', 'Гибридная\n(u+v+глобальный)']
        mae_values = [self.results[c]['mae'] for c in configs]

        # MAE сравнение
        axes[0, 0].bar(names, mae_values, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[0, 0].set_title('Сравнение MAE полной скорости ветра', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('MAE (м/с)')
        axes[0, 0].grid(True, alpha=0.3)

        for i, v in enumerate(mae_values):
            axes[0, 0].text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')

        # Относительное улучшение
        improvement_local = (mae_values[0] - mae_values[2]) / mae_values[0] * 100 if mae_values[0] > 0 else 0
        improvement_global = (mae_values[1] - mae_values[2]) / mae_values[1] * 100 if mae_values[1] > 0 else 0

        colors = ['#27ae60' if improvement_local > 0 else '#c0392b',
                  '#27ae60' if improvement_global > 0 else '#c0392b']
        axes[0, 1].bar(['vs Локальная', 'vs Глобальная'], [improvement_local, improvement_global], color=colors)
        axes[0, 1].set_title('Относительное улучшение\nгибридной модели (%)', fontsize=12, fontweight='bold')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 1].grid(True, alpha=0.3)

        for i, v in enumerate([improvement_local, improvement_global]):
            axes[0, 1].text(i, v + (0.5 if v >= 0 else -2), f'{v:.1f}%', ha='center', fontweight='bold')

        # RMSE сравнение
        rmse_values = [self.results[c]['rmse'] for c in configs]
        axes[0, 2].bar(names, rmse_values, color=['#3498db', '#2ecc71', '#e74c3c'])
        axes[0, 2].set_title('Сравнение RMSE полной скорости ветра', fontsize=12, fontweight='bold')
        axes[0, 2].set_ylabel('RMSE (м/с)')
        axes[0, 2].grid(True, alpha=0.3)

        for i, v in enumerate(rmse_values):
            axes[0, 2].text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')

        # Графики предсказаний
        test_len = min(300, len(self.results['hybrid']['y_true']))

        for i, config in enumerate(configs):
            ax = axes[1, i]
            y_true = self.results[config]['y_true'][:test_len]
            y_pred = self.results[config]['y_pred'][:test_len]

            ax.plot(y_true, label='Факт', alpha=0.7, linewidth=1.5, color='blue')
            ax.plot(y_pred, label='Прогноз', alpha=0.7, linewidth=1.5, color='red', linestyle='--')
            ax.set_title(f'{names[i]} модель\nMAE={self.results[config]["mae"]:.3f} м/с', fontsize=10)
            ax.set_xlabel('Временной шаг')
            ax.set_ylabel('Полная скорость ветра (м/с)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Распределение ошибок
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        errors = {
            'Локальная': np.abs(self.results['local_only']['y_true'] - self.results['local_only']['y_pred']),
            'Глобальная': np.abs(self.results['global_only']['y_true'] - self.results['global_only']['y_pred']),
            'Гибридная': np.abs(self.results['hybrid']['y_true'] - self.results['hybrid']['y_pred'])
        }

        bp = ax2.boxplot(errors.values(), labels=errors.keys(), patch_artist=True)

        colors_box = ['#3498db', '#2ecc71', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_title('Распределение абсолютных ошибок прогноза полной скорости', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Абсолютная ошибка (м/с)')
        ax2.grid(True, alpha=0.3, axis='y')

        # Добавим статистику на график
        y_max = max([np.percentile(err, 95) for err in errors.values()])
        for i, (name, err) in enumerate(errors.items()):
            median = np.median(err)
            mean_val = np.mean(err)
            ax2.text(i + 1, y_max * 0.95, f'med={median:.3f}\nmean={mean_val:.3f}',
                    ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

        # График компонент для гибридной модели
        if 'true_horizontal' in self.results['hybrid']:
            fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8))

            test_len_comp = min(300, len(self.results['hybrid']['true_horizontal']))

            # Горизонтальная компонента
            axes3[0].plot(self.results['hybrid']['true_horizontal'][:test_len_comp],
                         label='Факт', alpha=0.7, linewidth=1.5, color='blue')
            axes3[0].plot(self.results['hybrid']['pred_horizontal'][:test_len_comp],
                         label='Прогноз', alpha=0.7, linewidth=1.5, color='red', linestyle='--')
            axes3[0].set_title('Горизонтальная компонента скорости ветра (гибридная модель)', fontsize=12, fontweight='bold')
            axes3[0].set_ylabel('Скорость (м/с)')
            axes3[0].legend()
            axes3[0].grid(True, alpha=0.3)

            # Вертикальная компонента
            axes3[1].plot(self.results['hybrid']['true_vertical'][:test_len_comp],
                         label='Факт', alpha=0.7, linewidth=1.5, color='blue')
            axes3[1].plot(self.results['hybrid']['pred_vertical'][:test_len_comp],
                         label='Прогноз', alpha=0.7, linewidth=1.5, color='red', linestyle='--')
            axes3[1].set_title('Вертикальная компонента скорости ветра (гибридная модель)', fontsize=12, fontweight='bold')
            axes3[1].set_xlabel('Временной шаг')
            axes3[1].set_ylabel('Скорость (м/с)')
            axes3[1].legend()
            axes3[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()


# ============================================================================
# ФУНКЦИЯ ДЛЯ ЗАПУСКА ЭКСПЕРИМЕНТА
# ============================================================================

def run_hypothesis_test(data_paths, target_column='wind_speed_horizontal'):
    """
    Запуск проверки гипотезы

    Параметры:
    -----------
    data_paths : list
        Список путей к CSV файлам
    target_column : str
        Название целевой колонки (для загрузчика)
    """

    print("="*80)
    print("ПРОВЕРКА ГИПОТЕЗЫ: Гибридный прогноз полной скорости ветра")
    print("="*80)
    print("\nФизическая постановка:")
    print("- Данные масштабированы в [-1, 1]")
    print("- Локальные данные: прогнозируем компоненты wind_speed_horizontal и wind_speed_vertical")
    print("- Глобальный прогноз: wind_forecast_three_hours (прогноз полной скорости на 3 часа вперёд)")
    print("- Гибрид: используем глобальный прогноз как дополнительный признак")
    print("\nВсе метрики вычисляются в физических единицах (м/с)")
    print("\nГипотезы:")
    print("H0: Гибридный подход НЕ улучшает точность прогноза полной скорости")
    print("H1: Гибридный подход статистически значимо улучшает точность\n")

    # Загрузка данных
    loader = MultiFileDataLoader(
        data_paths=data_paths,
        target_column=target_column,
        exclude_columns=['date']
    )
    loader.load_and_extract_segments()
    all_data = loader.get_all_data()
    feature_columns = loader.feature_columns

    print(f"\nЗагружено данных: {len(all_data)} строк")
    print(f"Количество признаков: {all_data.shape[1]}")
    print(f"Диапазон данных (масштабированных): [{all_data.min():.3f}, {all_data.max():.3f}]")

    # Запуск эксперимента
    tester = HypothesisTesterWithPhysics()
    results = tester.run_experiment(
        all_data=all_data,
        feature_columns=feature_columns,
        lookback_minutes=60,
        forecast_minutes=180,
        frequency_minutes=1,
        stride_minutes=60
    )

    # Статистические тесты
    print("\n" + "="*70)
    print("СТАТИСТИЧЕСКАЯ ПРОВЕРКА ГИПОТЕЗ (ошибки в м/с)")
    print("="*70)

    stats_results = tester.statistical_test()

    print("\n1. Bootstrap анализ (5000 итераций):")
    for comparison, data in stats_results['bootstrap'].items():
        print(f"\n   {comparison}:")
        print(f"     Средняя разница в MAE: {data['mean_diff']:.6f} м/с")
        print(f"     95% ДИ: [{data['ci_95'][0]:.6f}, {data['ci_95'][1]:.6f}] м/с")
        print(f"     p-value: {data['p_value']:.4f}")
        if data['p_value'] < 0.05:
            print(f"     → Статистически значимое улучшение (p < 0.05)")
        else:
            print(f"     → Нет статистически значимого улучшения")

    print("\n2. Парный t-тест:")
    for comparison, data in stats_results['paired_ttest'].items():
        print(f"\n   {comparison}:")
        print(f"     t-статистика: {data['t_stat']:.4f}")
        print(f"     p-value: {data['p_value']:.6f}")
        if data['p_value'] < 0.05:
            print(f"     → Отвергаем H0 (гибридная модель лучше)")
        else:
            print(f"     → Не отвергаем H0")

    print("\n3. Тест Уилкоксона (непараметрический):")
    for comparison, data in stats_results['wilcoxon'].items():
        print(f"\n   {comparison}:")
        print(f"     p-value: {data['p_value']:.6f}")
        if data['p_value'] < 0.05:
            print(f"     → Гибридная модель статистически лучше")
        else:
            print(f"     → Нет статистически значимых различий")

    # Визуализация
    tester.plot_results()

    # Итоговый вывод
    print("\n" + "="*70)
    print("ИТОГОВЫЙ ВЫВОД")
    print("="*70)

    hybrid_mae = results['hybrid']['mae']
    local_mae = results['local_only']['mae']
    global_mae = results['global_only']['mae']

    print(f"\nMAE полной скорости ветра (прогноз на 3 часа):")
    print(f"  Локальная модель (горизонтальная+вертикальная→полная): {local_mae:.4f} м/с")
    print(f"  Глобальная модель (wind_forecast_three_hours): {global_mae:.4f} м/с")
    print(f"  Гибридная модель: {hybrid_mae:.4f} м/с")

    p_value = stats_results['paired_ttest']['hybrid_vs_local']['p_value']

    if p_value < 0.05 and hybrid_mae < local_mae:
        improvement = (local_mae - hybrid_mae) / local_mae * 100
        print(f"\n✓✓✓ ГИПОТЕЗА ПРИНИМАЕТСЯ!")
        print(f"  Гибридный подход улучшает прогноз на {improvement:.2f}%")
        print(f"  Абсолютное улучшение: {(local_mae - hybrid_mae):.4f} м/с")
        print(f"  Статистическая значимость подтверждена (p={p_value:.6f})")
    elif hybrid_mae < local_mae:
        print(f"\n≈≈≈ ГИПОТЕЗА ЧАСТИЧНО ПОДТВЕРЖДАЕТСЯ")
        print(f"  Есть улучшение ({improvement:.2f}%), но без статистической значимости")
        print(f"  Рекомендуется увеличить объём данных")
    else:
        print(f"\n✗✗✗ ГИПОТЕЗА НЕ ПОДТВЕРЖДАЕТСЯ")
        print(f"  Гибридная модель не показала улучшения")

    return results, stats_results

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    data_paths = [
        "/content/winter_oblkom_27m.csv",
        "/content/spring_oblkom_27m.csv",
        "/content/summer_oblkom_26m.csv",
        "/content/autumn_oblkom_27m.csv"
    ]

    # Запускаем проверку гипотезы
    results, stats = run_physics_aware_hypothesis_test(data_paths)