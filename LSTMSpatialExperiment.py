# Spatial experiment

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import os
import gc
from glob import glob
from math import radians, sin, cos, sqrt, atan2

# Проверка CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Установка random seed
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ==================== ФУНКЦИИ ДЛЯ РАСЧЁТА РАССТОЯНИЙ В МЕТРАХ ====================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Вычисление расстояния между двумя точками на сфере (в метрах)"""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def find_nearest_station_meters(target_station, station_coords, all_stations):
    """Находит БЛИЖАЙШУЮ станцию к целевой (расстояние в метрах)"""
    target_lat, target_lon = station_coords[target_station]
    nearest_station = None
    min_distance = float('inf')
    for station in all_stations:
        if station != target_station:
            lat, lon = station_coords[station]
            dist = haversine_distance(target_lat, target_lon, lat, lon)
            if dist < min_distance:
                min_distance = dist
                nearest_station = station
    return nearest_station, min_distance

# ==================== ПАРАМЕТРЫ МАСШТАБИРОВАНИЯ ====================
WIND_HOR_MIN = 0.0
WIND_HOR_MAX = 25.0
WIND_VER_MIN = -25.0
WIND_VER_MAX = 25.0

def scale_horizontal_to_original(scaled_values):
    return (scaled_values + 1) / 2 * (WIND_HOR_MAX - WIND_HOR_MIN) + WIND_HOR_MIN

def scale_vertical_to_original(scaled_values):
    return (scaled_values + 1) / 2 * (WIND_VER_MAX - WIND_VER_MIN) + WIND_VER_MIN

def scale_horizontal_to_normalized(original_values):
    return (original_values - WIND_HOR_MIN) / (WIND_HOR_MAX - WIND_HOR_MIN) * 2 - 1

def scale_vertical_to_normalized(original_values):
    return (original_values - WIND_VER_MIN) / (WIND_VER_MAX - WIND_VER_MIN) * 2 - 1

# ==================== ПРИВЕДЕНИЕ ВЕТРА К ВЫСОТЕ ====================
def wind_speed_to_height(wind_speed, from_height, to_height, alpha=0.2):
    from_height = float(from_height)
    to_height = float(to_height)
    if from_height == to_height:
        return wind_speed
    return wind_speed * (to_height / from_height) ** alpha

# ==================== 1. ЗАГРУЗКА ДАННЫХ ====================

def load_multiple_csv_files(file_patterns, station_name, station_height, target_height=10.0):
    all_dfs = []
    for pattern in file_patterns:
        if '*' in pattern or '?' in pattern:
            files = glob(pattern)
        else:
            files = [pattern] if os.path.exists(pattern) else []
        for file_path in files:
            print(f"    Загрузка файла: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError(f"Не найдено файлов для станции {station_name}")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    combined_df = combined_df.drop_duplicates(subset=['date'])

    # Горизонтальная скорость
    wind_hor_scaled = combined_df['wind_speed_horizontal'].values
    wind_hor_original = scale_horizontal_to_original(wind_hor_scaled)
    wind_hor_corrected = wind_speed_to_height(wind_hor_original, station_height, target_height)
    wind_hor_corrected_scaled = scale_horizontal_to_normalized(wind_hor_corrected)

    # Вертикальная скорость
    wind_ver_scaled = combined_df['wind_speed_vertical'].values
    wind_ver_original = scale_vertical_to_original(wind_ver_scaled)
    wind_ver_corrected_scaled = wind_ver_scaled

    print(f"    Всего записей: {len(combined_df)}")
    print(f"    Диапазон дат: {combined_df['date'].min()} - {combined_df['date'].max()}")

    return {
        'df': combined_df,
        'wind_hor_original': wind_hor_original,
        'wind_hor_corrected': wind_hor_corrected,
        'wind_hor_scaled': wind_hor_scaled,
        'wind_hor_corrected_scaled': wind_hor_corrected_scaled,
        'wind_ver_scaled': wind_ver_scaled,
        'wind_ver_original': wind_ver_original,
        'timestamps': combined_df['date'].values,
        'name': station_name,
        'height': station_height
    }

def create_sequences_from_single_station(station_data, lookback_minutes=360, forecast_minutes=180,
                                         stride_minutes=60, component='horizontal'):
    """
    Создание последовательностей для ОДНОЙ станции.

    Параметры:
        lookback_minutes: история в минутах (по умолчанию 360 минут = 6 часов)
        forecast_minutes: прогноз в минутах (по умолчанию 180 минут = 3 часа)
        stride_minutes: шаг между примерами в минутах (по умолчанию 60 минут = 1 час)
        component: 'horizontal' или 'vertical'

    ВНИМАНИЕ: Данные поминутные, поэтому lookback, forecast, stride - это минуты.
    """
    if component == 'horizontal':
        wind_speed = station_data['wind_hor_corrected_scaled']
    else:
        wind_speed = station_data['wind_ver_scaled']

    timestamps = station_data['timestamps']

    # Преобразуем минуты в шаги (1 шаг = 1 минута)
    lookback_steps = lookback_minutes
    forecast_steps = forecast_minutes
    stride_steps = stride_minutes

    print(f"    Параметры: lookback={lookback_steps} мин ({lookback_steps/60:.1f} ч), "
          f"forecast={forecast_steps} мин ({forecast_steps/60:.1f} ч), "
          f"stride={stride_steps} мин ({stride_steps/60:.1f} ч)")

    # Находим непрерывные сегменты
    continuous_segments = []
    current_segment = []
    current_timestamps = []

    for i, (ws, ts) in enumerate(zip(wind_speed, timestamps)):
        if pd.isna(ws):
            if len(current_segment) >= lookback_steps + forecast_steps:
                continuous_segments.append((current_segment, current_timestamps))
            current_segment = []
            current_timestamps = []
        else:
            current_segment.append(ws)
            current_timestamps.append(ts)

    if len(current_segment) >= lookback_steps + forecast_steps:
        continuous_segments.append((current_segment, current_timestamps))

    print(f"    Найдено {len(continuous_segments)} непрерывных сегментов")

    X = []
    y = []
    ts_list = []

    for segment, segment_ts in continuous_segments:
        n_possible = len(segment) - lookback_steps - forecast_steps + 1
        for i in range(0, n_possible, stride_steps):
            X_seq = segment[i:i+lookback_steps]
            # Предсказываем значение через forecast_steps минут
            y_val = segment[i + lookback_steps + forecast_steps - 1]
            ts_val = segment_ts[i + lookback_steps + forecast_steps - 1]

            X.append(X_seq)
            y.append(y_val)
            ts_list.append(ts_val)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    print(f"    Создано {len(X)} последовательностей (прогноз на {forecast_minutes} мин = {forecast_minutes/60:.1f} ч)")

    return X, y, ts_list

# ==================== LSTM МОДЕЛЬ ====================

class HeightAwareLSTM(nn.Module):
    """LSTM, который учитывает разницу высот между станциями"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1, dropout=0.2):
        super(HeightAwareLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.height_encoder = nn.Linear(1, hidden_size // 2)
        self.fc = nn.Linear(hidden_size + hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, height_diff):
        lstm_out, _ = self.lstm(x)
        temporal_features = lstm_out[:, -1, :]
        height_features = self.height_encoder(height_diff)
        height_features = self.relu(height_features)
        combined = torch.cat([temporal_features, height_features], dim=1)
        combined = self.dropout(combined)
        output = self.fc(combined)
        return output

def train_lstm_with_heights(model, X_train, y_train, height_diffs_train,
                            X_val, y_val, height_diffs_val,
                            epochs=50, lr=0.001, batch_size=64):
    model = model.to(device)

    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    height_diffs_train_t = torch.FloatTensor(height_diffs_train).to(device)

    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    height_diffs_val_t = torch.FloatTensor(height_diffs_val).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t, height_diffs_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y, batch_h in train_loader:
            optimizer.zero_grad()
            output = model(batch_x, batch_h).squeeze()
            loss = criterion(output, batch_y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t, height_diffs_val_t).squeeze()
            val_loss = criterion(val_pred, y_val_t.squeeze())

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}')

    if best_model_state:
        model.load_state_dict(best_model_state)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return model.cpu()

# ==================== ОСНОВНОЙ ЭКСПЕРИМЕНТ ====================

def run_experiment_for_component(station_files_config, station_coords, station_heights,
                                 lookback_minutes=360, forecast_minutes=180, stride_minutes=60,
                                 component='horizontal'):

    print(f"\n{'='*80}")
    print(f"ЭКСПЕРИМЕНТ ДЛЯ {component.upper()} КОМПОНЕНТЫ ВЕТРА")
    print(f"Прогноз на {forecast_minutes} минут ({forecast_minutes/60:.1f} часа) вперёд")
    print(f"{'='*80}")

    # Загрузка данных
    print("\nЗагрузка данных...")
    stations_raw = {}
    for name in station_files_config.keys():
        file_patterns = station_files_config[name]
        print(f"\nЗагрузка станции {name}:")
        stations_raw[name] = load_multiple_csv_files(file_patterns, name, station_heights[name], target_height=10.0)

    # Создание последовательностей
    print("\nСоздание последовательностей...")
    all_sequences = {}
    for name, data in stations_raw.items():
        print(f"\nСтанция {name}:")
        X, y, ts = create_sequences_from_single_station(
            data, lookback_minutes, forecast_minutes, stride_minutes, component
        )
        all_sequences[name] = {'X': X, 'y': y, 'timestamps': ts, 'height': station_heights[name]}

    station_names = list(station_files_config.keys())
    results = {}

    # Для каждой станции как целевой
    for target_station in station_names:
        print(f"\n{'='*70}")
        print(f"Целевая станция: {target_station}")
        print(f"Высота датчика: {station_heights[target_station]} м над землёй")
        print(f"{'='*70}")

        # Находим ближайшую станцию
        nearest_station, nearest_distance_m = find_nearest_station_meters(target_station, station_coords, station_names)
        nearest_height_diff = station_heights[nearest_station] - station_heights[target_station]

        print(f"\nБлижайшая станция: {nearest_station}")
        print(f"  Расстояние: {nearest_distance_m:.0f} м ({nearest_distance_m/1000:.2f} км)")
        print(f"  Разница высот датчиков: {nearest_height_diff:+.1f} м")

        # Данные целевой станции
        target_X = all_sequences[target_station]['X']
        target_y = all_sequences[target_station]['y']
        n_samples = len(target_X)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        test_indices = list(range(train_size + val_size, n_samples))

        # ===== BASELINE: обучение на собственной истории =====
        print(f"\n--- Baseline: LSTM на истории этой станции ---")
        print(f"  Всего: {n_samples}, Train: {train_size}, Val: {val_size}, Test: {len(test_indices)}")

        baseline_train_X = target_X[:train_size]
        baseline_train_y = target_y[:train_size]
        baseline_val_X = target_X[train_size:train_size+val_size]
        baseline_val_y = target_y[train_size:train_size+val_size]

        if len(baseline_train_X.shape) == 2:
            baseline_train_X = baseline_train_X.reshape(-1, lookback_minutes, 1)
            baseline_val_X = baseline_val_X.reshape(-1, lookback_minutes, 1)

        baseline_model = HeightAwareLSTM(input_size=1, hidden_size=64, num_layers=1, output_size=1, dropout=0.2)

        baseline_height_diffs_train = np.zeros((len(baseline_train_X), 1))
        baseline_height_diffs_val = np.zeros((len(baseline_val_X), 1))

        baseline_model = train_lstm_with_heights(baseline_model, baseline_train_X, baseline_train_y,
                                                baseline_height_diffs_train, baseline_val_X, baseline_val_y,
                                                baseline_height_diffs_val, epochs=50)

        baseline_test_X = target_X[test_indices]
        baseline_test_y = target_y[test_indices]

        if len(baseline_test_X.shape) == 2:
            baseline_test_X = baseline_test_X.reshape(-1, lookback_minutes, 1)

        baseline_model.eval()
        baseline_model = baseline_model.to(device)
        baseline_height_diffs_test = np.zeros((len(baseline_test_X), 1))

        with torch.no_grad():
            baseline_pred_scaled = baseline_model(torch.FloatTensor(baseline_test_X).to(device),
                                                  torch.FloatTensor(baseline_height_diffs_test).to(device)).cpu().numpy()

        if component == 'horizontal':
            baseline_pred = scale_horizontal_to_original(baseline_pred_scaled.flatten())
            baseline_actual = scale_horizontal_to_original(baseline_test_y.flatten())
        else:
            baseline_pred = scale_vertical_to_original(baseline_pred_scaled.flatten())
            baseline_actual = scale_vertical_to_original(baseline_test_y.flatten())

        baseline_mae = mean_absolute_error(baseline_actual, baseline_pred)
        baseline_rmse = np.sqrt(mean_squared_error(baseline_actual, baseline_pred))
        baseline_r2 = r2_score(baseline_actual, baseline_pred)

        print(f"  Baseline MAE: {baseline_mae:.4f} м/с, RMSE: {baseline_rmse:.4f} м/с, R²: {baseline_r2:.4f}")

        # ===== TRANSFER: перенос с ближайшей станции =====
        print(f"\n--- Transfer: перенос с ближайшей станции {nearest_station} ---")

        source_X = all_sequences[nearest_station]['X']
        source_y = all_sequences[nearest_station]['y']
        n_source = len(source_X)
        train_size_source = int(0.7 * n_source)
        val_size_source = int(0.15 * n_source)

        print(f"  Источник: всего {n_source}, Train: {train_size_source}, Val: {val_size_source}")

        transfer_train_X = source_X[:train_size_source]
        transfer_train_y = source_y[:train_size_source]
        transfer_val_X = source_X[train_size_source:train_size_source+val_size_source]
        transfer_val_y = source_y[train_size_source:train_size_source+val_size_source]

        if len(transfer_train_X.shape) == 2:
            transfer_train_X = transfer_train_X.reshape(-1, lookback_minutes, 1)
            transfer_val_X = transfer_val_X.reshape(-1, lookback_minutes, 1)

        transfer_height_diffs_train = np.full((len(transfer_train_X), 1), float(nearest_height_diff))
        transfer_height_diffs_val = np.full((len(transfer_val_X), 1), float(nearest_height_diff))

        transfer_model = HeightAwareLSTM(input_size=1, hidden_size=64, num_layers=1, output_size=1, dropout=0.2)
        transfer_model = train_lstm_with_heights(transfer_model, transfer_train_X, transfer_train_y,
                                                transfer_height_diffs_train, transfer_val_X, transfer_val_y,
                                                transfer_height_diffs_val, epochs=50)

        transfer_test_X = target_X[test_indices]
        transfer_test_y = target_y[test_indices]

        if len(transfer_test_X.shape) == 2:
            transfer_test_X = transfer_test_X.reshape(-1, lookback_minutes, 1)

        transfer_model.eval()
        transfer_model = transfer_model.to(device)
        transfer_height_diffs_test = np.full((len(transfer_test_X), 1), float(nearest_height_diff))

        with torch.no_grad():
            transfer_pred_scaled = transfer_model(torch.FloatTensor(transfer_test_X).to(device),
                                                  torch.FloatTensor(transfer_height_diffs_test).to(device)).cpu().numpy()

        if component == 'horizontal':
            transfer_pred = scale_horizontal_to_original(transfer_pred_scaled.flatten())
            transfer_actual = scale_horizontal_to_original(transfer_test_y.flatten())
        else:
            transfer_pred = scale_vertical_to_original(transfer_pred_scaled.flatten())
            transfer_actual = scale_vertical_to_original(transfer_test_y.flatten())

        transfer_mae = mean_absolute_error(transfer_actual, transfer_pred)
        transfer_rmse = np.sqrt(mean_squared_error(transfer_actual, transfer_pred))
        transfer_r2 = r2_score(transfer_actual, transfer_pred)

        print(f"  Transfer MAE: {transfer_mae:.4f} м/с, RMSE: {transfer_rmse:.4f} м/с, R²: {transfer_r2:.4f}")

        # ===== СРАВНЕНИЕ (проверка гипотезы) =====
        diff = transfer_mae - baseline_mae
        relative_diff = diff / baseline_mae * 100

        print(f"\n  Проверка гипотезы:")
        print(f"    Baseline MAE: {baseline_mae:.4f} м/с")
        print(f"    Transfer MAE: {transfer_mae:.4f} м/с")
        print(f"    Абсолютная разница: {diff:+.4f} м/с")
        print(f"    Относительная разница: {relative_diff:+.1f}%")

        if abs(relative_diff) < 20:
            print(f"    ✓ СОПОСТАВИМО (разница < 20%)")
        else:
            print(f"    ✗ НЕ СОПОСТАВИМО (разница >= 20%)")

        results[target_station] = {
            'baseline': {'MAE': baseline_mae, 'RMSE': baseline_rmse, 'R2': baseline_r2},
            'transfer': {
                'MAE': transfer_mae,
                'RMSE': transfer_rmse,
                'R2': transfer_r2,
                'source_station': nearest_station,
                'distance_m': nearest_distance_m,
                'distance_km': nearest_distance_m / 1000,
                'height_diff': nearest_height_diff
            },
            'diff': diff,
            'relative_diff': relative_diff,
            'comparable': abs(relative_diff) < 20
        }

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    return results

# ==================== ПРОВЕРКА ГИПОТЕЗЫ ====================

def test_hypothesis(results, component_name, forecast_minutes=180):
    """Проверка гипотезы о сопоставимости переноса обучения"""

    print(f"\n{'='*80}")
    print(f"ПРОВЕРКА ГИПОТЕЗЫ ДЛЯ {component_name.upper()} КОМПОНЕНТЫ")
    print(f"{'='*80}")
    print("Гипотеза: прогноз скорости ветра в локации, не имеющей собственной истории,")
    print("может быть выполнен с точностью, СОПОСТАВИМОЙ с точностью прогноза")
    print("для станций, располагающих репрезентативным историческим рядом,")
    print("путем переноса знаний от обученной модели с использованием данных окружающих станций.")
    print(f"{'='*80}")

    stations = list(results.keys())
    n_stations = len(stations)

    baseline_maes = [results[s]['baseline']['MAE'] for s in stations]
    transfer_maes = [results[s]['transfer']['MAE'] for s in stations]
    relative_diffs = [results[s]['relative_diff'] for s in stations]
    comparable_stations = [s for s in stations if results[s]['comparable']]

    mean_baseline = np.mean(baseline_maes)
    mean_transfer = np.mean(transfer_maes)
    mean_relative_diff = np.mean(relative_diffs)
    median_relative_diff = np.median(relative_diffs)

    comparable_count = len(comparable_stations)
    comparable_ratio = comparable_count / n_stations

    print(f"\n--- РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ---")
    print(f"  Количество станций: {n_stations}")
    print(f"  Горизонт прогноза: {forecast_minutes} мин ({forecast_minutes/60:.1f} ч)")
    print(f"  Средняя MAE Baseline:  {mean_baseline:.4f} м/с")
    print(f"  Средняя MAE Transfer:  {mean_transfer:.4f} м/с")
    print(f"  Средняя относительная разница: {mean_relative_diff:+.1f}%")
    print(f"  Медианная относительная разница: {median_relative_diff:+.1f}%")

    print(f"\n--- СОПОСТАВИМОСТЬ ПО СТАНЦИЯМ ---")
    print(f"  Сопоставимо (разница < 20%): {comparable_count} из {n_stations} ({comparable_ratio*100:.0f}%)")

    for station in stations:
        status = "✓ СОПОСТАВИМО" if results[station]['comparable'] else "✗ НЕ СОПОСТАВИМО"
        print(f"    {station}: {status} (разница {results[station]['relative_diff']:+.1f}%)")

    print(f"\n--- ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ---")
    print(f"{'Целевая':<16} {'Источник':<16} {'Расст, км':<10} {'Перепад':<10} {'Baseline':<8} {'Transfer':<8} {'Разница':<10} {'Сопоставимо':<12}")
    print("-" * 100)

    for station in stations:
        dist_km = results[station]['transfer']['distance_km']
        height_diff = results[station]['transfer']['height_diff']
        baseline = results[station]['baseline']['MAE']
        transfer = results[station]['transfer']['MAE']
        diff = results[station]['diff']
        comparable = "✓" if results[station]['comparable'] else "✗"
        source = results[station]['transfer']['source_station']
        print(f"{station:<16} {source:<16} {dist_km:<10.2f} {height_diff:+8.1f}м {baseline:<8.4f} {transfer:<8.4f} {diff:+8.4f} {comparable:<12}")

    # Статистический тест
    t_stat, p_value = ttest_rel(baseline_maes, transfer_maes)
    print(f"\n--- СТАТИСТИЧЕСКИЙ ТЕСТ (парный t-test) ---")
    print(f"  t = {t_stat:.4f}, p = {p_value:.4f}")

    if p_value < 0.05:
        print("  Различия статистически ЗНАЧИМЫ (p < 0.05)")
    else:
        print("  Различия статистически НЕ ЗНАЧИМЫ (p ≥ 0.05)")

    # Итоговый вывод
    print("\n" + "="*80)
    print("ВЫВОД ПО ГИПОТЕЗЕ")
    print("="*80)

    if comparable_ratio >= 0.5:
        print(f"✅ ГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ")
        print(f"   Для {comparable_count} из {n_stations} станций ({comparable_ratio*100:.0f}%)")
        print(f"   перенос обучения с ближайшей станции даёт точность, сопоставимую")
        print(f"   с обучением на собственной истории (разница MAE < 20%).")
    else:
        print(f"❌ ГИПОТЕЗА НЕ ПОДТВЕРЖДАЕТСЯ")
        print(f"   Только для {comparable_count} из {n_stations} станций ({comparable_ratio*100:.0f}%)")
        print(f"   перенос обучения с ближайшей станции НЕ обеспечивает")
        print(f"   сопоставимую точность (разница MAE > 20%).")

    print(f"\nСтатистика:")
    print(f"  Средняя MAE Baseline:  {mean_baseline:.4f} м/с")
    print(f"  Средняя MAE Transfer:  {mean_transfer:.4f} м/с")
    print(f"  Средняя разница:       {mean_transfer - mean_baseline:+.4f} м/с ({mean_relative_diff:+.1f}%)")
    print(f"  p-value: {p_value:.4f}")

    return {
        'mean_baseline': mean_baseline,
        'mean_transfer': mean_transfer,
        'comparable_ratio': comparable_ratio,
        'p_value': p_value,
        'hypothesis_confirmed': comparable_ratio >= 0.5
    }

# ==================== ВИЗУАЛИЗАЦИЯ ====================

def visualize_results(results_hor, results_ver, forecast_minutes=180):
    stations = list(results_hor.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Сравнение MAE (горизонтальная)
    ax1 = axes[0]
    x = np.arange(len(stations))
    width = 0.35
    baseline_hor = [results_hor[s]['baseline']['MAE'] for s in stations]
    transfer_hor = [results_hor[s]['transfer']['MAE'] for s in stations]

    ax1.bar(x - width/2, baseline_hor, width, label='Baseline (своя история)', color='#2ecc71', alpha=0.7)
    ax1.bar(x + width/2, transfer_hor, width, label='Transfer (ближайшая)', color='#e74c3c', alpha=0.7)
    ax1.set_xlabel('Станция')
    ax1.set_ylabel('MAE (м/с)')
    ax1.set_title(f'Горизонтальная компонента\n(прогноз на {forecast_minutes/60:.0f} ч)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stations, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for i, (b, t) in enumerate(zip(baseline_hor, transfer_hor)):
        ax1.text(i - width/2, b + 0.01, f'{b:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, t + 0.01, f'{t:.3f}', ha='center', va='bottom', fontsize=8)

    # 2. Сравнение MAE (вертикальная)
    ax2 = axes[1]
    baseline_ver = [results_ver[s]['baseline']['MAE'] for s in stations]
    transfer_ver = [results_ver[s]['transfer']['MAE'] for s in stations]

    ax2.bar(x - width/2, baseline_ver, width, label='Baseline (своя история)', color='#2ecc71', alpha=0.7)
    ax2.bar(x + width/2, transfer_ver, width, label='Transfer (ближайшая)', color='#e74c3c', alpha=0.7)
    ax2.set_xlabel('Станция')
    ax2.set_ylabel('MAE (м/с)')
    ax2.set_title(f'Вертикальная компонента\n(прогноз на {forecast_minutes/60:.0f} ч)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stations, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for i, (b, t) in enumerate(zip(baseline_ver, transfer_ver)):
        ax2.text(i - width/2, b + 0.01, f'{b:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width/2, t + 0.01, f'{t:.3f}', ha='center', va='bottom', fontsize=8)

    # 3. Относительная разница
    ax3 = axes[2]
    rel_diffs_hor = [results_hor[s]['relative_diff'] for s in stations]
    colors = ['#2ecc71' if abs(rd) < 20 else '#e74c3c' for rd in rel_diffs_hor]
    bars = ax3.bar(stations, rel_diffs_hor, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axhline(y=20, color='red', linestyle='--', linewidth=1.5, label='Порог 20%')
    ax3.axhline(y=-20, color='red', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Станция')
    ax3.set_ylabel('Относительная разница (%)')
    ax3.set_title('Transfer vs Baseline\n(отрицательное = Transfer лучше)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    for bar, value in zip(bars, rel_diffs_hor):
        ax3.text(bar.get_x() + bar.get_width()/2, value + (3 if value >= 0 else -10),
                f'{value:.1f}%', ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)

    plt.tight_layout()
    plt.show()

# ==================== ЗАПУСК ====================

def main():
    station_files_config = {
        'kinzjar_10m': ["autumn_kinzjar_10m.csv", "summer_kinzjar_10m.csv", "winter_kinzjar_10m.csv"],
        'kireevsk_10m': ["autumn_kireevsk_10m.csv", "summer_kireevsk_10m.csv", "winter_kireevsk_10m.csv"],
        'imces_27m': ["autumn_imces_27m.csv", "summer_imces_30m.csv", "winter_imces_27m.csv"],
        'oblkom_27m': ["autumn_oblkom_27m.csv", "summer_oblkom_26m.csv", "winter_oblkom_27m.csv"],
    }

    station_coords = {
        'kinzjar_10m': (57.6217, 82.3392),
        'kireevsk_10m': (56.4150, 84.0678),
        'imces_27m': (56.4752778, 85.0544444),
        'oblkom_27m': (56.4672222, 84.9575),
    }

    # ВЫСОТЫ ДАТЧИКОВ НАД ЗЕМЛЁЙ (исходя из названия станции)
    station_heights = {
        'kinzjar_10m': 10.0,   # 10-метровая мачта
        'kireevsk_10m': 10.0,  # 10-метровая мачта
        'imces_27m': 27.0,     # 27-метровая мачта
        'oblkom_27m': 27.0,    # 27-метровая мачта
    }

    # ПАРАМЕТРЫ ЭКСПЕРИМЕНТА (для поминутных данных)
    lookback_minutes = 360      # часов истории
    forecast_minutes = 180      # прогноз на 3 часа (180 минут)
    stride_minutes = 180         # шаг между примерами

    print("="*80)
    print("ПРОВЕРКА ГИПОТЕЗЫ О ПЕРЕНОСЕ ОБУЧЕНИЯ ДЛЯ ПРОГНОЗА ВЕТРА")
    print("="*80)
    print(f"Данные: поминутные")
    print(f"Lookback: {lookback_minutes} минут ({lookback_minutes/60:.1f} часов)")
    print(f"Прогноз: на {forecast_minutes} минут ({forecast_minutes/60:.1f} часа)")
    print(f"Stride: {stride_minutes} минут ({stride_minutes/60:.1f} час) между примерами")
    print(f"Количество станций: {len(station_files_config)}")
    print(f"Устройство: {device}")

    try:
        # Запуск для горизонтальной компоненты
        results_horizontal = run_experiment_for_component(
            station_files_config, station_coords, station_heights,
            lookback_minutes, forecast_minutes, stride_minutes, component='horizontal'
        )

        # Запуск для вертикальной компоненты
        results_vertical = run_experiment_for_component(
            station_files_config, station_coords, station_heights,
            lookback_minutes, forecast_minutes, stride_minutes, component='vertical'
        )

        # Проверка гипотезы
        hypothesis_hor = test_hypothesis(results_horizontal, 'horizontal', forecast_minutes)
        hypothesis_ver = test_hypothesis(results_vertical, 'vertical', forecast_minutes)

        # Визуализация
        visualize_results(results_horizontal, results_vertical, forecast_minutes)

        # Итоговый вердикт
        print("\n" + "="*80)
        print("ИТОГОВЫЙ ВЕРДИКТ ПО ОБЕИМ КОМПОНЕНТАМ")
        print("="*80)

        print(f"\n--- Горизонтальная компонента (прогноз на {forecast_minutes/60:.0f} ч) ---")
        print(f"  Доля сопоставимых случаев: {hypothesis_hor['comparable_ratio']*100:.0f}%")
        print(f"  Средняя MAE Baseline:  {hypothesis_hor['mean_baseline']:.4f} м/с")
        print(f"  Средняя MAE Transfer:  {hypothesis_hor['mean_transfer']:.4f} м/с")
        print(f"  p-value: {hypothesis_hor['p_value']:.4f}")
        print(f"  Гипотеза: {'ПОДТВЕРЖДАЕТСЯ ✅' if hypothesis_hor['hypothesis_confirmed'] else 'НЕ ПОДТВЕРЖДАЕТСЯ ❌'}")

        print(f"\n--- Вертикальная компонента (прогноз на {forecast_minutes/60:.0f} ч) ---")
        print(f"  Доля сопоставимых случаев: {hypothesis_ver['comparable_ratio']*100:.0f}%")
        print(f"  Средняя MAE Baseline:  {hypothesis_ver['mean_baseline']:.4f} м/с")
        print(f"  Средняя MAE Transfer:  {hypothesis_ver['mean_transfer']:.4f} м/с")
        print(f"  p-value: {hypothesis_ver['p_value']:.4f}")
        print(f"  Гипотеза: {'ПОДТВЕРЖДАЕТСЯ ✅' if hypothesis_ver['hypothesis_confirmed'] else 'НЕ ПОДТВЕРЖДАЕТСЯ ❌'}")

        print("="*80)

    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()