# Spatial experiment

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import os
import gc
from glob import glob

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

# ==================== ПАРАМЕТРЫ МАСШТАБИРОВАНИЯ ====================
WIND_HOR_MIN = 0.0
WIND_HOR_MAX = 25.0
WIND_VER_MIN = -25.0
WIND_VER_MAX = 25.0

def scale_horizontal_to_original(scaled_values):
    """Преобразование горизонтальной скорости из [-1, 1] в [0, 25] м/с"""
    return (scaled_values + 1) / 2 * (WIND_HOR_MAX - WIND_HOR_MIN) + WIND_HOR_MIN

def scale_vertical_to_original(scaled_values):
    """Преобразование вертикальной скорости из [-1, 1] в [-25, 25] м/с"""
    return (scaled_values + 1) / 2 * (WIND_VER_MAX - WIND_VER_MIN) + WIND_VER_MIN

def scale_horizontal_to_normalized(original_values):
    """Преобразование горизонтальной скорости из [0, 25] в [-1, 1]"""
    return (original_values - WIND_HOR_MIN) / (WIND_HOR_MAX - WIND_HOR_MIN) * 2 - 1

def scale_vertical_to_normalized(original_values):
    """Преобразование вертикальной скорости из [-25, 25] в [-1, 1]"""
    return (original_values - WIND_VER_MIN) / (WIND_VER_MAX - WIND_VER_MIN) * 2 - 1

# ==================== ПРИВЕДЕНИЕ ВЕТРА К ВЫСОТЕ ====================
def wind_speed_to_height(wind_speed, from_height, to_height, alpha=0.2):
    """Приведение скорости ветра к другой высоте по степенному закону"""
    from_height = float(from_height)
    to_height = float(to_height)
    if from_height == to_height:
        return wind_speed
    return wind_speed * (to_height / from_height) ** alpha

# ==================== 1. ЗАГРУЗКА ДАННЫХ ====================

def load_multiple_csv_files(file_patterns, station_name, station_height, target_height=10.0):
    """Загрузка данных из нескольких CSV файлов для одной станции"""
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
        raise ValueError(f"Не найдено файлов для станции {station_name} по паттернам: {file_patterns}")

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

    print(f"    Всего записей после объединения: {len(combined_df)}")
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

def align_stations_data(stations_dict, lookback=24, forecast_horizon=3, component='horizontal'):
    """Выравнивание данных по времени для выбранной компоненты ветра"""
    common_timestamps = None
    for station_name, station_data in stations_dict.items():
        timestamps_set = set(station_data['timestamps'])
        if common_timestamps is None:
            common_timestamps = timestamps_set
        else:
            common_timestamps = common_timestamps.intersection(timestamps_set)

    common_timestamps = sorted(list(common_timestamps))
    print(f"Найдено общих временных меток: {len(common_timestamps)}")

    aligned_data = {}
    for station_name, station_data in stations_dict.items():
        if component == 'horizontal':
            wind_scaled = station_data['wind_hor_corrected_scaled']
        else:
            wind_scaled = station_data['wind_ver_scaled']

        timestamp_to_value = dict(zip(station_data['timestamps'], wind_scaled))
        aligned_values = np.array([timestamp_to_value.get(ts, np.nan) for ts in common_timestamps])

        aligned_data[station_name] = {
            'timestamps': common_timestamps,
            'wind_speed_scaled': aligned_values,
            'name': station_name,
            'height': station_data['height']
        }

    return aligned_data

def create_continuous_sequences(data_dict, lookback=24, forecast_horizon=3, stride=10):
    """
    Создание непрерывных последовательностей с разрежением

    stride: шаг между последовательностями (1 - все, 10 - каждую 10-ю)
    """
    sequences = {station: {'X': [], 'y': [], 'timestamps': []} for station in data_dict}

    for station_name, station_data in data_dict.items():
        wind_speed = station_data['wind_speed_scaled']
        timestamps = station_data['timestamps']

        continuous_segments = []
        current_segment = []
        current_timestamps = []

        for i, (ws, ts) in enumerate(zip(wind_speed, timestamps)):
            if pd.isna(ws):
                if len(current_segment) >= lookback + forecast_horizon:
                    continuous_segments.append((current_segment, current_timestamps))
                current_segment = []
                current_timestamps = []
            else:
                current_segment.append(ws)
                current_timestamps.append(ts)

        if len(current_segment) >= lookback + forecast_horizon:
            continuous_segments.append((current_segment, current_timestamps))

        print(f"{station_name}: найдено {len(continuous_segments)} непрерывных сегментов")

        # Используем шаг stride для разрежения
        total_before = 0
        for segment, segment_ts in continuous_segments:
            n_possible = len(segment) - lookback - forecast_horizon + 1
            total_before += n_possible
            for i in range(0, n_possible, stride):
                X_seq = segment[i:i+lookback]
                y_seq = segment[i+lookback:i+lookback+forecast_horizon]
                ts_seq = segment_ts[i+lookback:i+lookback+forecast_horizon]

                sequences[station_name]['X'].append(X_seq)
                sequences[station_name]['y'].append(y_seq)
                sequences[station_name]['timestamps'].append(ts_seq)

        sequences[station_name]['X'] = np.array(sequences[station_name]['X'])
        sequences[station_name]['y'] = np.array(sequences[station_name]['y'])
        print(f"  До разрежения: {total_before} последовательностей")
        print(f"  После разрежения (stride={stride}): {len(sequences[station_name]['X'])} последовательностей")

    return sequences

# ==================== IDW ИНТЕРПОЛЯЦИЯ ====================

def idw_interpolation(target_coord, source_coords, source_values, power=2):
    """IDW интерполяция"""
    distances = np.sqrt(((source_coords - target_coord) ** 2).sum(axis=1))
    distances = np.maximum(distances, 1e-10)
    weights = 1.0 / (distances ** power)
    weights = weights / weights.sum()
    return np.sum(source_values * weights, axis=0)

def idw_predict_sequence(target_coord, source_coords_dict, sequences, test_indices):
    """IDW прогноз на последовательность времени"""
    predictions = []
    source_stations = list(source_coords_dict.keys())
    source_coords_array = np.array([source_coords_dict[s] for s in source_stations])

    for idx in test_indices:
        source_values = []
        for source in source_stations:
            source_values.append(sequences[source]['y'][idx])
        source_values = np.array(source_values)
        pred = idw_interpolation(target_coord, source_coords_array, source_values)
        predictions.append(pred)
    return np.array(predictions)

# ==================== 2. LSTM МОДЕЛЬ ====================

class HeightAwareLSTM(nn.Module):
    """LSTM, который учитывает разницу высот между станциями"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=3, dropout=0.2):
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
    """Обучение LSTM с учетом разницы высот"""
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
            output = model(batch_x, batch_h)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t, height_diffs_val_t)
            val_loss = criterion(val_pred, y_val_t)

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

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def find_nearest_stations(target_station, station_coords, all_stations, n_neighbors=3):
    """Находит n ближайших станций к целевой"""
    target_coord = station_coords[target_station]
    distances = []

    for station in all_stations:
        if station != target_station:
            dist = np.linalg.norm(np.array(station_coords[station]) - np.array(target_coord))
            distances.append((station, dist))

    distances.sort(key=lambda x: x[1])
    return distances[:n_neighbors]

# ==================== ОСНОВНОЙ ЭКСПЕРИМЕНТ ====================

def run_experiment_for_component(station_files_config, station_coords, station_heights,
                                 lookback=24, forecast_horizon=3, stride=10, component='horizontal'):
    """
    Эксперимент для одной компоненты ветра

    stride: шаг разрежения (1 - все последовательности, 10 - каждая 10-я)
    """

    print(f"\n{'='*80}")
    print(f"ЭКСПЕРИМЕНТ ДЛЯ {component.upper()} КОМПОНЕНТЫ ВЕТРА")
    print(f"{'='*80}")
    print(f"Параметры: lookback={lookback}, forecast_horizon={forecast_horizon}, stride={stride}")

    # Загрузка данных
    print("\nЗагрузка данных...")
    stations_raw = {}
    for name in station_files_config.keys():
        file_patterns = station_files_config[name]
        print(f"\nЗагрузка станции {name}:")
        stations_raw[name] = load_multiple_csv_files(file_patterns, name, station_heights[name], target_height=10.0)

    # Выравнивание
    print("\nВыравнивание данных по времени...")
    aligned_stations = align_stations_data(stations_raw, lookback, forecast_horizon, component)

    # Создание последовательностей с разрежением
    print("\nСоздание непрерывных последовательностей с разрежением...")
    sequences = create_continuous_sequences(aligned_stations, lookback, forecast_horizon, stride)

    results = {}
    predictions_details = {}

    station_names = list(station_files_config.keys())

    # Для каждой станции как целевой
    for target_idx, target_station in enumerate(station_names):
        print(f"\n{'='*70}")
        print(f"Целевая станция: {target_station}")
        print(f"Высота: {station_heights[target_station]} м над уровнем моря")
        print(f"{'='*70}")

        # Находим ближайшие станции
        nearest = find_nearest_stations(target_station, station_coords, station_names, n_neighbors=3)

        print(f"\nБлижайшие станции (с учетом расположения):")
        for station, dist in nearest:
            height_diff = station_heights[station] - station_heights[target_station]
            print(f"  {station}: расстояние {dist:.4f}°, разница высот {height_diff:+.1f} м")

        nearest_station = nearest[0][0]
        nearest_distance = nearest[0][1]
        nearest_height_diff = station_heights[nearest_station] - station_heights[target_station]

        farthest_station = nearest[-1][0]
        farthest_distance = nearest[-1][1]
        farthest_height_diff = station_heights[farthest_station] - station_heights[target_station]

        # Данные целевой станции
        target_X = sequences[target_station]['X']
        target_y = sequences[target_station]['y']

        n_samples = len(target_X)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        test_indices = list(range(train_size + val_size, n_samples))

        # ===== МЕТОД 1: IDW ИНТЕРПОЛЯЦИЯ =====
        print(f"\nМетод 1: IDW интерполяция")

        source_coords_dict = {s: station_coords[s] for s in station_names if s != target_station}
        idw_predictions_scaled = idw_predict_sequence(
            station_coords[target_station], source_coords_dict, sequences, test_indices
        )

        idw_actual_scaled = target_y[test_indices]

        if component == 'horizontal':
            idw_predictions_original = scale_horizontal_to_original(idw_predictions_scaled.flatten())
            idw_actual_original = scale_horizontal_to_original(idw_actual_scaled.flatten())
        else:
            idw_predictions_original = scale_vertical_to_original(idw_predictions_scaled.flatten())
            idw_actual_original = scale_vertical_to_original(idw_actual_scaled.flatten())

        idw_mae = mean_absolute_error(idw_actual_original, idw_predictions_original)
        idw_rmse = np.sqrt(mean_squared_error(idw_actual_original, idw_predictions_original))
        idw_r2 = r2_score(idw_actual_original, idw_predictions_original)
        print(f"  MAE: {idw_mae:.4f} м/с, RMSE: {idw_rmse:.4f} м/с, R²: {idw_r2:.4f}")

        # ===== МЕТОД 2: ОБУЧЕНИЕ НА СОБСТВЕННОЙ ИСТОРИИ =====
        print(f"\nМетод 2: LSTM, обученный на истории этой станции (baseline)")

        baseline_train_X = target_X[:train_size]
        baseline_train_y = target_y[:train_size]
        baseline_val_X = target_X[train_size:train_size+val_size]
        baseline_val_y = target_y[train_size:train_size+val_size]

        if len(baseline_train_X.shape) == 2:
            baseline_train_X = baseline_train_X.reshape(-1, lookback, 1)
            baseline_val_X = baseline_val_X.reshape(-1, lookback, 1)

        baseline_model = HeightAwareLSTM(input_size=1, hidden_size=64, num_layers=1,
                                        output_size=forecast_horizon, dropout=0.2)

        baseline_height_diffs_train = np.zeros((len(baseline_train_X), 1))
        baseline_height_diffs_val = np.zeros((len(baseline_val_X), 1))

        baseline_model = train_lstm_with_heights(baseline_model, baseline_train_X, baseline_train_y,
                                                baseline_height_diffs_train, baseline_val_X, baseline_val_y,
                                                baseline_height_diffs_val, epochs=50)

        baseline_test_X = target_X[test_indices]
        baseline_test_y = target_y[test_indices]

        if len(baseline_test_X.shape) == 2:
            baseline_test_X = baseline_test_X.reshape(-1, lookback, 1)

        baseline_model.eval()
        baseline_model = baseline_model.to(device)
        baseline_height_diffs_test = np.zeros((len(baseline_test_X), 1))

        with torch.no_grad():
            baseline_pred_scaled = baseline_model(torch.FloatTensor(baseline_test_X).to(device),
                                                  torch.FloatTensor(baseline_height_diffs_test).to(device)).cpu().numpy()

        if component == 'horizontal':
            baseline_pred_original = scale_horizontal_to_original(baseline_pred_scaled.flatten())
            baseline_actual_original = scale_horizontal_to_original(baseline_test_y.flatten())
        else:
            baseline_pred_original = scale_vertical_to_original(baseline_pred_scaled.flatten())
            baseline_actual_original = scale_vertical_to_original(baseline_test_y.flatten())

        baseline_mae = mean_absolute_error(baseline_actual_original, baseline_pred_original)
        baseline_rmse = np.sqrt(mean_squared_error(baseline_actual_original, baseline_pred_original))
        baseline_r2 = r2_score(baseline_actual_original, baseline_pred_original)

        print(f"  MAE: {baseline_mae:.4f} м/с, RMSE: {baseline_rmse:.4f} м/с, R²: {baseline_r2:.4f}")

        # ===== МЕТОД 3: ПЕРЕНОС С БЛИЖАЙШЕЙ СТАНЦИИ =====
        print(f"\nМетод 3: Перенос с БЛИЖАЙШЕЙ станции {nearest_station}")
        print(f"  Расстояние: {nearest_distance:.4f}°, разница высот: {nearest_height_diff:+.1f} м")

        transfer_train_X = sequences[nearest_station]['X'][:train_size]
        transfer_train_y = sequences[nearest_station]['y'][:train_size]
        transfer_val_X = sequences[nearest_station]['X'][train_size:train_size+val_size]
        transfer_val_y = sequences[nearest_station]['y'][train_size:train_size+val_size]

        if len(transfer_train_X.shape) == 2:
            transfer_train_X = transfer_train_X.reshape(-1, lookback, 1)
            transfer_val_X = transfer_val_X.reshape(-1, lookback, 1)

        transfer_height_diffs_train = np.full((len(transfer_train_X), 1), float(nearest_height_diff))
        transfer_height_diffs_val = np.full((len(transfer_val_X), 1), float(nearest_height_diff))

        transfer_model = HeightAwareLSTM(input_size=1, hidden_size=64, num_layers=1,
                                        output_size=forecast_horizon, dropout=0.2)
        transfer_model = train_lstm_with_heights(transfer_model, transfer_train_X, transfer_train_y,
                                                transfer_height_diffs_train, transfer_val_X, transfer_val_y,
                                                transfer_height_diffs_val, epochs=50)

        transfer_test_X = target_X[test_indices]
        transfer_test_y = target_y[test_indices]

        if len(transfer_test_X.shape) == 2:
            transfer_test_X = transfer_test_X.reshape(-1, lookback, 1)

        transfer_model.eval()
        transfer_model = transfer_model.to(device)
        transfer_height_diffs_test = np.full((len(transfer_test_X), 1), float(nearest_height_diff))

        with torch.no_grad():
            transfer_pred_scaled = transfer_model(torch.FloatTensor(transfer_test_X).to(device),
                                                  torch.FloatTensor(transfer_height_diffs_test).to(device)).cpu().numpy()

        if component == 'horizontal':
            transfer_pred_original = scale_horizontal_to_original(transfer_pred_scaled.flatten())
            transfer_actual_original = scale_horizontal_to_original(transfer_test_y.flatten())
        else:
            transfer_pred_original = scale_vertical_to_original(transfer_pred_scaled.flatten())
            transfer_actual_original = scale_vertical_to_original(transfer_test_y.flatten())

        transfer_mae = mean_absolute_error(transfer_actual_original, transfer_pred_original)
        transfer_rmse = np.sqrt(mean_squared_error(transfer_actual_original, transfer_pred_original))
        transfer_r2 = r2_score(transfer_actual_original, transfer_pred_original)

        print(f"  MAE: {transfer_mae:.4f} м/с, RMSE: {transfer_rmse:.4f} м/с, R²: {transfer_r2:.4f}")

        # ===== МЕТОД 4: ПЕРЕНОС С ДАЛЬНЕЙ СТАНЦИИ =====
        print(f"\nМетод 4: Перенос с ДАЛЬНЕЙ станции {farthest_station}")
        print(f"  Расстояние: {farthest_distance:.4f}°, разница высот: {farthest_height_diff:+.1f} м")

        far_train_X = sequences[farthest_station]['X'][:train_size]
        far_train_y = sequences[farthest_station]['y'][:train_size]
        far_val_X = sequences[farthest_station]['X'][train_size:train_size+val_size]
        far_val_y = sequences[farthest_station]['y'][train_size:train_size+val_size]

        if len(far_train_X.shape) == 2:
            far_train_X = far_train_X.reshape(-1, lookback, 1)
            far_val_X = far_val_X.reshape(-1, lookback, 1)

        far_height_diffs_train = np.full((len(far_train_X), 1), float(farthest_height_diff))
        far_height_diffs_val = np.full((len(far_val_X), 1), float(farthest_height_diff))

        far_model = HeightAwareLSTM(input_size=1, hidden_size=64, num_layers=1,
                                   output_size=forecast_horizon, dropout=0.2)
        far_model = train_lstm_with_heights(far_model, far_train_X, far_train_y,
                                           far_height_diffs_train, far_val_X, far_val_y,
                                           far_height_diffs_val, epochs=50)

        far_test_X = target_X[test_indices]
        far_test_y = target_y[test_indices]

        if len(far_test_X.shape) == 2:
            far_test_X = far_test_X.reshape(-1, lookback, 1)

        far_model.eval()
        far_model = far_model.to(device)
        far_height_diffs_test = np.full((len(far_test_X), 1), float(farthest_height_diff))

        with torch.no_grad():
            far_pred_scaled = far_model(torch.FloatTensor(far_test_X).to(device),
                                        torch.FloatTensor(far_height_diffs_test).to(device)).cpu().numpy()

        if component == 'horizontal':
            far_pred_original = scale_horizontal_to_original(far_pred_scaled.flatten())
            far_actual_original = scale_horizontal_to_original(far_test_y.flatten())
        else:
            far_pred_original = scale_vertical_to_original(far_pred_scaled.flatten())
            far_actual_original = scale_vertical_to_original(far_test_y.flatten())

        far_mae = mean_absolute_error(far_actual_original, far_pred_original)
        far_rmse = np.sqrt(mean_squared_error(far_actual_original, far_pred_original))
        far_r2 = r2_score(far_actual_original, far_pred_original)

        print(f"  MAE: {far_mae:.4f} м/с, RMSE: {far_rmse:.4f} м/с, R²: {far_r2:.4f}")

        # ===== СРАВНЕНИЕ =====
        print(f"\n  СРАВНЕНИЕ МЕТОДОВ:")
        print(f"  IDW (интерполяция):           {idw_mae:.4f} м/с")
        print(f"  Baseline (своя история):      {baseline_mae:.4f} м/с")
        print(f"  Transfer (ближайшая):         {transfer_mae:.4f} м/с")
        print(f"  Transfer (дальняя):           {far_mae:.4f} м/с")

        diff_idw_vs_baseline = abs(idw_mae - baseline_mae)
        diff_transfer_vs_baseline = abs(baseline_mae - transfer_mae)
        relative_diff_transfer = diff_transfer_vs_baseline / baseline_mae * 100

        print(f"\n  IDW vs Baseline: разница {diff_idw_vs_baseline:.4f} м/с")
        print(f"  Transfer vs Baseline: разница {diff_transfer_vs_baseline:.4f} м/с ({relative_diff_transfer:.1f}%)")

        comparable = relative_diff_transfer < 20
        print(f"  Вывод: {'✓ СОПОСТАВИМ' if comparable else '✗ НЕ СОПОСТАВИМ'} (порог 20%)")

        results[target_station] = {
            'idw': {'MAE': idw_mae, 'RMSE': idw_rmse, 'R2': idw_r2},
            'baseline': {'MAE': baseline_mae, 'RMSE': baseline_rmse, 'R2': baseline_r2},
            'transfer_nearest': {'MAE': transfer_mae, 'RMSE': transfer_rmse, 'R2': transfer_r2,
                                'distance': nearest_distance, 'station': nearest_station,
                                'height_diff': nearest_height_diff},
            'transfer_farthest': {'MAE': far_mae, 'RMSE': far_rmse, 'R2': far_r2,
                                  'distance': farthest_distance, 'station': farthest_station,
                                  'height_diff': farthest_height_diff},
            'diff_transfer': diff_transfer_vs_baseline,
            'relative_diff': relative_diff_transfer,
            'comparable': comparable
        }

        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    return results

# ==================== ПРОВЕРКА ГИПОТЕЗЫ ====================

def test_hypothesis(results, component_name):
    """Проверка гипотезы"""
    print(f"\n{'='*80}")
    print(f"ПРОВЕРКА ГИПОТЕЗЫ ДЛЯ {component_name.upper()} КОМПОНЕНТЫ")
    print(f"{'='*80}")

    baseline_maes = [results[s]['baseline']['MAE'] for s in results]
    transfer_maes = [results[s]['transfer_nearest']['MAE'] for s in results]
    idw_maes = [results[s]['idw']['MAE'] for s in results]

    comparable_stations = [s for s in results if results[s]['comparable']]

    mean_baseline = np.mean(baseline_maes)
    mean_transfer = np.mean(transfer_maes)
    mean_idw = np.mean(idw_maes)

    print(f"\nСРЕДНИЕ ЗНАЧЕНИЯ ПО ВСЕМ СТАНЦИЯМ:")
    print(f"  IDW (интерполяция):           {mean_idw:.4f} м/с")
    print(f"  Baseline (своя история):      {mean_baseline:.4f} м/с")
    print(f"  Transfer (ближайшая):         {mean_transfer:.4f} м/с")

    print(f"\nРЕЗУЛЬТАТЫ ПО КАЖДОЙ СТАНЦИИ:")
    print(f"{'Станция':<20} {'Расст':<8} {'IDW':<10} {'Baseline':<10} {'Transfer':<10} {'Разница':<10} {'Статус':<12}")
    print("-" * 85)

    for station in results:
        dist = results[station]['transfer_nearest']['distance']
        idw_val = results[station]['idw']['MAE']
        baseline = results[station]['baseline']['MAE']
        transfer = results[station]['transfer_nearest']['MAE']
        diff = results[station]['diff_transfer']
        status = "✓ СОПОСТАВИМ" if results[station]['comparable'] else "✗ НЕ СОПОСТАВИМ"
        print(f"{station:<20} {dist:<8.3f} {idw_val:<10.4f} {baseline:<10.4f} {transfer:<10.4f} {diff:<10.4f} {status}")

    comparable_count = len(comparable_stations)
    total_count = len(results)
    comparable_ratio = comparable_count / total_count

    t_stat, p_value = ttest_rel(baseline_maes, transfer_maes)
    print(f"\nСТАТИСТИЧЕСКИЙ ТЕСТ (Baseline vs Transfer):")
    print(f"  t={t_stat:.4f}, p={p_value:.4f}")

    print("\n" + "="*80)
    print("ИТОГОВЫЙ ВЫВОД")
    print("="*80)

    if comparable_ratio >= 0.5:
        print(f"✅ ГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ для {component_name} компоненты")
        print(f"  Для {comparable_count} из {total_count} станций ({comparable_ratio*100:.0f}%)")
        print("  перенос с БЛИЖАЙШЕЙ станции дает точность, сопоставимую с собственной историей")
    else:
        print(f"❌ ГИПОТЕЗА НЕ ПОДТВЕРЖДАЕТСЯ для {component_name} компоненты")
        print(f"  Только для {comparable_count} из {total_count} станций ({comparable_ratio*100:.0f}%)")

    return {
        'mean_baseline': mean_baseline,
        'mean_transfer': mean_transfer,
        'mean_idw': mean_idw,
        'comparable_ratio': comparable_ratio,
        'p_value': p_value,
        'hypothesis_confirmed': comparable_ratio >= 0.5
    }

# ==================== ВИЗУАЛИЗАЦИЯ ====================

def visualize_comparison(results_hor, results_ver):
    """Визуализация сравнения двух компонент"""

    stations = list(results_hor.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Сравнение MAE горизонтальной компоненты
    ax1 = axes[0, 0]
    x = np.arange(len(stations))
    width = 0.25
    idw_hor = [results_hor[s]['idw']['MAE'] for s in stations]
    baseline_hor = [results_hor[s]['baseline']['MAE'] for s in stations]
    transfer_hor = [results_hor[s]['transfer_nearest']['MAE'] for s in stations]

    ax1.bar(x - width, idw_hor, width, label='IDW', color='#3498db', alpha=0.7, edgecolor='black')
    ax1.bar(x, baseline_hor, width, label='Baseline', color='#2ecc71', alpha=0.7, edgecolor='black')
    ax1.bar(x + width, transfer_hor, width, label='Transfer', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Станция')
    ax1.set_ylabel('MAE (м/с)')
    ax1.set_title('Горизонтальная компонента ветра')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stations, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Сравнение MAE вертикальной компоненты
    ax2 = axes[0, 1]
    idw_ver = [results_ver[s]['idw']['MAE'] for s in stations]
    baseline_ver = [results_ver[s]['baseline']['MAE'] for s in stations]
    transfer_ver = [results_ver[s]['transfer_nearest']['MAE'] for s in stations]

    ax2.bar(x - width, idw_ver, width, label='IDW', color='#3498db', alpha=0.7, edgecolor='black')
    ax2.bar(x, baseline_ver, width, label='Baseline', color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.bar(x + width, transfer_ver, width, label='Transfer', color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Станция')
    ax2.set_ylabel('MAE (м/с)')
    ax2.set_title('Вертикальная компонента ветра')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stations, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Сравнение Transfer с Baseline (горизонтальная)
    ax3 = axes[0, 2]
    rel_errors_hor = [results_hor[s]['relative_diff'] for s in stations]
    colors_hor = ['#2ecc71' if results_hor[s]['comparable'] else '#e74c3c' for s in stations]
    bars = ax3.bar(stations, rel_errors_hor, color=colors_hor, alpha=0.7, edgecolor='black')
    ax3.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Порог 20%')
    ax3.set_xlabel('Станция')
    ax3.set_ylabel('Относительная ошибка (%)')
    ax3.set_title('Горизонтальная: Transfer vs Baseline')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    for bar, value in zip(bars, rel_errors_hor):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)

    # 4. Сравнение Transfer с Baseline (вертикальная)
    ax4 = axes[1, 0]
    rel_errors_ver = [results_ver[s]['relative_diff'] for s in stations]
    colors_ver = ['#2ecc71' if results_ver[s]['comparable'] else '#e74c3c' for s in stations]
    bars = ax4.bar(stations, rel_errors_ver, color=colors_ver, alpha=0.7, edgecolor='black')
    ax4.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Порог 20%')
    ax4.set_xlabel('Станция')
    ax4.set_ylabel('Относительная ошибка (%)')
    ax4.set_title('Вертикальная: Transfer vs Baseline')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    for bar, value in zip(bars, rel_errors_ver):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)

    # 5. Сравнение IDW и Transfer (горизонтальная)
    ax5 = axes[1, 1]
    idw_vs_transfer_hor = [abs(results_hor[s]['idw']['MAE'] - results_hor[s]['transfer_nearest']['MAE']) for s in stations]
    bars = ax5.bar(stations, idw_vs_transfer_hor, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Станция')
    ax5.set_ylabel('Абсолютная разница (м/с)')
    ax5.set_title('Горизонтальная: IDW vs Transfer')
    ax5.grid(True, alpha=0.3)

    for bar, value in zip(bars, idw_vs_transfer_hor):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    # 6. Сравнение IDW и Transfer (вертикальная)
    ax6 = axes[1, 2]
    idw_vs_transfer_ver = [abs(results_ver[s]['idw']['MAE'] - results_ver[s]['transfer_nearest']['MAE']) for s in stations]
    bars = ax6.bar(stations, idw_vs_transfer_ver, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Станция')
    ax6.set_ylabel('Абсолютная разница (м/с)')
    ax6.set_title('Вертикальная: IDW vs Transfer')
    ax6.grid(True, alpha=0.3)

    for bar, value in zip(bars, idw_vs_transfer_ver):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('wind_components_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== ЗАПУСК ====================

def main():
    """Главная функция"""

    # Конфигурация файлов для каждой станции
    station_files_config = {
        'kinzjar_10m': ["autumn_kinzjar_10m.csv", "summer_kinzjar_10m.csv", "winter_kinzjar_10m.csv"],
        'kireevsk_10m': ["autumn_kireevsk_10m.csv", "summer_kireevsk_10m.csv", "winter_kireevsk_10m.csv"],
        'imces_27m': ["autumn_imces_27m.csv", "summer_imces_30m.csv", "winter_imces_27m.csv"],
        'oblkom_27m': ["autumn_oblkom_27m.csv", "summer_oblkom_26m.csv", "winter_oblkom_27m.csv"],
    }

    # Координаты станций
    station_coords = {
        'kinzjar_10m': (57.6217, 82.3392),
        'kireevsk_10m': (56.4150, 84.0678),
        'imces_27m': (56.4752778, 85.0544444),
        'oblkom_27m': (56.4672222, 84.9575),
    }

    # Высоты станций
    station_heights = {
        'kinzjar_10m': 80.0,
        'kireevsk_10m': 91.0,
        'imces_27m': 194.0,
        'oblkom_27m': 139.0,
    }

    # Параметры эксперимента
    lookback = 6          # 6 часов (360 минут)
    forecast_horizon = 3  # прогноз на 3 часа
    stride = 70           # разрежение

    print("="*80)
    print("ЭКСПЕРИМЕНТ ДЛЯ ДВУХ КОМПОНЕНТ ВЕТРА")
    print("="*80)
    print(f"Lookback window: {lookback} часов")
    print(f"Прогнозный горизонт: {forecast_horizon} часа")
    print(f"Stride (разрежение): {stride}")
    print(f"Количество станций: {len(station_files_config)}")
    print(f"Устройство: {device}")

    print("\nИнформация о станциях:")
    for name in station_files_config.keys():
        coords = station_coords[name]
        height = station_heights[name]
        print(f"  {name}: ({coords[0]:.4f}, {coords[1]:.4f}), высота {height} м")

    try:
        # Запуск для горизонтальной компоненты
        results_horizontal = run_experiment_for_component(
            station_files_config, station_coords, station_heights,
            lookback, forecast_horizon, stride=stride, component='horizontal'
        )

        # Запуск для вертикальной компоненты
        results_vertical = run_experiment_for_component(
            station_files_config, station_coords, station_heights,
            lookback, forecast_horizon, stride=stride, component='vertical'
        )

        # Проверка гипотез
        hypothesis_hor = test_hypothesis(results_horizontal, 'horizontal')
        hypothesis_ver = test_hypothesis(results_vertical, 'vertical')

        # Визуализация
        visualize_comparison(results_horizontal, results_vertical)

        # Итоговый вывод
        print("\n" + "="*80)
        print("ИТОГОВЫЙ ВЕРДИКТ ПО ОБЕИМ КОМПОНЕНТАМ")
        print("="*80)

        print(f"\nГоризонтальная компонента (горизонт прогноза {forecast_horizon} ч):")
        if hypothesis_hor['hypothesis_confirmed']:
            print("  ✅ ГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ")
            print(f"     Доля сопоставимых случаев: {hypothesis_hor['comparable_ratio']*100:.0f}%")
            print(f"     Средняя MAE Baseline: {hypothesis_hor['mean_baseline']:.3f} м/с")
            print(f"     Средняя MAE Transfer: {hypothesis_hor['mean_transfer']:.3f} м/с")
            print(f"     Средняя MAE IDW:      {hypothesis_hor['mean_idw']:.3f} м/с")
            print(f"     p-value: {hypothesis_hor['p_value']:.4f}")
        else:
            print("  ❌ ГИПОТЕЗА НЕ ПОДТВЕРЖДАЕТСЯ")

        print(f"\nВертикальная компонента (горизонт прогноза {forecast_horizon} ч):")
        if hypothesis_ver['hypothesis_confirmed']:
            print("  ✅ ГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ")
            print(f"     Доля сопоставимых случаев: {hypothesis_ver['comparable_ratio']*100:.0f}%")
            print(f"     Средняя MAE Baseline: {hypothesis_ver['mean_baseline']:.3f} м/с")
            print(f"     Средняя MAE Transfer: {hypothesis_ver['mean_transfer']:.3f} м/с")
            print(f"     Средняя MAE IDW:      {hypothesis_ver['mean_idw']:.3f} м/с")
            print(f"     p-value: {hypothesis_ver['p_value']:.4f}")
        else:
            print("  ❌ ГИПОТЕЗА НЕ ПОДТВЕРЖДАЕТСЯ")

        print("="*80)

    except Exception as e:
        print(f"\nОшибка при выполнении эксперимента: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()