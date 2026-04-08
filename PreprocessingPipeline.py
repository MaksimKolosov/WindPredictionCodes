import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import warnings


class PreprocessingPipeline:
    def __init__(self, input_file_path: str, output_file_path: str):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.df = None
        self.physical_limits_tomsk_dict = {"air_temperature": [-55, 38], "wind_speed_horizontal": [0, 32],
                                          "wind_direction": [0, 360], "wind_speed_min": [0, 32],
                                          "wind_speed_max": [0, 32], "wind_speed_vertical": [-40, 40],
                                          "atmospheric_pressure": [710, 805], "relative_humidity": [10, 100],
                                          "dew_point_temperature": [-50, 24], "pressure_derivative": [-0.00038, 0.00038],
                                          "wind_forecast_three_hours": [0, 32], "day_of_year": [0, 366]}
        self.time_threshold_for_gaps = 15
        self.minmax_for_scaling_dict = {"air_temperature": [-30, 30], "wind_speed_horizontal": [0, 25],
                                          "wind_direction": [0, 360], "wind_speed_min": [0, 25],
                                          "wind_speed_max": [0, 25], "wind_speed_vertical": [-25, 25],
                                          "atmospheric_pressure": [730, 780], "relative_humidity": [30, 100],
                                          "dew_point_temperature": [-30, 30], "pressure_derivative": [-0.00026, 0.00026],
                                          "wind_forecast_three_hours": [0, 25], "day_of_year": [0, 366]}

    def _load_data(self):
        print("=" * 50)
        print("ЗАГРУЗКА ДАННЫХ")
        print("=" * 50)

        # Читаем датафрейм из файла
        self.df = pd.read_csv(self.input_file_path)
        
        print(f"Данные успешно загружены из файла {self.input_file_path}")
        return self

    def _modify_date_column(self):
        print("\n" + "=" * 50)
        print("МОДИФИКАЦИЯ КОЛОНКИ ДАТЫ-ВРЕМЕНИ")
        print("=" * 50)

        # Для удобства преобразуем время в DateTime без часовой зоны
        self.df['date'] = pd.to_datetime(self.df['date']).dt.tz_localize(None)

        print("Признак date преобразован в формат DateTime без часовой зоны")
        return self

    def _delete_old_data(self):
        # Удаление данных за годы, предшествующие 2022
        print("\n" + "=" * 50)
        print("УДАЛЕНИЕ СТАРЫХ ДАННЫХ")
        print("=" * 50)

        old_count = len(self.df[self.df['date'].dt.year < 2022])
        if old_count > 0:
            self.df = self.df[self.df['date'].dt.year >= 2022]
            print("Данные старше 2022 года удалены")
        else:
            print("В датасете нет старых данных, удалять ничего не нужно")
        
        return self

    def _delete_duplicates(self):
        print("\n" + "=" * 50)
        print("УДАЛЕНИЕ ДУБЛИКАТОВ (ЕСЛИ ЕСТЬ)")
        print("=" * 50)

        str_num_before = self.df.shape[0]
        print(f"Количество строк до удаления дубликатов: {str_num_before}")

        self.df = self.df.drop_duplicates()

        str_num_after = self.df.shape[0]
        str_deleted = str_num_before - str_num_after
        print(f"Количество удалённых дубликатов: {str_deleted}")
        print(f"Количество строк после удаления дубликатов: {str_num_after}")

        return self

    def _remove_unnecessary_features(self):
        print("\n" + "=" * 50)
        print("УДАЛЕНИЕ НЕНУЖНЫХ ПРИЗНАКОВ")
        print("=" * 50)

        # Удаляем признаки (малоинформативные или сильно коррелирующие с другими признаками)
        features_to_drop = ['vapor_pressure', 'absolute_humidity', 'air_density', 
                            'speed_of_sound', 'CFSv2_temperature', 'cloud_mixing']
        self.df = self.df.drop(features_to_drop, axis=1)

        print(f"Удалены признаки: {", ".join(features_to_drop)}")
        return self

    def _add_day_of_year_feature(self):
        print("\n" + "=" * 50)
        print("ДОБАВЛЕНИЕ ПРИЗНАКА ДНЯ ГОДА")
        print("=" * 50)

        # День года (1-365/366)
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        
        print("Добавлен признак: day_of_year")
        
        return self

    def _add_pressure_derivative(self):
        # Добавление производной атмосферного давления по времени (dp/dt)
        print("\n" + "=" * 50)
        print("ДОБАВЛЕНИЕ ПРОИЗВОДНОЙ АТМОСФЕРНОГО ДАВЛЕНИЯ")
        print("=" * 50)
        
        # Признак атмосферного давления
        pressure_col = "atmospheric_pressure"
        
        # Создаём копию датафрейма и работаем с временным индексом
        df_temp = self.df.copy()
        
        # Преобразуем столбец date в datetime и устанавливаем как индекс
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp = df_temp.set_index('date')
        
        # Шаг 1: Сглаживание давления (медиана, окно 5 минут)
        window_seconds_pressure = 300  # 5 минут
        pressure_smoothed = df_temp[pressure_col].rolling(
            window=window_seconds_pressure, min_periods=1, center=True
        ).median()
        
        # Шаг 2: Вычисление производной от сглаженного давления
        time_diffs = df_temp.index.to_series().diff().dt.total_seconds()
        pressure_diff = pressure_smoothed.diff()
        derivative = pressure_diff / time_diffs
        derivative = derivative.replace([np.inf, -np.inf], np.nan)
        
        # Пропускаем большие разрывы (> 1 минуты)
        derivative[time_diffs > 60] = np.nan
        
        # Шаг 3: Сглаживание производной (среднее, окно 3 минуты)
        window_seconds_derivative = 180  # 3 минуты
        derivative_smoothed = derivative.rolling(
            window=window_seconds_derivative, min_periods=1, center=True
        ).mean()
        
        # Добавляем в исходный df
        self.df['pressure_derivative'] = derivative_smoothed.fillna(0).values
        
        print("Добавлен признак: pressure_derivative")
        
        return self

    def _remove_outliers(self):
        # Устранение выбросов и аномалий
        print("\n" + "=" * 50)
        print("УСТРАНЕНИЕ ВЫБРОСОВ И АНОМАЛИЙ")
        print("=" * 50)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['sin_time_of_day', 'cos_time_of_day', 'day_of_year']
        
        total_removed = 0
        
        for col in numerical_cols:
            if col in exclude_cols:
                continue
                
            original_count = self.df[col].notna().sum()
            if original_count == 0:
                continue
            
            # ========== 1. СТАТИСТИЧЕСКИЕ МЕТОДЫ ==========
            data = self.df[col].copy()
            
            # IQR метод
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_lower = Q1 - 3 * IQR
            iqr_upper = Q3 + 3 * IQR
            
            # Z-score метод
            if data.std() > 1e-6:
                z_scores = np.abs((data - data.mean()) / data.std())
                z_score_mask = z_scores > 5
            else:
                z_score_mask = pd.Series(False, index=data.index)
            
            # Локальные выбросы
            rolling_mean = data.rolling(window=11, center=True, min_periods=3).mean()
            rolling_std = data.rolling(window=11, center=True, min_periods=3).std()
            rolling_std = rolling_std.replace(0, np.nan)
            
            if rolling_std.notna().any():
                z_score_local = np.abs((data - rolling_mean) / rolling_std)
                local_outlier_mask = z_score_local > 6
            else:
                local_outlier_mask = pd.Series(False, index=data.index)
            
            # Комбинированная маска
            outlier_mask = (
                (data < iqr_lower) | (data > iqr_upper) |
                z_score_mask |
                local_outlier_mask
            )
            
            # ========== 2. ФИЗИЧЕСКИЕ ЛИМИТЫ ==========
            if hasattr(self, 'physical_limits_tomsk_dict') and col in self.physical_limits_tomsk_dict:
                min_limit, max_limit = self.physical_limits_tomsk_dict[col]
                
                if col == 'wind_direction':
                    mask_physical = (self.df[col] < min_limit) | (self.df[col] >= max_limit)
                else:
                    mask_physical = (self.df[col] < min_limit) | (self.df[col] > max_limit)
                
                outlier_mask = outlier_mask | mask_physical
            
            num_removed = outlier_mask.sum()
            
            if num_removed > 0:
                self.df.loc[outlier_mask, col] = np.nan
                total_removed += num_removed
        
        print(f"Найдено выбросов и аномалий и заменено на NaN: {total_removed}")
        
        return self

    def _fill_in_short_time_gaps(self):
        print("\n" + "=" * 50)
        print("ВОСПОЛНЕНИЕ КОРОТКИХ ВРЕМЕННЫХ ПРОПУСКОВ")
        print("=" * 50)

        time_threshold = self.time_threshold_for_gaps
        date_col = 'date'

        # Начальное количество строк
        num_rows_before = self.df.shape[0]
        
        # Сортируем по дате
        self.df = self.df.sort_values(date_col).reset_index(drop=True)

        # Создаём копию для результата
        result_df = self.df.copy()
        
        # Находим все пропуски между последовательными строками
        for i in range(len(self.df) - 1):
            current_time = self.df.loc[i, date_col]
            next_time = self.df.loc[i + 1, date_col]
            
            # Вычисляем разницу в минутах
            time_diff = (next_time - current_time).total_seconds() / 60
            
            # Если разница больше 1 минуты (пропуск) но меньше или равна порогу
            if 1 < time_diff <= time_threshold:
                # Создаём недостающие временные метки
                missing_times = pd.date_range(
                    start=current_time + pd.Timedelta(minutes=1),
                    end=next_time - pd.Timedelta(minutes=1),
                    freq='1min'
                )
                
                # Создаём DataFrame с пропущенными временами и NaN в остальных колонках
                missing_data = []
                for missing_time in missing_times:
                    row_data = {date_col: missing_time}
                    # Для всех остальных колонок добавляем NaN
                    for col in self.df.columns:
                        if col != date_col:
                            row_data[col] = np.nan
                    missing_data.append(row_data)
                
                # Добавляем в результат
                missing_df = pd.DataFrame(missing_data)
                result_df = pd.concat([result_df, missing_df], ignore_index=True)
        
        # Сортируем по дате итоговый результат
        self.df = result_df.sort_values(date_col).reset_index(drop=True)
        
        # Количество строк после добавления
        num_rows_after = self.df.shape[0]

        # Количество добавленных строк
        num_rows_added = num_rows_after - num_rows_before
        
        print(f"Временные пропуски длиной не более {time_threshold} минут восполнены NaN-значениями")
        print(f"Добавлено строк: {num_rows_added}")
        
        return self

    def _remove_consecutive_missing_values(self):
        print("\n" + "=" * 50)
        print("УДАЛЕНИЕ ПРОДОЛЖИТЕЛЬНЫХ NAN-ПРОПУСКОВ")
        print("=" * 50)

        print(f"Строк в датафрейме: {len(self.df)}")

        m_minutes = self.time_threshold_for_gaps
        
        # Создаём копию датафрейма и работаем с временным индексом
        df_temp = self.df.copy()
        
        # Преобразуем столбец date в datetime и устанавливаем как индекс
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp = df_temp.set_index('date')
        
        # Создаём маску для удаления
        mask_to_remove = pd.Series(False, index=df_temp.index)
        
        for column in df_temp.columns:
            # Пропускаем столбец с датой, так как он теперь индекс
            if column == 'date':
                continue
                
            # Находим пропуски
            is_null = df_temp[column].isna()
            
            if not is_null.any():
                continue
                
            # Находим группы последовательных пропусков
            null_groups = (is_null != is_null.shift()).cumsum()
            
            # Для каждой группы пропусков вычисляем длительность
            for group_id in null_groups[is_null].unique():
                group_mask = (null_groups == group_id)
                group_indices = df_temp.index[group_mask]
                
                if len(group_indices) > 1:
                    # Вычисляем длительность группы в минутах
                    time_start = group_indices[0]
                    time_end = group_indices[-1]
                    duration_minutes = (time_end - time_start).total_seconds() / 60
                    
                    # Если длительность больше m_minutes, помечаем для удаления
                    if duration_minutes > m_minutes:
                        mask_to_remove = mask_to_remove | group_mask
                else:
                    # Для одиночных пропусков проверяем интервалы с соседними точками
                    idx = group_indices[0]
                    pos = df_temp.index.get_loc(idx)
                    
                    # Проверяем интервал с предыдущей точкой
                    if pos > 0:
                        prev_time = df_temp.index[pos - 1]
                        gap_minutes = (idx - prev_time).total_seconds() / 60
                        if gap_minutes > m_minutes:
                            mask_to_remove = mask_to_remove | group_mask
                    
                    # Проверяем интервал со следующей точкой
                    if pos < len(df_temp) - 1:
                        next_time = df_temp.index[pos + 1]
                        gap_minutes = (next_time - idx).total_seconds() / 60
                        if gap_minutes > m_minutes:
                            mask_to_remove = mask_to_remove | group_mask
        
        # Получаем индексы строк для удаления из исходного датафрейма
        indices_to_remove = df_temp[mask_to_remove].index
        self.df = self.df[~self.df['date'].isin(indices_to_remove)]

        print(f"Удалено строк: {len(indices_to_remove)}")
        print(f"Осталось строк: {len(self.df)}")
        
        return self

    def _fill_missing_values(self):
        print("\n" + "=" * 50)
        print("ЗАПОЛНЕНИЕ NAN-ПРОПУСКОВ")
        print("=" * 50)

        # Выведем количество NaN-пропусков (до заполнения)
        print("Количество пропусков (до заполнения):")
        print(self.df.isna().sum())

        # Добавим признаки sin_time_of_day, cos_time_of_day (если их нет) и заполним
        if 'sin_time_of_day' not in self.df.columns:
            self.df['sin_time_of_day'] = np.nan
        mask = self.df['sin_time_of_day'].isna()
        if mask.any():
            minutes_since_epoch = (self.df.loc[mask, 'date'].astype('int64') // (60 * 10**9))
            self.df.loc[mask, 'sin_time_of_day'] = np.sin((2 * np.pi * minutes_since_epoch) / 1440)
        if 'cos_time_of_day' not in self.df.columns:
            self.df['cos_time_of_day'] = np.nan
        mask = self.df['cos_time_of_day'].isna()
        if mask.any():
            minutes_since_epoch = (self.df.loc[mask, 'date'].astype('int64') // (60 * 10**9))
            self.df.loc[mask, 'cos_time_of_day'] = np.cos((2 * np.pi * minutes_since_epoch) / 1440)

        # Устанавливаем единый порядок колонок
        self.df = self.df[['date', 'day_of_year', 'air_temperature', 'wind_speed_horizontal', 
                           'wind_direction', 'wind_speed_min', 'wind_speed_max', 
                           'wind_speed_vertical', 'atmospheric_pressure', 'pressure_derivative', 
                           'relative_humidity', 'dew_point_temperature', 
                           'sin_time_of_day', 'cos_time_of_day']]

        # Заполняем пропуски с помощью интерполяции по времени
        df_temp = self.df.set_index('date')
        numeric_cols = df_temp.select_dtypes(include=['number']).columns
        df_temp[numeric_cols] = df_temp[numeric_cols].interpolate(method='time')
        self.df = df_temp.reset_index()
        print("\nПропуски заполнены интерполяцией по времени\n")
        
        # Выведем количество NaN-пропусков (после заполнения)
        print("Количество пропусков (после заполнения):")
        print(self.df.isna().sum())

        return self

    def _smooth_noisy_features(self):
        # Сглаживание зашумленных признаков
        print("\n" + "=" * 50)
        print("СГЛАЖИВАНИЕ ЗАШУМЛЕННЫХ ПРИЗНАКОВ")
        print("=" * 50)
        
        # Исключаемые колонки (не сглаживаем)
        exclude_columns = ['date', 'day_of_year', 'sin_time_of_day', 'cos_time_of_day',
                          'wind_direction', 'relative_humidity']
        
        # Числовые колонки для сглаживания
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        columns_to_smooth = [col for col in numeric_columns if col not in exclude_columns]
        
        if not columns_to_smooth:
            print("Нет признаков для сглаживания")
            return self
        
        df_temp = self.df.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp = df_temp.set_index('date')
        
        smoothed_features = []
        not_smoothed_features = []
        
        for col in columns_to_smooth:
            series = df_temp[col].dropna()
            
            if len(series) < 100:
                print(f"{col}: недостаточно данных, сглаживание пропущено")
                continue
            
            # Определяем, нужно ли сглаживать
            need_smoothing = False
            window_minutes = 10  # окно по умолчанию
            
            # Атмосферное давление и температуры сглаживаем всегда
            if col == 'atmospheric_pressure' or 'temperature' in col:
                need_smoothing = True
                window_minutes = 10
            elif col == 'pressure_derivative':
                # Анализируем уровень шума для производной
                diff_1 = series.diff().abs().median()
                rolling_std = series.rolling(window=10, min_periods=3).std().median()
                ma_short = series.rolling(window=5, min_periods=3).mean()
                high_freq_noise = (series - ma_short).abs().median()
                noise_score = (diff_1 + rolling_std + high_freq_noise) / 3
                
                # Ожидаемый уровень шума для производной (мм рт.ст./сек)
                expected_noise = 0.00005
                
                if noise_score > expected_noise * 1.5:
                    need_smoothing = True
                    # Для производной используем маленькое окно (2-3 минуты)
                    window_minutes = 3 if noise_score < expected_noise * 2.5 else 5
            else:
                # Для остальных признаков анализируем уровень шума
                diff_1 = series.diff().abs().median()
                rolling_std = series.rolling(window=10, min_periods=3).std().median()
                ma_short = series.rolling(window=5, min_periods=3).mean()
                high_freq_noise = (series - ma_short).abs().median()
                noise_score = (diff_1 + rolling_std + high_freq_noise) / 3
                
                # Определяем ожидаемый уровень шума
                if 'air_temperature' in col.lower():
                    expected_noise = 0.15
                elif 'wind_speed' in col.lower():
                    expected_noise = 0.25
                elif 'dew_point' in col.lower():
                    expected_noise = 0.2
                else:
                    expected_noise = 0.15
                
                # Решение о сглаживании
                if noise_score > expected_noise * 1.5:
                    need_smoothing = True
                    if 'wind_speed' in col.lower():
                        window_minutes = 3 if noise_score < expected_noise * 2.5 else 8
                    else:
                        window_minutes = 5 if noise_score < expected_noise * 2.5 else 15
            
            # Выполняем сглаживание если нужно
            if need_smoothing:
                # Сохраняем оригинал
                original_series = self.df[col].copy()
                
                # Выполняем сглаживание
                smoothed_count = 0
                for i in range(len(series)):
                    current_time = series.index[i]
                    start_time = current_time - pd.Timedelta(minutes=window_minutes/2)
                    end_time = current_time + pd.Timedelta(minutes=window_minutes/2)
                    
                    window_points = series[(series.index >= start_time) & (series.index <= end_time)]
                    
                    if len(window_points) >= 3:
                        smoothed_value = window_points.median()
                        mask = self.df['date'] == current_time
                        if col == 'pressure_derivative':
                            self.df.loc[mask, col] = round(smoothed_value, 6)
                        else:
                            self.df.loc[mask, col] = round(smoothed_value, 2)
                        smoothed_count += 1
                
                # Оценка
                original_diff = original_series.diff().abs().median()
                after_smoothing = self.df[col].copy()
                smoothed_diff = after_smoothing.diff().abs().median()
                noise_reduction = (1 - smoothed_diff / original_diff) * 100 if original_diff > 0 else 0
                
                print(f"\n{col}:")
                print(f"   → Выполняется сглаживание")
                print(f"   → Снижение шума: {noise_reduction:.1f}%")
                
                smoothed_features.append(col)
            else:
                # Признак не требует сглаживания
                not_smoothed_features.append(col)
        
        print(f"\nСглаживание завершено")
        
        return self

    def _add_wind_forecast_column(self):
        print("\n" + "=" * 60)
        print("ДОБАВЛЕНИЕ КОЛОНКИ ПРОГНОЗА СКОРОСТИ ВЕТРА НА 3 ЧАСА ВПЕРЁД")
        print("(АРХИВНЫЕ ПРОГНОЗЫ ДЛЯ ТОМСКА И ТОМСКОЙ ОБЛАСТИ С OPEN METEO API)")
        print("=" * 60)
        
        # Координаты станции
        input_file_name = os.path.basename(self.input_file_path)
        if "imces" in input_file_name:
            STATION_LAT = 56.4752778
            STATION_LON = 85.0544444
        elif "oblkom" in input_file_name:
            STATION_LAT = 56.4672222
            STATION_LON = 84.9575
        elif "bek" in input_file_name:
            STATION_LAT = 56.4811111
            STATION_LON = 85.1013889
        elif "voronino" in input_file_name:
            STATION_LAT = 56.5577778
            STATION_LON = 85.2561111
        elif "kurlek" in input_file_name:
            STATION_LAT = 56.2258333
            STATION_LON = 84.8662222
        elif "kireevsk" in input_file_name:
            STATION_LAT = 56.415
            STATION_LON = 84.0677778
        elif "vasuganie" in input_file_name:
            STATION_LAT = 56.95
            STATION_LON = 82.5
        elif "kinzjar" in input_file_name:
            STATION_LAT = 57.6216667
            STATION_LON = 82.3391667
        else:
            STATION_LAT = 56.4672222
            STATION_LON = 84.9575
        
        # Создаём копию dataframe
        df_result = self.df.copy()
        
        # Преобразуем дату в правильный формат
        if 'date' in df_result.columns:
            df_result['date'] = pd.to_datetime(df_result['date'], format='%Y-%m-%d %H:%M:%S')
            df_result = df_result.set_index('date').sort_index()
        
        # Добавляем новый столбец
        df_result['wind_forecast_three_hours'] = np.nan
        
        # Получаем уникальные даты из dataframe
        target_dates = df_result.index.normalize().unique()
        
        # ========== ЗАГРУЗКА ПО КУСОЧКАМ ==========
        print(f"Всего уникальных дат для загрузки: {len(target_dates)}")
        
        # Размер чанка (количество дней за один запрос)
        chunk_days = 30  # можно уменьшить до 14, если всё равно падает
        
        all_forecasts = []
        total_chunks = (len(target_dates) + chunk_days - 1) // chunk_days
        
        for chunk_idx in range(0, len(target_dates), chunk_days):
            chunk_dates = target_dates[chunk_idx:chunk_idx + chunk_days]
            
            # Добавляем ещё один день для прогноза (последний день + 1)
            last_date = chunk_dates[-1]
            next_day = last_date + pd.Timedelta(days=1)
            chunk_dates_with_forecast = chunk_dates.append(pd.Index([next_day]))
            
            start_date = chunk_dates[0]
            end_date = chunk_dates[-1]
            
            print(f"\n   Загрузка чанка {chunk_idx//chunk_days + 1}/{total_chunks}: "
                  f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
            
            try:
                # Загружаем прогнозы для текущего чанка
                gfs_forecasts = self.__get_historical_forecast(
                    STATION_LAT, STATION_LON, chunk_dates_with_forecast
                )
                
                if gfs_forecasts is not None and len(gfs_forecasts) > 0:
                    all_forecasts.append(gfs_forecasts)
                    print(f"      Загружено {len(gfs_forecasts)} записей")
                else:
                    print(f"      Не удалось загрузить данные для чанка")
                    
            except Exception as e:
                print(f"      Ошибка загрузки: {e}")
                continue
        
        # Объединяем все прогнозы
        if all_forecasts:
            combined_forecasts = pd.concat(all_forecasts)
            # Удаляем дубликаты индексов
            combined_forecasts = combined_forecasts[~combined_forecasts.index.duplicated(keep='first')]
            combined_forecasts = combined_forecasts.sort_index()
            print(f"\nВсего загружено: {len(combined_forecasts)} записей прогнозов")
        else:
            combined_forecasts = None
            print("\nНе удалось загрузить данные ни для одного чанка")
        
        # ========== СОПОСТАВЛЕНИЕ ПРОГНОЗОВ ==========
        if combined_forecasts is not None:
            print("\nИдёт сопоставление прогнозов...")
            matched_count = 0
            
            for idx, row in df_result.iterrows():
                # Время для которого нужен прогноз (текущее время + 3 часа)
                forecast_target_time = idx + timedelta(hours=3)
                
                # Ищем ближайший доступный прогноз
                time_diff = abs(combined_forecasts.index - forecast_target_time)
                closest_idx = time_diff.argmin()
                
                if time_diff[closest_idx] < timedelta(hours=1):
                    forecast_value = combined_forecasts.iloc[closest_idx]['wind_speed_forecast']
                    df_result.at[idx, 'wind_forecast_three_hours'] = forecast_value
                    matched_count += 1
                
                # Прогресс-бар для длительных операций
                if matched_count % 10000 == 0 and matched_count > 0:
                    print(f"      Обработано {matched_count} записей...")
            
            print(f"Успешно сопоставлено {matched_count} прогнозов")
            
        else:
            print("\n⚠Не удалось загрузить данные, используем демо-данные")
            # Заполняем демо-данными на основе текущей скорости ветра
            for idx, row in df_result.iterrows():
                current_wind = row.get('wind_speed_horizontal', 3.0)
                forecast = current_wind * (1 + np.random.uniform(-0.1, 0.1))
                df_result.at[idx, 'wind_forecast_three_hours'] = max(0.1, forecast)
        
        # ========== ИНТЕРПОЛЯЦИЯ ==========
        print("\nИнтерполяция пропущенных значений...")
        
        # Линейная интерполяция временного ряда
        df_result['wind_forecast_three_hours'] = df_result['wind_forecast_three_hours'].interpolate(
            method='linear', 
            limit_direction='both',
            limit=10
        )
        
        # Если остались пропуски, заполняем скользящим средним
        if df_result['wind_forecast_three_hours'].isna().any():
            window_size = min(6, len(df_result) // 10)
            df_result['wind_forecast_three_hours'] = df_result['wind_forecast_three_hours'].fillna(
                df_result['wind_forecast_three_hours'].rolling(window=window_size, min_periods=1).mean()
            )
        
        # Последняя проверка - если все еще есть пропуски, заполняем средним
        if df_result['wind_forecast_three_hours'].isna().any():
            mean_val = df_result['wind_forecast_three_hours'].mean()
            df_result['wind_forecast_three_hours'].fillna(mean_val, inplace=True)
    
        # Переводим скорость ветра из км/ч в метры в секунду
        df_result['wind_forecast_three_hours'] = df_result['wind_forecast_three_hours'] / 3.6
        
        # Статистика
        filled_count = df_result['wind_forecast_three_hours'].notna().sum()
        print(f"\nИТОГОВАЯ СТАТИСТИКА:")
        print(f"   • Заполнено записей: {filled_count}/{len(df_result)} ({filled_count/len(df_result)*100:.1f}%)")
        print(f"   • Среднее значение прогноза: {df_result['wind_forecast_three_hours'].mean():.2f} м/с")
        print(f"   • Min/Max: {df_result['wind_forecast_three_hours'].min():.2f} / {df_result['wind_forecast_three_hours'].max():.2f} м/с")
    
        self.df = df_result.reset_index()
        
        return self

    def _compare_wind_speed_and_forecast(self):
        print("\n" + "=" * 50)
        print("СРАВНЕНИЕ СКОРОСТИ ВЕТРА И ЕЁ ПРОГНОЗА")
        print("(НА КОРОТКИХ ВРЕМЕННЫХ УЧАСТКАХ)")
        print("=" * 50)
        
        # Создаём копию dataframe чтобы не модифицировать оригинал
        df_copy = self.df.copy()
        
        # Преобразуем колонку date в datetime индекс
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy = df_copy.set_index('date').sort_index()
        
        # Вычисляем разницы между последовательными точками
        time_diffs = df_copy.index.to_series().diff()
        
        # Ищем последовательности с разницей 1 минута
        one_minute = pd.Timedelta(minutes=1)
        continuous_blocks = (time_diffs != one_minute).cumsum()
        
        # Размеры каждого блока непрерывных данных
        block_sizes = continuous_blocks.value_counts().sort_values(ascending=False)
        
        duration_in_hours = 24  # 1 сутки (было 27)
        duration_in_minutes = duration_in_hours * 60
        
        # Находим все блоки достаточной длины
        valid_blocks = []
        for block_id, size in block_sizes.items():
            if size >= duration_in_minutes:
                block_data = df_copy[continuous_blocks == block_id]
                if len(block_data) >= duration_in_minutes:
                    block_data_limited = block_data.head(duration_in_minutes)
                    valid_blocks.append((block_id, block_data_limited))
        
        if len(valid_blocks) == 0:
            print("Не найдено непрерывных участков достаточной длины")
            return self
        
        # Берем первые 4 блока (или меньше, если их меньше)
        n_plots = min(4, len(valid_blocks))
        selected_blocks = valid_blocks[:n_plots]
        
        # Создаем сетку графиков 2x2
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (block_id, block_data) in enumerate(selected_blocks):
            if idx >= 4:
                break
            
            # Создаем копию блока для работы
            continuous_data = block_data.copy()
            
            # Сдвигаем на 3 часа прогноз (180 минут)
            continuous_data['wind_forecast_three_hours_shifted'] = continuous_data['wind_forecast_three_hours'].shift(180)
            # Удаляем первые 180 строк (где прогноз сдвинут и нет данных)
            continuous_data = continuous_data.iloc[180:]
            
            # Определяем время начала и конца участка
            start_time = continuous_data.index[0]
            end_time = continuous_data.index[-1]
            
            # Строим график
            ax = axes[idx]
            
            ax.plot(continuous_data.index, continuous_data['wind_speed_horizontal'], 
                    label='Горизонтальная скорость ветра', linewidth=2, alpha=0.8, color='blue')
            ax.plot(continuous_data.index, continuous_data['wind_forecast_three_hours_shifted'], 
                    label='Спрогнозированная скорость ветра', linewidth=2, alpha=0.8, color='red')
            
            # Форматирование оси X - показываем каждый день
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
            
            ax.set_xlabel('Время', fontsize=10)
            ax.set_ylabel('Скорость ветра, м/с', fontsize=10)
            ax.set_title(f'Участок {idx+1}: {start_time.strftime("%Y-%m-%d %H:%M")} - {end_time.strftime("%Y-%m-%d %H:%M")}', 
                         fontsize=11, pad=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=0)
        
        # Скрываем пустые подграфики (если блоков меньше 4)
        for idx in range(len(selected_blocks), 4):
            axes[idx].set_visible(False)
        
        plt.suptitle('Сравнение скорости ветра и её прогноза', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
        return self

    def _visualize_data(self):
        # Визуализация данных
        print("\n" + "=" * 50)
        print("ВИЗУАЛИЗАЦИЯ ДАННЫХ")
        print("=" * 50)
        
        # Получаем список колонок для построения графиков
        columns_to_plot = self.df.columns[1:]

        # Создаём subplot с нужным количеством графиков
        fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(12, 4 * len(columns_to_plot)))
        
        # Преобразуем первую колонку в datetime
        date_column = self.df.columns[0]
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        
        # Создаём форматтер для дат
        date_format = DateFormatter('%d-%m %H:%M')
        
        # Строим графики для каждой колонки
        for i, column in enumerate(columns_to_plot):
            axes[i].plot(self.df[date_column], self.df[column])
            axes[i].set_title(f'График: {column}')
            axes[i].set_xlabel('Дата')
            axes[i].set_ylabel(column)
            axes[i].grid(True, alpha=0.3)
            
            # Применяем форматирование дат
            axes[i].xaxis.set_major_formatter(date_format)
            
            # Автоподбор расположения меток и поворот
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha='center')
        
        plt.tight_layout()
        plt.show()

        return self

    def _scale_data(self):
        print("\n" + "=" * 50)
        print("МАСШТАБИРОВАНИЕ ДАННЫХ")
        print("=" * 50)

        # Столбцы для исключения
        exclude_columns = ['date', 'sin_time_of_day', 'cos_time_of_day']

        for column in self.df.columns[1:]:
            if column not in exclude_columns:
                print(f"Обрабатываем колонку: {column}")
                y1 = self.minmax_for_scaling_dict[column][0]
                y2 = self.minmax_for_scaling_dict[column][1]
                self.df[column] = ((2 * (self.df[column] - y1)) / (y2 - y1)) - 1
                # Округляем числа
                self.df[column] = round(self.df[column], 8)
        
        print("\nНормировка данных в [-1, 1] успешно выполнена")
        return self

    def _missing_values_check(self):
        # Проверка пропущенных значений
        print("\n" + "=" * 50)
        print("ПРОВЕРКА NAN-ПРОПУСКОВ")
        print("=" * 50)

        missing = self.df.isna().sum()
        missing_total = missing.sum()
        if missing_total > 0:
            missing_pct = round((missing / len(self.df)) * 100, 1)
            missing_df = pd.DataFrame({
                'Количество NaN-пропусков': missing,
                'Процент NaN-пропусков': missing_pct
            }).sort_values('Процент NaN-пропусков', ascending=False)
            print(missing_df[missing_df['Количество NaN-пропусков'] > 0])
        else:
            print('Количество NaN-пропусков: 0')
            
        return self

    def _display_number_of_continuous_segments(self):
        print("\n" + "=" * 50)
        print("ПОДСЧЁТ НЕПЕРЕСЕКАЮЩИХСЯ ОТРЕЗКОВ")
        print("С НЕПРЕРЫВНЫМИ ПОМИНУТНЫМИ ДАННЫМИ")
        print("=" * 50)

        # Список длин отрезков (в часах)
        hours_list = [4, 6, 12, 24, 48]

        for hours in hours_list:
            total_segments = self.__count_continuous_segments(hours)
            print(f"Количество {hours}-часовых отрезков: {total_segments}")

        return self

    def _save_preprocessed_data(self):
        print("\n" + "=" * 50)
        print("СОХРАНЕНИЕ ПРЕДОБРАБОТАННЫХ ДАННЫХ")
        print("=" * 50)

        self.df.to_csv(self.output_file_path, index=False)

        print(f"Предобработанные данные успешно сохранены в файл {self.output_file_path}")
        return self

    def __get_historical_forecast(self, lat, lon, target_dates):
        """
        Получение архивных прогнозов
        """
        # Настройка кэша и повторных запросов
        cache_session = requests_cache.CachedSession('.gfs_cache', expire_after=86400)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        
        # Определяем временной диапазон
        start_date = min(target_dates)
        end_date = max(target_dates)
        
        print(f"Загрузка архивных прогнозов с {start_date} по {end_date} ...")
        
        # Параметры для запроса исторических прогнозов
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "hourly": "wind_speed_10m",
            "models": "gfs_seamless",
            "timezone": "auto"
        }
        
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            print(f"Получены данные для {response.Latitude()}°N, {response.Longitude()}°E")
            
            # Извлекаем данные
            hourly = response.Hourly()
            time_range = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit='s'),
                end=pd.to_datetime(hourly.TimeEnd(), unit='s'),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive='left'
            )
            
            wind_speeds = hourly.Variables(0).ValuesAsNumpy()
            
            # Создаем DataFrame с прогнозами
            forecast_df = pd.DataFrame({
                'forecast_time': time_range,
                'wind_speed_forecast': wind_speeds
            })
            
            return forecast_df.set_index('forecast_time')
            
        except Exception as e:
            print(f"Ошибка при загрузке данных GFS: {e}")
            return None

    def __count_continuous_segments(self, hours):
        # hours: Длина отрезка

        df_temp = self.df.copy()

        # Создаём числовое представление времени в минутах
        df_temp['minutes'] = (df_temp['date'] - df_temp['date'].min()).dt.total_seconds() / 60
        
        # Находим разрывы (более 1 минуты между записями)
        gaps = df_temp['minutes'].diff() > 1
        df_temp['segment'] = gaps.cumsum()
        
        total_segments = 0
        minutes_in_segment = hours * 60  # Количество минут в отрезке
        
        for segment_id in df_temp['segment'].unique():
            segment = df_temp[df_temp['segment'] == segment_id]
            
            if len(segment) < minutes_in_segment:  # Меньше нужного количества данных
                continue
                
            # Преобразуем минуты в целые числа для более точной работы
            minutes = segment['minutes'].round(6).values
            
            # Ищем последовательные блоки
            start_idx = 0
            segment_count = 0
            
            while start_idx <= len(segment) - minutes_in_segment:
                # Проверяем блок из N последовательных минут
                expected_end = minutes[start_idx] + (minutes_in_segment - 1)
                
                if minutes[start_idx + minutes_in_segment - 1] == expected_end:
                    segment_count += 1
                    start_idx += minutes_in_segment  # Перескакиваем чтобы избежать пересечений
                else:
                    start_idx += 1
                    
            total_segments += segment_count
        
        return total_segments

    def run_preprocessing_pipeline(self):
        # Запуск PREPROCESSING-Pipeline
        print("***** ЗАПУСК PREPROCESSING-PIPELINE *****\n")
        
        (self._load_data()
          ._modify_date_column()
          ._delete_old_data()
          ._delete_duplicates()
          ._remove_unnecessary_features()
          ._add_day_of_year_feature()
          ._add_pressure_derivative()
          ._remove_outliers()
          ._fill_in_short_time_gaps()
          ._remove_consecutive_missing_values()
          ._fill_missing_values()
          ._smooth_noisy_features()
          ._add_wind_forecast_column()
          ._compare_wind_speed_and_forecast()
          ._visualize_data()
          ._scale_data()
          ._missing_values_check()
          ._display_number_of_continuous_segments()
          ._save_preprocessed_data()
        )

        print("\n***** PREPROCESSING-PIPELINE УСПЕШНО ЗАВЕРШЁН! *****\n")