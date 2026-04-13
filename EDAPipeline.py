import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.dates import DateFormatter
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class EDAPipeline:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        
    def _load_data(self):
        print("=" * 50)
        print("ЗАГРУЗКА ДАННЫХ")
        print("=" * 50)

        # Читаем датафрейм из файла
        self.df = pd.read_csv(self.file_path)
        
        print(f"Данные успешно загружены из файла {self.file_path}")
        return self

    def _basic_info(self):
        # Базовая информация о данных
        print("\n" + "=" * 50)
        print("БАЗОВАЯ ИНФОРМАЦИЯ О ДАННЫХ")
        print("=" * 50)

        # Размеры датасета, структура данных, несколько случайных строк
        print(f"Размер: {self.df.shape}")
        print(f"Количество признаков: {self.df.shape[1]}")
        print(f"Количество наблюдений: {self.df.shape[0]}")
    
        print("\nСТРУКТУРА ДАННЫХ (df.info()):")
        print("-" * 40)
        self.df.info()
    
        print("\nСЛУЧАЙНЫЕ 3 СТРОКИ (df.sample(3)):")
        print("-" * 40)
        sample_train = self.df.sample(3, random_state=42)
        display(sample_train)
    
        # Временной период
        print(f"\nВРЕМЕННОЙ ПЕРИОД:")
        print(f"{self.df['date'].min()} - {self.df['date'].max()}")
    
        return self

    def _missing_values_analysis(self):
        # Анализ пропущенных значений
        print("\n" + "=" * 50)
        print("АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
        print("=" * 50)

        missing = self.df.isna().sum()
        missing_total = missing.sum()
        if missing_total > 0:
            missing_pct = round((missing / len(self.df)) * 100, 1)
            missing_df = pd.DataFrame({
                'Количество пропусков': missing,
                'Процент пропусков': missing_pct
            }).sort_values('Процент пропусков', ascending=False)
            print(missing_df[missing_df['Количество пропусков'] > 0])
        else:
            print('Количество пропусков: 0')
            
        return self

    def _time_gaps_analysis(self):
        # Анализ временных пропусков
        print("\n" + "=" * 50)
        print("АНАЛИЗ ВРЕМЕННЫХ ПРОПУСКОВ")
        print("=" * 50)
    
        dates = pd.to_datetime(self.df['date']).sort_values().reset_index(drop=True)
    
        gaps = []
        expected_td = pd.Timedelta(minutes=1)
    
        for i in range(1, len(dates)):
            time_diff = dates[i] - dates[i-1]
        
            # Если разница больше ожидаемого интервала (плюс небольшой запас)
            if time_diff > expected_td + pd.Timedelta(seconds=10):
                gap_duration_minutes = time_diff.total_seconds() / 60
                missing_intervals = int(gap_duration_minutes) - 1
                
                gaps.append({
                    'gap_id': len(gaps) + 1,
                    'start_time': dates[i-1],
                    'end_time': dates[i],
                    'gap_duration_minutes': round(gap_duration_minutes, 2),
                    'missing_records': missing_intervals,
                    'actual_interval_minutes': round(time_diff.total_seconds() / 60, 2),
                    'position_between': f"({i-1}) - ({i})"
                })
    
        if not gaps:
            print("Временных пропусков не найдено")
        else:
            # Группировка пропусков по длительности
            short_gaps = []   # до 10 минут
            medium_gaps = []  # 10-60 минут
            long_gaps = []    # 1-6 часов
            very_long_gaps = []  # более 6 часов
            
            for gap in gaps:
                duration = gap['gap_duration_minutes']
                if duration <= 10:
                    short_gaps.append(gap)
                elif duration <= 60:
                    medium_gaps.append(gap)
                elif duration <= 360:  # 6 часов
                    long_gaps.append(gap)
                else:
                    very_long_gaps.append(gap)
            
            # Вывод статистики
            print(f"\nСТАТИСТИКА ВРЕМЕННЫХ ПРОПУСКОВ:")
            print(f"   • Всего пропусков: {len(gaps)}")
            print(f"   • Суммарная длительность: {sum(g['gap_duration_minutes'] for g in gaps):.1f} мин "
                  f"({sum(g['gap_duration_minutes'] for g in gaps)/60:.1f} часов)")
            print(f"   • Потеряно записей: {sum(g['missing_records'] for g in gaps)}")
            
            print(f"\nРАСПРЕДЕЛЕНИЕ ПО ДЛИТЕЛЬНОСТИ:")
            
            if short_gaps:
                total_short_min = sum(g['gap_duration_minutes'] for g in short_gaps)
                print(f"   • Короткие пропуски (до 10 мин): {len(short_gaps)} шт, "
                      f"всего {total_short_min:.1f} мин, "
                      f"потеряно {sum(g['missing_records'] for g in short_gaps)} записей")
            
            if medium_gaps:
                total_medium_min = sum(g['gap_duration_minutes'] for g in medium_gaps)
                print(f"   • Средние пропуски (10-60 мин): {len(medium_gaps)} шт, "
                      f"всего {total_medium_min:.1f} мин, "
                      f"потеряно {sum(g['missing_records'] for g in medium_gaps)} записей")
            
            if long_gaps:
                total_long_min = sum(g['gap_duration_minutes'] for g in long_gaps)
                print(f"   • Длинные пропуски (1-6 часов): {len(long_gaps)} шт, "
                      f"всего {total_long_min:.1f} мин ({total_long_min/60:.1f} час), "
                      f"потеряно {sum(g['missing_records'] for g in long_gaps)} записей")
            
            if very_long_gaps:
                total_vlong_min = sum(g['gap_duration_minutes'] for g in very_long_gaps)
                print(f"   • Очень длинные пропуски (>6 часов): {len(very_long_gaps)} шт, "
                      f"всего {total_vlong_min:.1f} мин ({total_vlong_min/60:.1f} час), "
                      f"потеряно {sum(g['missing_records'] for g in very_long_gaps)} записей")
        
        return self

    def _duplicates_check(self):
        # Проверка на дубликаты
        print("\n" + "=" * 50)
        print("ПРОВЕРКА НА ДУБЛИКАТЫ")
        print("=" * 50)

        print(f"Количество дубликатов: {self.df.duplicated().sum()}")
        
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

    def _features_values_distribution(self):
        # Распределение значений переменных
        print("\n" + "=" * 50)
        print("РАСПРЕДЕЛЕНИЕ ЗНАЧЕНИЙ ПЕРЕМЕННЫХ")
        print("=" * 50)
        
        # Числовые колонки
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
    
        # Основные статистики
        print("\nОСНОВНЫЕ СТАТИСТИКИ:")
        print(self.df[numerical_cols].describe())
    
        # Визуализация распределений
        n_cols = len(numerical_cols)
        n_rows = (n_cols + 3) // 4
    
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, n_rows * 4))
        axes = axes.flatten()

        # Гистограммы
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                axes[i].hist(self.df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Распределение {col}')
                axes[i].set_xlabel(col)

        # Скрываем пустые subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
    
        plt.tight_layout()
        plt.show()
        
        return self

    def _boxplots_and_outliers(self):
        # Ящики с усами и выбросы
        print("\n" + "=" * 50)
        print("ЯЩИКИ С УСАМИ И ВЫБРОСЫ")
        print("=" * 50)

        # Числовые колонки
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        n_cols = len(numerical_cols)
        n_rows = (n_cols + 3) // 4

        # Визуализация
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, n_rows * 4))
        axes = axes.flatten()

        # Ящики с усами
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                self.df.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'Boxplot {col}')
    
        # Скрываем пустые subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
    
        plt.tight_layout()
        plt.show()

        # Выбросы (по методу IQR)
        outliers_list = []
        outliers_pct_list = []
        for col in numerical_cols:
            feature = self.df[col]
            Q1 = feature.quantile(0.25)
            Q3 = feature.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = len(feature[(feature < lower_bound) | (feature > upper_bound)])
            outliers_pct = round((outliers / len(feature)) * 100, 1)
            outliers_list.append(outliers)
            outliers_pct_list.append(outliers_pct)

        print("\nВЫБРОСЫ (ПО МЕТОДУ IQR):")
        outliers_df = pd.DataFrame({
            'Признак': numerical_cols,
            'Количество выбросов': outliers_list,
            'Процент выбросов': outliers_pct_list
        })
        print(outliers_df.to_string(index=False))

        return self

    def _correlation_analysis(self):
        # Анализ корреляций
        print("\n" + "=" * 50)
        print("АНАЛИЗ КОРРЕЛЯЦИЙ")
        print("=" * 50)

        numeric_df = self.df.select_dtypes(include=[np.number])

        # Матрица корреляций
        correlation_matrix = numeric_df.corr().round(2)

        # Визуализация
        plt.figure(figsize=(12, 10))
    
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
            center=0, fmt='.2f', square=True,
            annot_kws={'size': 12, 'weight': 'bold'})
    
        plt.title('Матрица корреляций', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(fontsize=12, rotation=90)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

        # Создаём DataFrame с сильно коррелирующими парами
        correlation_threshold = 0.8
        
        # Преобразуем матрицу в long format
        corr_pairs = correlation_matrix.unstack().reset_index()
        corr_pairs.columns = ['Признак 1', 'Признак 2', 'Корреляция']
        
        # Убираем дубликаты: оставляем только где Признак 1 < Признак 2
        corr_pairs = corr_pairs[corr_pairs['Признак 1'] < corr_pairs['Признак 2']]
        
        # Фильтруем по порогу корреляции
        strong_corr_df = corr_pairs[abs(corr_pairs['Корреляция']) > correlation_threshold]\
            .sort_values('Корреляция', key=abs, ascending=False)
        
        print("СИЛЬНО КОРРЕЛИРУЮЩИЕ ПРИЗНАКИ:")
        if not strong_corr_df.empty:
            print(strong_corr_df.to_string(index=False))
        else:
            print(f"Нет пар с корреляцией > {correlation_threshold}")

        return self

    def _time_series_autocorrelation(self):
        # Автокорреляция временных рядов
        print("\n" + "=" * 50)
        print("АВТОКОРРЕЛЯЦИЯ ВРЕМЕННЫХ РЯДОВ")
        print("=" * 50)

        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        nlags = 144 # Максимальный лаг для анализа (144 минуты = 24 часа)
        n_cols = 3 # Количество колонок в сетке графиков

        print(f"Максимальный лаг: {nlags} ({(nlags/6):.1f} часов)")

        # Результаты анализа
        results = []
        
        # Создание сетки графиков
        n_plots = len(numeric_columns)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.reshape(-1, n_cols) if n_rows > 1 else axes.reshape(1, -1)

        for i, col in enumerate(numeric_columns):
            row_acf = (i // n_cols) * 2
            row_pacf = (i // n_cols) * 2 + 1
            col_pos = i % n_cols
            
            # Подготовка данных
            data = self.df[col].dropna()
            
            if len(data) < nlags * 2:
                continue
                
            try:
                # Расчёт ACF и PACF
                acf_values = acf(data, nlags=nlags, fft=True)
                pacf_values = pacf(data, nlags=nlags)
                
                # Статистики
                significance_threshold = 2 / np.sqrt(len(data))
                significant_acf_lags = np.where(np.abs(acf_values[1:]) > significance_threshold)[0] + 1
                significant_pacf_lags = np.where(np.abs(pacf_values[1:]) > significance_threshold)[0] + 1
                
                # Классификация
                acf_lag1 = acf_values[1]
                acf_lag24 = acf_values[24] if 24 < len(acf_values) else np.nan
                
                if abs(acf_lag1) < 0.1:
                    classification = "Белый шум"
                elif acf_lag1 > 0.8:
                    classification = "Сильная персистентность"
                elif not pd.isna(acf_lag24) and abs(acf_lag24) > 0.2:
                    classification = "Сезонность 24ч"
                elif acf_lag1 > 0.3:
                    classification = "Автокорреляция"
                else:
                    classification = "Слабая зависимость"
                
                # Сохранение результатов
                results.append({
                    'Признак': col,
                    'ACF лаг 1': acf_lag1,
                    'ACF лаг 24': acf_lag24,
                    'Классификация': classification,
                    'Значимых ACF лагов': len(significant_acf_lags),
                    'Значимых PACF лагов': len(significant_pacf_lags)
                })
                
                # Построение ACF
                plot_acf(data, ax=axes[row_acf, col_pos], lags=nlags, 
                        title=f'ACF: {col}\nLag1: {acf_lag1:.3f}')
                axes[row_acf, col_pos].set_ylim(-1.1, 1.1)
                
                # Построение PACF
                plot_pacf(data, ax=axes[row_pacf, col_pos], lags=nlags, 
                         title=f'PACF: {col}\nClass: {classification}')
                axes[row_pacf, col_pos].set_ylim(-1.1, 1.1)
                
            except Exception as e:
                print(f"Ошибка для {col}: {e}")
                continue

        # Скрытие пустых subplots
        for j in range(i + 1, n_rows * n_cols):
            row_acf = (j // n_cols) * 2
            row_pacf = (j // n_cols) * 2 + 1
            col_pos = j % n_cols
            if row_acf < len(axes) and col_pos < len(axes[row_acf]):
                axes[row_acf, col_pos].set_visible(False)
            if row_pacf < len(axes) and col_pos < len(axes[row_pacf]):
                axes[row_pacf, col_pos].set_visible(False)
        
        plt.tight_layout()
        plt.show()

        # Сводная таблица результатов
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            print("\n" + "="*80)
            print("СВОДКА АВТОКОРРЕЛЯЦИОННОГО АНАЛИЗА")
            print("="*80)
            display_cols = ['Признак', 'ACF лаг 1', 'Классификация']
            print(results_df[display_cols].round(2).to_string(index=False))

        return self

    def _analyze_daily_seasonality(self):
        # Анализ суточной сезонности
        print("\n" + "=" * 50)
        print("АНАЛИЗ СУТОЧНОЙ СЕЗОННОСТИ")
        print("=" * 50)

        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        n_cols = 3 # Количество колонок в сетке графиков

        # Добавление часового признака
        df_analysis = self.df.copy() 
        df_analysis['date'] = pd.to_datetime(df_analysis['date'])
        df_analysis['hour'] = df_analysis['date'].dt.hour
        
        results = []
        
        # Создание сетки графиков
        n_plots = len(numeric_columns)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        for i, col in enumerate(numeric_columns):
            if i >= len(axes):
                break
                
            data = df_analysis[[col, 'hour']].dropna()
            
            if len(data) < 24:
                continue
                
            # Суточные паттерны
            hourly_stats = data.groupby('hour')[col].agg(['mean', 'std', 'min', 'max']).reset_index()
            
            # Статистика сезонности
            amplitude = hourly_stats['mean'].max() - hourly_stats['mean'].min()
            peak_hour = hourly_stats.loc[hourly_stats['mean'].idxmax(), 'hour']
            min_hour = hourly_stats.loc[hourly_stats['mean'].idxmin(), 'hour']
            relative_amplitude = amplitude / data[col].std() if data[col].std() > 0 else 0
            
            # Классификация сезонности
            if relative_amplitude > 1.0:
                seasonality_strength = "Сильная"
            elif relative_amplitude > 0.5:
                seasonality_strength = "Умеренная"
            elif relative_amplitude > 0.2:
                seasonality_strength = "Слабая"
            else:
                seasonality_strength = "Отсутствует"
            
            results.append({
                'Признак': col,
                'Амплитуда': amplitude,
                'Относительная амплитуда': relative_amplitude,
                'Сила сезонности': seasonality_strength,
                'Пиковый час': peak_hour,
                'Минимальный час': min_hour,
                'Разница пик-минимум': f"{peak_hour:02d}:00 - {min_hour:02d}:00"
            })
            
            # Построение графика
            axes[i].plot(hourly_stats['hour'], hourly_stats['mean'], 
                        marker='o', linewidth=2, label='Среднее', color='blue')
            axes[i].fill_between(hourly_stats['hour'],
                               hourly_stats['mean'] - hourly_stats['std'],
                               hourly_stats['mean'] + hourly_stats['std'],
                               alpha=0.3, label='±1 стд', color='blue')
            axes[i].set_title(f'{col}\nСуточная сезонность: {seasonality_strength}')
            axes[i].set_xlabel('Час')
            axes[i].set_ylabel(col)
            axes[i].set_xticks(range(0, 24, 3))
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(fontsize=8)

        # Скрытие пустых subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Сводная таблица результатов
        results_df = pd.DataFrame(results)

        if not results_df.empty:
            print("\n" + "="*70)
            print("СВОДКА АНАЛИЗА СУТОЧНОЙ СЕЗОННОСТИ")
            print("="*70)
            display_cols = ['Признак', 'Сила сезонности', 'Пиковый час', 'Минимальный час']
            print(results_df[display_cols].round(4).to_string(index=False))

        return self

    def _stationarity_check(self):
        # Проверка стационарности
        print("\n" + "=" * 50)
        print("ПРОВЕРКА СТАЦИОНАРНОСТИ ВРЕМЕННЫХ РЯДОВ")
        print("=" * 50)
        
        max_points = 5000
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].nunique() <= 1:
                continue
                
            data = self.df[col].dropna()
            if len(data) < 100:
                continue
            
            # Сэмплирование для больших данных
            if len(data) > max_points:
                idx = np.linspace(0, len(data)-1, max_points, dtype=int)
                data = data.iloc[idx]
            
            # Ограничиваем максимальный лаг
            maxlag = min(30, len(data) // 10)
            adf_result = adfuller(data, autolag='AIC', maxlag=maxlag)
            is_stationary = adf_result[1] < 0.05
            
            status = "Стационарный" if is_stationary else "Нестационарный"
            print(f"{col}: {status}")
        
        return self

    def _estimate_optimal_lookback(self):
        # Оценка оптимальной длины исторического окна (lookback window) на основе ACF/PACF
        print("\n" + "=" * 50)
        print("ОЦЕНКА ОПТИМАЛЬНОГО ИСТОРИЧЕСКОГО ОКНА (LOOKBACK)")
        print("=" * 50)
        
        # Исключаем временные признаки
        exclude_columns = ['date', 'sin_time_of_day', 'cos_time_of_day']
        
        # Все числовые признаки
        all_columns = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_columns]
        
        if not all_columns:
            print("Не найдено признаков для анализа")
            return self
        
        # Ограничение на количество точек для анализа
        MAX_POINTS = 5000
        MAX_LAGS = 200  # Максимальное количество лагов для ACF/PACF
        
        print(f"\nАнализируемые признаки:")
        print("-" * 85)
        print(f"{'Признак':<35} {'30 мин':<10} {'1 час':<10} {'2 часа':<10} {'3 часа':<10}")
        print("-" * 85)
        
        all_lookbacks_30min = []
        all_lookbacks_1h = []
        all_lookbacks_2h = []
        all_lookbacks_3h = []
        
        for col in all_columns:
            data = self.df[col].dropna()
            
            if len(data) < 144:
                print(f"{col:<35} {'недостаточно данных':<42}")
                continue
            
            # Сэмплирование для больших данных
            if len(data) > MAX_POINTS:
                idx = np.linspace(0, len(data)-1, MAX_POINTS, dtype=int)
                data = data.iloc[idx]
            
            # Уменьшаем количество лагов для ACF/PACF
            nlags = min(MAX_LAGS, len(data) // 3)
            
            # Используем fft только для больших данных
            use_fft = len(data) > 1000
            acf_values = acf(data, nlags=nlags, fft=use_fft)
            
            # PACF считаем только с ограниченным числом лагов
            pacf_values = pacf(data, nlags=min(nlags, 50))
            
            significance_threshold = 2 / np.sqrt(len(data))
            
            # Объединяем поиск в один проход
            decay_lag_02 = None
            zero_crossing = None
            decay_lag_368 = None
            
            for lag in range(1, min(len(acf_values), MAX_LAGS)):
                abs_val = abs(acf_values[lag])
                
                # Затухание до 0.2
                if decay_lag_02 is None and abs_val < 0.2:
                    decay_lag_02 = lag
                
                # Затухание до 0.368
                if decay_lag_368 is None and abs_val < 0.368:
                    decay_lag_368 = lag
                
                # Переход через ноль
                if zero_crossing is None and lag > 0 and acf_values[lag] * acf_values[lag-1] < 0:
                    zero_crossing = lag
                
                # Ранний выход, если нашли всё
                if decay_lag_02 and zero_crossing and decay_lag_368:
                    break
            
            # PACF порядок за один проход
            pacf_order = 0
            for lag in range(1, min(len(pacf_values), 50)):
                if abs(pacf_values[lag]) > significance_threshold:
                    pacf_order = lag
            
            # Комбинированная оценка
            estimates = []
            if decay_lag_02:
                estimates.append(decay_lag_02)
            if zero_crossing:
                estimates.append(zero_crossing)
            if decay_lag_368:
                estimates.append(decay_lag_368)
            if pacf_order > 0:
                estimates.append(pacf_order * 2)
            
            if estimates:
                char_length = int(np.median(estimates))
            else:
                char_length = 60
            
            char_length = max(30, min(char_length, 360))
            
            # Расчёт lookback (упрощённая формула)
            # Оптимизация 7: Упрощаем расчёт lookback
            if char_length < 60:
                multiplier = 1
            elif char_length < 120:
                multiplier = 1
            else:
                multiplier = 0.5
            
            lookback_30min = max(30, min(int(char_length * (0.5 + multiplier * 0.5)), 240))
            lookback_1h = max(60, min(int(char_length * (1 + multiplier)), 480))
            lookback_2h = max(90, min(int(char_length * (2 + multiplier)), 720))
            lookback_3h = max(120, min(int(char_length * (3 + multiplier)), 1440))
            
            all_lookbacks_30min.append(lookback_30min)
            all_lookbacks_1h.append(lookback_1h)
            all_lookbacks_2h.append(lookback_2h)
            all_lookbacks_3h.append(lookback_3h)
            
            print(f"{col:<35} {lookback_30min:<10} {lookback_1h:<10} {lookback_2h:<10} {lookback_3h:<10}")
        
        # ИТОГОВЫЕ РЕКОМЕНДАЦИИ
        if all_lookbacks_30min:
            recommended_30min = int(np.percentile(all_lookbacks_30min, 90))
            recommended_1h = int(np.percentile(all_lookbacks_1h, 90))
            recommended_2h = int(np.percentile(all_lookbacks_2h, 90))
            recommended_3h = int(np.percentile(all_lookbacks_3h, 90))
            
            print("\nРекомендации по выбору lookback window:")
            print("-"*60)
            print(f"\n{'Горизонт':<12} {'LSTM/GRU':<25} {'Transformer':<25}")
            print("-" * 62)
            print(f"{'30 мин':<12} {recommended_30min} мин ({recommended_30min/60:.1f} ч){'':<6} {min(recommended_30min*2, 480)} мин ({min(recommended_30min*2, 480)/60:.1f} ч)")
            print(f"{'1 час':<12} {recommended_1h} мин ({recommended_1h/60:.1f} ч){'':<6} {min(recommended_1h*2, 720)} мин ({min(recommended_1h*2, 720)/60:.1f} ч)")
            print(f"{'2 часа':<12} {recommended_2h} мин ({recommended_2h/60:.1f} ч){'':<6} {min(recommended_2h*2, 1440)} мин ({min(recommended_2h*2, 1440)/60:.1f} ч)")
            print(f"{'3 часа':<12} {recommended_3h} мин ({recommended_3h/60:.1f} ч){'':<6} {min(recommended_3h*2, 2880)} мин ({min(recommended_3h*2, 2880)/60:.1f} ч)")
            
            print("\n" + "="*60)
        
        return self

    def run_eda_pipeline(self):
        # Запуск EDA-Pipeline
        print("***** ЗАПУСК EDA-PIPELINE *****\n")
        
        (self._load_data()
          ._basic_info()
          ._missing_values_analysis()
          ._time_gaps_analysis()
          ._duplicates_check()
          ._visualize_data()
          ._features_values_distribution()
          ._boxplots_and_outliers()
          ._correlation_analysis()
          ._time_series_autocorrelation()
          ._analyze_daily_seasonality()
          ._stationarity_check()
          ._estimate_optimal_lookback()
        )

        print("\n***** EDA-PIPELINE УСПЕШНО ЗАВЕРШЁН! *****\n")