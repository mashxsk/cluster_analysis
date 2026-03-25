import kagglehub
import sys
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QLabel, QPushButton, QGroupBox,
    QHeaderView, QMessageBox, QTextEdit, QDialog, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

try:
    print("Завантаження датасету з Kaggle...")
    path = kagglehub.dataset_download("nelgiriyewithana/countries-of-the-world-2023")
    print("Path to dataset files:", path)

    download_path = path
    csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("У завантаженому каталозі не знайдено файлів .csv")
    DATA_FILE_PATH = os.path.join(download_path, csv_files[0])
    print(f"Знайдено файл даних: {DATA_FILE_PATH}")
except Exception as e:
    print(f"Критична помилка завантаження даних з Kaggle: {e}")
    DATA_FILE_PATH = r'C:\Users\Маша\.cache\kagglehub\datasets\nelgiriyewithana\countries-of-the-world-2023\versions\1\world-data-2023.csv'
    print(f"Використано резервний шлях: {DATA_FILE_PATH}")


class CountryDataApp(QMainWindow):
    def __init__(self, data_path):
        super().__init__()
        self.setWindowTitle("📈 Кластеризація Країн Світу (Ієрархічний метод)")
        self.setGeometry(100, 100, 1400, 800)
        self.data_path = data_path
        self.data = self.load_data()
        self.last_report_text = ""
        self.features_for_clustering = [
            'GDP', 'Unemployment_rate', 'Life_expectancy', 'Birth_Rate',
            'Fertility_Rate', 'CPI', 'Tax_revenue_(%)', 'Physicians_per_thousand'
        ]

        if self.data is None:
            return
        self.setup_ui()
        self.display_data(self.data)

    def load_data(self):
        try:
            df = pd.read_csv(self.data_path, sep=',', encoding='utf-8')
            df.columns = df.columns.str.strip().str.replace(' ', '_', regex=False)
            df['Country'] = df['Country'].str.strip()

            cols_to_clean = [
                'Density(P/Km2)', 'Agricultural_Land(%)', 'Birth_Rate', 'Co2-Emissions', 'CPI',
                'CPI_Change_(%)', 'Fertility_Rate', 'Forested_Area_(%)', 'Gasoline_Price', 'GDP',
                'Gross_primary_education_enrollment_(%)', 'Gross_tertiary_education_enrollment_(%)',
                'Infant_mortality', 'Life_expectancy', 'Maternal_mortality_ratio', 'Minimum_wage',
                'Out_of_pocket_health_expenditure', 'Physicians_per_thousand', 'Population',
                'Population:_Labor_force_participation_(%)', 'Tax_revenue_(%)', 'Total_tax_rate',
                'Unemployment_rate', 'Urban_population'
            ]
            for col in cols_to_clean:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except FileNotFoundError:
            QMessageBox.critical(self, "Помилка Файлу",
                                 f"Файл даних не знайдено за шляхом: {self.data_path}. Переконайтеся, що шлях коректний.")
            return None
        except Exception as e:
            QMessageBox.critical(self, "Помилка Завантаження", f"Сталася помилка при обробці даних: {e}")
            return None

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        header_label = QLabel("📊 Аналіз Країн Світу За Економічно-Соціальним показником")
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header_label)

        control_layout = QHBoxLayout()
        self.clustering_group = QGroupBox("Параметри Кластеризації")
        group_layout = QHBoxLayout(self.clustering_group)

        cluster_label = QLabel("Кількість Кластерів: 3")
        group_layout.addWidget(cluster_label)

        self.cluster_button = QPushButton("Виконати Кластеризацію")
        self.cluster_button.setStyleSheet("background-color: #e1f5fe; font-weight: bold; padding: 5px;")
        self.cluster_button.clicked.connect(self.run_clustering)
        group_layout.addWidget(self.cluster_button)

        self.report_button = QPushButton("Показати детальний звіт")
        self.report_button.setEnabled(False)
        self.report_button.clicked.connect(self.show_report_window)
        group_layout.addWidget(self.report_button)

        self.dendrogram_button = QPushButton("📉 Показати Дендрограму")
        self.dendrogram_button.setStyleSheet("background-color: #fff9c4; font-weight: bold; padding: 5px;")
        self.dendrogram_button.clicked.connect(self.show_dendrogram_window)
        group_layout.addWidget(self.dendrogram_button)

        control_layout.addWidget(self.clustering_group)
        control_layout.addStretch(1)
        main_layout.addLayout(control_layout)

        self.table_widget = QTableWidget()
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.table_widget)

        self.status_label = QLabel(f"Завантажено: {len(self.data)} записів. Очікування кластеризації.")
        self.status_label.setStyleSheet("font-style: italic; color: gray; margin-top: 10px;")
        main_layout.addWidget(self.status_label)

    def display_data(self, df):
        if df is None or df.empty:
            self.table_widget.setRowCount(0)
            self.table_widget.setColumnCount(0)
            return
        self.table_widget.setRowCount(len(df))
        self.table_widget.setColumnCount(len(df.columns))
        self.table_widget.setHorizontalHeaderLabels(df.columns.tolist())

        color_map = {
            0: QColor(255, 204, 204),
            1: QColor(204, 255, 204),
            2: QColor(204, 204, 255),
        }

        cluster_col_index = df.columns.get_loc('Cluster_Label') if 'Cluster_Label' in df.columns else -1

        for i, row in df.iterrows():
            row_color = QColor(255, 255, 255)
            if cluster_col_index != -1:
                try:
                    cluster_label = int(row['Cluster_Label'])
                    row_color = color_map.get(cluster_label, QColor(220, 220, 220))
                except (ValueError, TypeError):
                    pass

            for j, value in enumerate(row):
                if isinstance(value, (int, float, np.int64, np.float64)):
                    item = QTableWidgetItem(f"{value:,.2f}")
                else:
                    item = QTableWidgetItem(str(value))
                item.setBackground(row_color)
                self.table_widget.setItem(i, j, item)

        self.status_label.setText(f"Завантажено: {len(df)} записів.")

    def prepare_clustering_data(self, df):
        available_features = [f for f in self.features_for_clustering if f in df.columns]
        if not available_features:
            raise ValueError("Відсутні необхідні ознаки для кластеризації. Перевірте назви стовпців.")

        df_cluster = df[available_features].copy()
        df_cluster = df_cluster.fillna(df_cluster.mean())

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_cluster)
        return scaled_data

    def run_clustering(self):
        try:
            self.status_label.setText("Виконання кластеризації... Зачекайте.")
            QApplication.processEvents()

            scaled_data = self.prepare_clustering_data(self.data)
            n_clusters = 3
            agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = agg_clustering.fit_predict(scaled_data)

            result_df = self.data.copy()
            if 'Cluster_Label' in result_df.columns:
                result_df = result_df.drop(columns=['Cluster_Label'])
            result_df['Cluster_Label'] = cluster_labels

            sil_score = silhouette_score(scaled_data, cluster_labels)

            cols_for_stats = self.features_for_clustering + ['Cluster_Label']
            cols_for_stats = [c for c in cols_for_stats if c in result_df.columns]

            cluster_means = result_df[cols_for_stats].groupby('Cluster_Label').mean()

            output_file = "clustering_results.csv"
            result_df.to_csv(output_file, index=False, encoding='utf-8')

            clusters = {}
            for i in range(n_clusters):
                cluster_data = result_df[result_df['Cluster_Label'] == i]
                cluster_data.to_csv(f"cluster_{i}.csv", index=False, encoding='utf-8')
                clusters[i] = cluster_data

            self.display_data(result_df)

            cluster_col_index = result_df.columns.get_loc('Cluster_Label')
            header_item = self.table_widget.horizontalHeaderItem(cluster_col_index)
            if header_item:
                header_item.setText("Cluster (РЕЗУЛЬТАТ)")

            self.last_report_text = f"✅ РЕЗУЛЬТАТИ КЛАСТЕРИЗАЦІЇ\n"
            self.last_report_text += f"📊 Якість розбиття (Silhouette): {sil_score:.3f}\n"
            self.last_report_text += "(Пояснення: > 0.5 - відмінно, 0.25-0.5 - нормально, < 0.25 - слабко)\n"
            self.last_report_text += "=" * 60 + "\n"

            for label, row in cluster_means.iterrows():
                count = len(clusters[label])
                self.last_report_text += f"\n🔷 КЛАСТЕР {label} (Кількість країн: {count})\n"
                self.last_report_text += "-" * 40 + "\n"

                if 'GDP' in row:
                    self.last_report_text += f"   💰 ВВП (середній):       ${row['GDP']:,.0f}\n"
                if 'Life_expectancy' in row:
                    self.last_report_text += f"   ❤️ Тривалість життя:     {row['Life_expectancy']:.1f} років\n"
                if 'Birth_Rate' in row:
                    self.last_report_text += f"   👶 Народжуваність:       {row['Birth_Rate']:.2f}\n"
                if 'Unemployment_rate' in row:
                    self.last_report_text += f"   📉 Безробіття:           {row['Unemployment_rate']:.2f}%\n"

                examples = clusters[label]['Country'].head(5).tolist()
                self.last_report_text += f"\n   🌍 Приклади країн: {', '.join(examples)}, ...\n"
                self.last_report_text += "=" * 60 + "\n"

            self.report_button.setEnabled(True)
            self.report_button.setStyleSheet("background-color: #c8e6c9; font-weight: bold; padding: 5px;")
            self.status_label.setText(f"Успіх! Файли збережено. Натисніть кнопку звіту для деталей.")

            QMessageBox.information(self, "Готово",
                                    "Кластеризацію завершено!\nНатисніть кнопку 'Показати детальний звіт' зверху, щоб побачити результати.")

        except ValueError as ve:
            QMessageBox.warning(self, "Помилка Кластеризації", str(ve))
            self.status_label.setText("Кластеризація завершилась помилкою.")
        except Exception as e:
            QMessageBox.critical(self, "Критична Помилка", f"Невідома помилка: {e}")
            print(e)
            self.status_label.setText("Критична помилка.")

    def show_report_window(self):
        if not self.last_report_text:
            return

        report_dialog = QDialog(self)
        report_dialog.setWindowTitle("Детальний Звіт Кластеризації")
        report_dialog.resize(900, 700)

        layout = QVBoxLayout()

        title = QLabel("Детальна статистика по кластерах")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        text_area = QTextEdit()
        text_area.setPlainText(self.last_report_text)
        text_area.setReadOnly(True)

        font = QFont("Consolas", 12)
        font.setStyleHint(QFont.StyleHint.Monospace)
        text_area.setFont(font)

        layout.addWidget(text_area)

        close_btn = QPushButton("Закрити")
        close_btn.clicked.connect(report_dialog.accept)
        layout.addWidget(close_btn)

        report_dialog.setLayout(layout)
        report_dialog.exec()

    def show_dendrogram_window(self):
        try:

            self.status_label.setText("Побудова дендрограми...")
            QApplication.processEvents()

            scaled_data = self.prepare_clustering_data(self.data)

            linked = linkage(scaled_data, method='ward')

            plt.figure(figsize=(15, 8))
            plt.title('Дендрограма ієрархічної кластеризації країн')
            plt.xlabel('Країни')
            plt.ylabel('Євклідова відстань (Ward linkage)')

            labels = self.data['Country'].tolist() if 'Country' in self.data.columns else None

            dendrogram(
                linked,
                orientation='top',
                labels=labels,
                distance_sort='descending',
                show_leaf_counts=True,
                leaf_rotation=90.,
                leaf_font_size=8.,
            )

            plt.axhline(y=18, c='red', lw=1, linestyle='dashed', label='Лінія відсікання')
            plt.legend()
            plt.tight_layout()

            self.status_label.setText("Дендрограму побудовано.")
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Помилка Дендрограми", f"Не вдалося побудувати графік: {e}")
            self.status_label.setText("Помилка при побудові дендрограми.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CountryDataApp(DATA_FILE_PATH)
    if window.data is not None:
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit(1)