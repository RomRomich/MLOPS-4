# Импортируем нужные библиотеки
import pandas as pd
import os

# Ссылка для файла train.csv
url_train = 'https://drive.google.com/file/d/1FY_mEbFRs7Ze9wuAPEuKd2d6H68spYv9/view?usp=sharing'
file_id_train = url_train.split('/')[-2]
url = 'https://drive.google.com/uc?id=' + file_id_train
# Путь для сохранения файла
datasets_dir = os.path.expanduser('~/datasets')
data_csv_path = os.path.join(datasets_dir, 'data.csv')

# Загрузка датасета
df = pd.read_csv(url)

# Сохраняем данные
df.to_csv(data_csv_path, index=False)

print("Датасет успешно загружен и сохранен по пути:", data_csv_path)
