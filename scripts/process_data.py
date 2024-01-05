
# Импортируем нужные библиотеки
import pandas as pd
import os

# Загрузка данных
datasets_dir = os.path.expanduser('~/datasets')
data_csv_path = os.path.join(datasets_dir, 'data.csv')
df = pd.read_csv(data_csv_path)

# Обновим значения в столбцах "churn" и "international_plan", удалим ненужные столбцы и сохраним нужные 
df["churn"] = df["churn"].map({"no": 0, "yes": 1})
df.drop(["total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge"], axis=1, inplace=True)
df["international_plan"] = df["international_plan"].map({"no": 0, "yes": 1})
df.drop("voice_mail_plan", axis=1, inplace=True)
df.drop(["state", "area_code"], axis=1, inplace=True)

# Запишем обработанные данные в файл по указанному пути
data_csv_path_proc = os.path.join(datasets_dir, 'data_proc.csv')
df.to_csv(data_csv_path_proc, index=False)

print("Датасет успешно обработан и сохранен по пути:", data_csv_path_proc)
