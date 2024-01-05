# Импортируем нужные библиотеки
import pandas as pd
import os
import pickle
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Загрузим данные обучающей выборки из csv-файлов и сохраним их в переменные X_train и y_train 
datasets_dir = os.path.expanduser('~/datasets')
data_csv_path_X_train = os.path.join(datasets_dir, 'data_X_train.csv')
data_csv_path_y_train = os.path.join(datasets_dir, 'data_y_train.csv')
X_train = pd.read_csv(data_csv_path_X_train)
y_train = pd.read_csv(data_csv_path_y_train)

# Создадим и обучим модель градиентного бустинга на основе алгоритма XGBoost на тренировочных данных 
gbdt_clf = XGBClassifier()
gbdt_clf.fit(X_train, y_train)


# Сохранение обученной модели
model_output_path = os.path.expanduser('~/models/model.pkl')
with open(model_output_path, "wb") as model_file:
    pickle.dump(gbdt_clf, model_file)

print("Модель сохранена по пути:", model_output_path)