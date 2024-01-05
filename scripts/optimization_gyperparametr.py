
# Импортируем нужные библиотеки
import pandas as pd
import os
import pickle
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Загружаем данные
datasets_dir = os.path.expanduser('~/datasets')
data_csv_path_X_train = os.path.join(datasets_dir, 'data_X_train.csv')
data_csv_path_y_train = os.path.join(datasets_dir, 'data_y_train.csv')
X_train = pd.read_csv(data_csv_path_X_train)
y_train = pd.read_csv(data_csv_path_y_train)

# Зададим значения параметров для перебора

parameters = {"n_estimators": [40, 50, 60, 70, 80, 90, 100],
              "max_depth": [1, 2, 3, 4, 5],
              "learning_rate": [0.1, 0.2, 0.3]}

# Инициализация классификатора XGBClassifier

gbdt_clf = XGBClassifier()

# Инициализация перебора параметров

gscv = GridSearchCV(gbdt_clf, parameters, n_jobs=-1, cv=5)

# Запуск перебора параметров

gscv.fit(X_train, y_train)

# Выведем наилучшие значения параметров для текущей модели

print("Наилучшие значения параметров: {}".format(gscv.best_params_),
      "Лучший результат на кросс-валидации: {:.2f}%".format(gscv.best_score_ * 100),
      sep="\n")


# Сохранение обученой модель
model_output_path = os.path.expanduser('~/models/model.pkl')
with open(model_output_path, "wb") as model_file:
    pickle.dump(gscv, model_file)

print("Модель сохранена по пути:", model_output_path)
