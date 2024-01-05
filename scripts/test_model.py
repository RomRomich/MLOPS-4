# Импортируем нужные библиотеки
import pandas as pd
import os
import pickle
from sklearn.metrics import f1_score, roc_auc_score

# Загрузка данных
datasets_dir = os.path.expanduser('~/datasets')
data_csv_path_X_test = os.path.join(datasets_dir, 'data_X_test.csv')
data_csv_path_y_test = os.path.join(datasets_dir, 'data_y_test.csv')
X_test = pd.read_csv(data_csv_path_X_test)
y_test = pd.read_csv(data_csv_path_y_test)

# Откроем файл `model.pkl`, и загрузим в переменную `gbdt_clf` с помощью модуля `pickle`
f_input = os.path.expanduser('~/models/model.pkl')
with open(f_input, "rb") as fd:
    gbdt_clf = pickle.load(fd)

# Оценка метрик качества классификатора на тестовой части датасета
gbdt_accuracy = gbdt_clf.score(X_test, y_test) * 100
gbdt_f1 = f1_score(y_test, gbdt_clf.predict(X_test))
gbdt_rocauc = roc_auc_score(y_test, gbdt_clf.predict_proba(X_test)[:, 1])

# Выведем полученные метрики качества
print("Test accuracy score: {:.2f}%".format(gbdt_accuracy),
      "Test F1 score: {:.2f}".format(gbdt_f1),
      "Test ROC-AUC score: {:.2f}".format(gbdt_rocauc),
      sep="\n")