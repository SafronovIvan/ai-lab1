import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

# --- Загрузка датасета Boston через fetch_openml ---
# (load_boston удалён из sklearn начиная с версии 1.2,
#  но сам датасет доступен через OpenML)
boston = datasets.fetch_openml(name='boston', version=1, as_frame=False)
X, y = boston.data, boston.target

print("Dataset: Boston Housing")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

# Перемешивание
X, y = shuffle(X, y, random_state=7)

# Разбивка 80/20
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# === Создание и обучение SVM-регрессора (как в методичке) ===
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)
sv_regressor.fit(X_train, y_train)

# === Оценка ===
y_test_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)

print("\n###### Performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# === Тестовая точка из методички (13 признаков Boston) ===
test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052,
             26, 666, 20.2, 351.34, 15.27]
print("\nPredicted price:", sv_regressor.predict([test_data])[0])

# === Две дополнительные точки (методичка требует не менее трёх) ===
test_data_2 = [11.9, 0, 18.1, 0, 0.74, 7.0, 90, 1.97,
               24, 666, 20.2, 396.9, 34.41]
test_data_3 = [0.05, 60, 2.32, 0, 0.53, 6.6, 62, 6.04,
               21, 226, 17.8, 393.9, 7.62]

print("Predicted price 2:", sv_regressor.predict([test_data_2])[0])
print("Predicted price 3:", sv_regressor.predict([test_data_3])[0])

# === Эксперименты с C и epsilon (требование методички) ===
print("\n--- Разные параметры C и epsilon ---")
for C in [0.5, 1.0, 5.0]:
    for eps in [0.05, 0.1, 0.5]:
        m = SVR(kernel='linear', C=C, epsilon=eps)
        m.fit(X_train, y_train)
        yp = m.predict(X_test)
        mse_ = mean_squared_error(y_test, yp)
        evs_ = explained_variance_score(y_test, yp)
        print(f"C={C}, epsilon={eps} → MSE={round(mse_, 2)}, EVS={round(evs_, 2)}")