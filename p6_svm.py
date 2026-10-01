import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score

input_file = 'income_data.txt'

# === Загрузка данных ===
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

print("Loaded:", count_class1, "'<=50K',", count_class2, "'>50K'")

X = np.array(X)

# === Кодирование признаков ===
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# === Создание и обучение SVM ===
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=5000))
classifier.fit(X, y)

# === Кросс-валидация через train_test_split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=5000))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# === F-мера ===
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

# === Предсказание для произвольной точки ===
def predict_point(input_data):
    input_data_encoded = [-1] * len(input_data)
    count = 0
    for i, item in enumerate(input_data):
        if item.isdigit():
            input_data_encoded[i] = int(input_data[i])
        else:
            input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
            count += 1
    input_data_encoded = np.array(input_data_encoded).reshape(1, -1)
    predicted_class = classifier.predict(input_data_encoded)
    return label_encoder[-1].inverse_transform(predicted_class)[0]

# --- Точка 1: как в методичке (низкий доход) ---
point1 = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
          'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
          '0', '0', '40', 'United-States']
print("Point 1 →", predict_point(point1))

# --- Точка 2: высокий доход (менеджер, женат, bachelor, 60ч) ---
point2 = ['44', 'Private', '198282', 'Bachelors', '13', 'Married-civ-spouse',
          'Exec-managerial', 'Husband', 'White', 'Male',
          '15024', '0', '60', 'United-States']
print("Point 2 →", predict_point(point2))

# --- Точка 3: молодая, part-time, без высшего ---
point3 = ['20', 'Private', '188300', 'Some-college', '10', 'Never-married',
          'Tech-support', 'Own-child', 'White', 'Female',
          '0', '0', '20', 'United-States']
print("Point 3 →", predict_point(point3))