import matplotlib
matplotlib.use('MacOSX')   # нативный backend macOS, без tkinter

import numpy as np
import matplotlib.pyplot as plt


def visualize_classifier(classifier, X, y):
    # Границы графика
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Шаг сетки
    mesh_step_size = 0.01

    # Сетка значений X и Y
    x_vals, y_vals = np.meshgrid(
        np.arange(min_x, max_x, mesh_step_size),
        np.arange(min_y, max_y, mesh_step_size)
    )

    # Прогон классификатора по всей сетке
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    # Рисуем
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, shading='auto')

    # Точки обучающей выборки
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black',
                linewidth=1, cmap=plt.cm.Paired)

    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    plt.xticks(np.arange(int(X[:, 0].min() - 1),
                         int(X[:, 0].max() + 1), 1.0))
    plt.yticks(np.arange(int(X[:, 1].min() - 1),
                         int(X[:, 1].max() + 1), 1.0))

    plt.show()