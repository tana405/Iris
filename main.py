# текст программы из jupyter


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns


# загрузка ириса
iris = load_iris()
X = iris.data  # массив признаков (4 характеристики цветка)
y = iris.target  # вектор целевых меток (3 вида ирисов)

# разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# создаем DataFrame для удобства
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df['species_name'] = [iris.target_names[i] for i in y]

# размер поля для графика
plt.figure(figsize=(10, 6))

# определение цветов для разных видов ирисов
colors = ['red', 'blue', 'green']

# цикл по всем видам ирисов (3 вида)
for i, species in enumerate(iris.target_names):
    # построение точечного графика для каждого вида:
    plt.scatter(X[y == i, 0], # sepal length - значения по оси х (длина чашелистика)
               X[y == i, 1],  # sepal width - значения по оси y (ширина чашелистика)
               c=colors[i],   # цвет для текущего вида
               label=species, # метка для легенды
               alpha=0.7)     # прозрачность точек (70%)


plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset: Sepal Length vs Sepal Width')
# добавление легенды (справа от графика)
plt.legend()
# включение сетки с прозрачностью 30%
plt.grid(True, alpha=0.3)

plt.show()




# импорт классификатора k-ближайших соседей
from sklearn.neighbors import KNeighborsClassifier

# создание модели
# n_neighbors=3 - количество соседей для принятия решения (гиперпараметр K)
model = KNeighborsClassifier(n_neighbors=3)

# обучение модели на тренировочных данных
# X_train - признаки тренировочной выборки
# y_train - целевые значения (метки классов) тренировочной выборки
model.fit(X_train, y_train)


# оценка модели
y_pred = model.predict(X_test)

# точность (accuracy) модели
# accuracy_score сравнивает предсказанные значения (y_pred) с истинными (y_test) и вычисляет долю правильных предсказаний
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

#рисуем матрицу пирсона (матрицу корреляции)
plt.figure(figsize=(8, 6))

# корреляция пирсона показывает линейную зависимость между признаками (-1 до +1)
correlation_matrix = df[iris.feature_names].corr()

# строим тепловую карту (heatmap) матрицы корреляции с помощью seaborn
sns.heatmap(correlation_matrix,
            annot=True,      # числовые значения коэффициентов корреляции в ячейках
            cmap='coolwarm', # цветовая палитра: синий для отрицательных, красный для положительных корреляций
            center=0,        # центр цветовой шкалы на 0 (нейтральная корреляция)
            square=True)     # делает ячейки квадратными


plt.title('Correlation Matrix of Iris Features')
# оптимизируем расположение элементов графика
plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# создаем матрицу ошибок (confusion matrix)
# confusion_matrix сравнивает реальные значения с предсказанными и создает таблицу, показывающую сколько примеров было правильно классифицировано
cm = confusion_matrix(y_test, y_pred)

# создаем сonfusionMatrixDisplay, принимает саму матрицу (cm) и подписи классов (названия видов ирисов)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

# cmap='Blues' задает синюю цветовую палитру (чем темнее синий, тем выше значение)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')

# отображаем график
plt.show()





