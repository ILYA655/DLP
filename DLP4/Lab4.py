import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Cоздадим простую нейронную сеть
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, out_size),
                                    nn.Sigmoid()
                                    )

    def forward(self, X):
        pred = self.layers(X)
        return pred

# Загружаем данные
df = pd.read_csv('dataset_simple.csv')

# Выделяем признаки и целевую переменную
X = torch.Tensor(df[['age', 'income']].values)
y = df['will_buy'].values
y = torch.Tensor(y.reshape(-1, 1))

# НОРМАЛИЗУЕМ ДАННЫЕ - это важно!
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X = (X - X_mean) / X_std

# Параметры сети
inputSize = X.shape[1]  # количество признаков задачи (2)
hiddenSizes = 8         # увеличиваем нейроны
outputSize = 1          # один выходной нейрон

# Создаем экземпляр нашей сети
net = NNet(inputSize, hiddenSizes, outputSize)

# Посчитаем ошибку нашего не обученного алгоритма
with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >= 0.5, 1, 0).reshape(-1, 1))
err = sum(abs(y - pred)) / 2
print("Ошибка до обучения:", err.item())

# Для обучения нам понадобится выбрать функцию вычисления ошибки
lossFn = nn.BCELoss()

# и алгоритм оптимизации весов (используем Adam вместо SGD)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# В цикле обучения "прогоняем" обучающую выборку
epochs = 500  # увеличиваем эпохи
for i in range(0, epochs):
    pred = net.forward(X)
    loss = lossFn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 50 == 0:
        print('Ошибка на ' + str(i + 1) + ' итерации: ', loss.item())

# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >= 0.5, 1, 0).reshape(-1, 1))
err = sum(abs(y - pred)) / 2
print('\nОшибка после обучения (количество несовпавших ответов):')
print(err.item())

# Пример предсказания для тестовых случаев
print("\nПримеры предсказаний:")
test_cases = [
    [25, 30000],
    [35, 40000],
    [45, 60000],
    [30, 35000],
    [55, 80000],
]

with torch.no_grad():
    test_tensor = torch.Tensor(test_cases)
    # Нормализуем тестовые данные
    test_tensor_normalized = (test_tensor - X_mean) / X_std
    predictions = net.forward(test_tensor_normalized)
    pred_classes = torch.where(predictions >= 0.5, 1, 0)

print("\nВозраст | Доход | Предсказание | Вероятность")
print("-" * 50)
for i, (age, income) in enumerate(test_cases):
    pred_text = "купит" if pred_classes[i].item() == 1 else "не купит"
    prob = predictions[i].item() * 100
    print(f"{age:6.0f} | {income:6.0f} | {pred_text:10} | {prob:5.1f}%")