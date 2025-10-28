
import json
import os
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST


CUDA_DEVICE_ID = 0
device = torch.device(f"cuda:{CUDA_DEVICE_ID}") if torch.cuda.is_available() else torch.device("cpu")


train_fmnist_data = FashionMNIST(
    ".", train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_fmnist_data = FashionMNIST(
    ".", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

train_data_loader = torch.utils.data.DataLoader(
    train_fmnist_data, batch_size=32, shuffle=True, num_workers=2
)
test_data_loader = torch.utils.data.DataLoader(
    test_fmnist_data, batch_size=32, shuffle=False, num_workers=2
)

# Проверим, что данные загружены
random_batch = next(iter(train_data_loader))
image, label = random_batch[0][0], random_batch[1][0]
plt.figure()
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Пример изображения: класс {label.item()}")
plt.show()



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Вход: (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)  # → (16, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)  # → (16, 14, 14)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # → (32, 14, 14)
        self.fc1 = nn.Linear(256 * 7 * 7, 128)  # после ещё одного pool → (32, 7, 7)
        self.fc2 = nn.Linear(128, 10)  # 10 классов
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # свёртка + ReLU
        x = self.pool(x)  # пулинг
        x = F.relu(self.conv2(x))  # свёртка + ReLU
        x = self.pool(x)  # пулинг (размер: 7x7)
        x = x.view(-1, 32 * 7 * 7)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # выход: 10 логитов
        return x


# Присваиваем модель переменной model_task_1 — ОБЯЗАТЕЛЬНО!
model_task_1 = SimpleCNN().to(device)

# =============================
# 5. Обучение модели
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_task_1.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model_task_1.train()  # режим обучения
    total_loss = 0
    for batch in train_data_loader:
        images, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = model_task_1(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Эпоха {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_data_loader):.4f}")


def get_accuracy(model, data_loader):
    predicted_labels = []
    real_labels = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            y_predicted = model(batch[0].to(device))
            predicted_labels.append(y_predicted.argmax(dim=1).cpu())
            real_labels.append(batch[1])
    predicted_labels = torch.cat(predicted_labels)
    real_labels = torch.cat(real_labels)
    accuracy_score = (predicted_labels == real_labels).type(torch.FloatTensor).mean()
    return accuracy_score


train_acc = get_accuracy(model_task_1, train_data_loader)
test_acc = get_accuracy(model_task_1, test_data_loader)

print(f"\n✅ Точность на обучающей выборке: {train_acc:.4f}")
print(f"✅ Точность на тестовой выборке: {test_acc:.4f}")


# Проверка наличия файла
assert os.path.exists("hw_overfitting_data_dict.npy"), "Файл hw_overfitting_data_dict.npy не найден!"


def get_predictions(model, eval_data, step=10):
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for idx in range(0, len(eval_data), step):
            y_predicted = model(eval_data[idx: idx + step].to(device))
            predicted_labels.append(y_predicted.argmax(dim=1).cpu())
    predicted_labels = torch.cat(predicted_labels)
    predicted_labels = ",".join([str(x.item()) for x in list(predicted_labels)])
    return predicted_labels


loaded_data_dict = np.load("hw_overfitting_data_dict.npy", allow_pickle=True).item()

submission_dict = {
    "train_predictions_task_1": get_predictions(model_task_1, torch.FloatTensor(loaded_data_dict["train"])),
    "test_predictions_task_1": get_predictions(model_task_1, torch.FloatTensor(loaded_data_dict["test"])),
}

with open("submission_dict_fmnist_task_1.json", "w") as iofile:
    json.dump(submission_dict, iofile)

print("\n✅ Файл для сдачи сохранён: submission_dict_fmnist_task_1.json")