import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

DATA_PATH = "./custom_dataset"
BATCH_SIZE = 10
EPOCHS = 10


def load_data():
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'train'),
        transform=data_transforms
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'test'),
        transform=data_transforms
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Найдены классы: {class_names}")
    print(f"Количество классов: {num_classes}")
    print(f"Размер обучающей выборки: {len(train_dataset)}")
    print(f"Размер тестовой выборки: {len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader, class_names, num_classes


def create_model(num_classes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    net = torchvision.models.alexnet(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False

    new_classifier = net.classifier[:-1]
    new_classifier.add_module('6', nn.Linear(4096, num_classes))
    net.classifier = new_classifier
    net = net.to(device)

    return net, device

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def train_model(model, train_loader, test_loader, device, epochs=10):
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.01)

    train_losses = []
    test_accuracies = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = lossFn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if i % 10 == 0:
                print(f'Эпоха {epoch + 1}/{epochs}, Батч {i}, Ошибка: {loss.item():.4f}')

        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)

        accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(accuracy)

        print(
            f'Эпоха {epoch + 1}/{epochs} завершена. Средняя ошибка: {avg_loss:.4f}, Точность на тесте: {accuracy:.2f}%')
        print('-' * 50)

    training_time = time.time() - start_time
    print(f'Обучение завершено за {training_time:.2f} секунд')

    return train_losses, test_accuracies


def visualize_predictions(model, test_loader, class_names, device):
    images_per_class = 3
    test_inputs_by_class = {i: [] for i in range(len(class_names))}
    test_classes_by_class = {i: [] for i in range(len(class_names))}

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            for i in range(len(labels)):
                class_idx = labels[i].item()
                if len(test_inputs_by_class[class_idx]) < images_per_class:
                    test_inputs_by_class[class_idx].append(images[i])
                    test_classes_by_class[class_idx].append(labels[i])

            if all(len(test_inputs_by_class[i]) >= images_per_class for i in range(len(class_names))):
                break

    test_inputs = []
    test_classes = []

    for class_idx in range(len(class_names)):
        for j in range(min(images_per_class, len(test_inputs_by_class[class_idx]))):
            test_inputs.append(test_inputs_by_class[class_idx][j])
            test_classes.append(test_classes_by_class[class_idx][j])

    test_inputs = torch.stack(test_inputs)
    test_classes = torch.tensor(test_classes)

    with torch.no_grad():
        test_inputs_gpu = test_inputs.to(device)
        predictions = model(test_inputs_gpu)
        _, predicted_classes = torch.max(predictions.data, 1)
        probabilities = torch.nn.functional.softmax(predictions, dim=1)

    num_images = len(test_inputs)
    ncols = 3
    nrows = (num_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx in range(num_images):
        img = test_inputs[idx].numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)

        true_label = class_names[test_classes[idx]]
        pred_label = class_names[predicted_classes[idx]]
        confidence = probabilities[idx][predicted_classes[idx]].item()

        axes[idx].imshow(img)

        color = 'green' if true_label == pred_label else 'red'
        axes[idx].set_title(f'Истина: {true_label}\nПредсказание: {pred_label}\nУверенность: {confidence:.2f}',
                            color=color, fontsize=9, pad=5)
        axes[idx].axis('off')

    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Примеры предсказаний модели (по 3 изображения от каждого класса)', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()


def plot_training_results(train_losses, test_accuracies, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, epochs + 1), train_losses, 'b-', marker='o')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Ошибка')
    ax1.set_title('Функция потерь во время обучения')
    ax1.grid(True)

    ax2.plot(range(1, epochs + 1), test_accuracies, 'r-', marker='o')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность (%)')
    ax2.set_title('Точность на тестовых данных')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()


def save_model(model, path='alexnet_custom_classes.pth'):
    torch.save(model.state_dict(), path)
    print(f"\nМодель сохранена в файл: {path}")

if __name__ == '__main__':
    print("=" * 60)
    print("ОБУЧЕНИЕ AlexNet С ТРАНСФЕРНЫМ ОБУЧЕНИЕМ")
    print("=" * 60)

    train_loader, test_loader, class_names, num_classes = load_data()
    model, device = create_model(num_classes)

    initial_accuracy = evaluate_model(model, test_loader, device)
    print(f'Точность модели до обучения: {initial_accuracy:.2f}%')

    print(f"\nНачало обучения на {EPOCHS} эпох")
    train_losses, test_accuracies = train_model(model, train_loader, test_loader, device, EPOCHS)

    plot_training_results(train_losses, test_accuracies, EPOCHS)

    final_accuracy = evaluate_model(model, test_loader, device)
    print(f'Финальная точность модели: {final_accuracy:.2f}%')
    print(f'Улучшение: {final_accuracy - initial_accuracy:.2f}%')

    visualize_predictions(model, test_loader, class_names, device)
    save_model(model)