import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from AgePredictor import AgePredictor, AgeLoss
from FaceMTCNNDetector import GetImagesFromFile, ShowImages, FacesDetectorFromImages


model_weights_path = 'model_weights.pth'
batch_size = 64
age_diff = 3
epochs = 1000
losses = []

images, ages = GetImagesFromFile()
ShowImages(images, 100, 10, 10)

cropped_images = np.array(FacesDetectorFromImages(images))
ShowImages(cropped_images, 30, 3, 10)


train_images, test_images, train_ages, test_ages = train_test_split(cropped_images, ages[0:len(cropped_images)],
                                                                    test_size=0.1, train_size=0.9, random_state=42)

# Компиляция модели
model = AgePredictor()
criterion = AgeLoss()
optimizer = torch.optim.Adam(model.parameters())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor_images = torch.Tensor(train_images)
tensor_ages = torch.Tensor(train_ages)
train_dataset = TensorDataset(tensor_images, tensor_ages)
train_dataloader = DataLoader(train_dataset)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

print(len(train_loader))


def train(epochs, model, criterion, optimizer, train_loader, eps):
    for epoch in range(epochs):
        loss = 0
        for batch_features, targets in train_loader:
            if (len(batch_features) != batch_size):
                continue
            batch_features = batch_features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)
            # compute training reconstruction loss
            train_loss = criterion(outputs, targets)
            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)
        losses.append(loss)
        # display the epoch training loss
        # if((epoch+1) % 10 == 0):
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        if loss < eps:
            break

    return model


model = train(epochs, model, criterion, optimizer, train_loader, age_diff)

# предположим, что `model` - это ваша обученная модель
torch.save(model.state_dict(), model_weights_path)

test_dataset = TensorDataset(torch.Tensor(test_images), torch.Tensor(test_ages))
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

def accuracy(predicted, actual):
    acc = 0
    for pred, act in zip(predicted, actual):
        if(np.abs(act - pred) <= age_diff):
            acc += 1
    return acc / len(predicted)

# Переводим модель в режим оценки
model.eval()

# Инициализируем переменную для хранения суммы квадратов ошибок
sum_of_squared_errors = 0.0

pred_ages = []

# Не вычисляем градиенты, так как мы не обучаем модель
i = 0
with torch.no_grad():
    right_predicted_ages = 0
    all_ages = 0
    for eval_images, true_ages in test_loader:
        if len(eval_images) != batch_size:
            continue
        # Перемещаем данные на тот же device, на котором находится модель
        eval_images = eval_images.to(device)
        true_ages = true_ages.to(device)
        # Получаем предсказания модели
        predicted_ages = model(eval_images)
        pred_ages.extend(predicted_ages.numpy())

print(accuracy(pred_ages, test_ages))
