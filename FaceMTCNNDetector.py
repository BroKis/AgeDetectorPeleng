import cv2
from mtcnn import MTCNN
import os
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(__file__)

base_dir = os.path.dirname(__file__)

def GetImagesFromFile():
    images = []
    ages = []
    for file in os.listdir(base_dir + r'\images'):
        file_name, file_extension = os.path.splitext(file)
        if (file_extension in ['.png', '.jpg']):
            image = cv2.imread(base_dir + r'\images\\' + file)

            name_parts = file.split('_')
            age = int(name_parts[0])

            # Добавляем изображение и информацию в список
            images.append(cv2.resize(image, (64, 64)))
            ages.append(age)

    return np.array(images), np.array(ages)


def ShowImages(images, count, rows, cols):
    # Отображаем первые 10 изображений
    for i in range(count):
        # Создаем подграфик для каждого изображения
        plt.subplot(rows, cols, i + 1)

        # Удаляем оси
        plt.axis('off')

        # Отображаем изображение
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    # Показываем все подграфики
    plt.show()


# Define paths



def FacesDetectorFromImages(images):
    cropped_images = []
    for image in images:
        # Детектирование лиц на изображении
        detector = MTCNN()
        result = detector.detect_faces(image)
        for i in range(len(result)):
            (x, y, w, h) = result[i]['box']
            # Обрезка изображения до области с лицом
            face_image = image[y:y + h, x:x + w]
            if face_image.size != 0:
                cropped_images.append(cv2.resize(face_image, (64, 64)))
                break
    return  cropped_images

