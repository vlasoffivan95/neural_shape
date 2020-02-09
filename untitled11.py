
!pip install pillow

import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

!wget "https://storage.googleapis.com/kaggle-data-sets/4880/7569/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1581458013&Signature=bouX3NGND6TWLdl8jno1o4OQ7%2BgyCS4MVf703neJTbqe4jdWAg58f7KLHNL06elFd2FrH5HAXZxZlog%2BpbLHzfAqm2PCjskKhY9WofqCpckA0mH9DX8MjbfKF0FQpcmfnL3%2FD9KaFUWAsHTHlQFPay806uIDqfDlMkWwvlw9Vkx1655u6K8Nv2hKC%2FKzmXGWgU%2B%2FlHxvGQr1vkmbZXMlYPom1Ldg0xbOma8w2lhtC2V7ufZ119KEBQQJIwhUjhDwt2dea0PRZRAyo1Ma6s%2BhJWaN%2FocMW%2BDD1pRH3%2Fs4PPYMWRMlsBx%2Fi7aOyfGecuojCpLdWcEzTVo8PlqskAoTrg%3D%3D&response-content-disposition=attachment%3B+filename%3Dfour-shapes.zip" -O data.zip
!unzip data.zip

PATH = "./shapes/"
IMG_SIZE = 64
Shapes = ["circle", "square", "triangle", "star"]
Labels = []
Dataset = []

for shape in Shapes:
    print("Getting data for: ", shape)
    # проходимся по каждому файлу в папке
    for path in os.listdir(PATH + shape):
        # добавляем картинку в список
        image = cv2.imread(PATH + shape + '/' + path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        Dataset.append(image) 
        Labels.append(Shapes.index(shape))

print("\nDataset Images size:", len(Dataset))
print("Image Shape:", Dataset[0].shape)
print("Labels size:", len(Labels))

sns.countplot(x= Labels) # датсет сбалансирован

print("Count of Star images:", Labels.count(Shapes.index("star")))
print("Count of Circles images:", Labels.count(Shapes.index("circle")))
print("Count of Squares images:", Labels.count(Shapes.index("square")))
print("Count of Triangle images:", Labels.count(Shapes.index("triangle")))

# Выводим остальные фигуры датасета
index = np.random.randint(0, len(Dataset) - 1, size= 20)
plt.figure(figsize=(15,10))

for i, ind in enumerate(index, 1):
    img = Dataset[ind]
    lab = Labels[ind]
    lab = Shapes[lab]
    plt.subplot(4, 5, i)
    plt.title(lab)
    plt.axis('off')
    plt.imshow(img)

# Нормализиуем картинки
Dataset = np.array(Dataset)
Dataset = Dataset.astype("float32") / 255.0

# Разбираемся с ярлыками
Labels = np.array(Labels)
Labels = to_categorical(Labels)

# Разделяем на обучающую и тестовую выборки
(trainX, testX, trainY, testY) = train_test_split(Dataset, Labels, test_size=0.2, random_state=42)

print("X Train shape:", trainX.shape)
print("X Test shape:", testX.shape)
print("Y Train shape:", trainY.shape)
print("Y Test shape:", testY.shape)

class LeNet():
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses,  pooling= "max", activation= "relu"):
        # инициализируем метод
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

        # добавляем первый набор слоев: Conv -> Activation -> Pool
        model.add(Conv2D(filters= 6, kernel_size= 5, input_shape= inputShape))
        model.add(Activation(activation))

        if pooling == "max":
            model.add(MaxPooling2D(pool_size= (2, 2), strides= (2, 2)))
        else:
            model.add(AveragePooling2D(pool_size= (2, 2), strides= (2, 2)))

        # добавляем второй набор слоев: Conv -> Activation -> Pool
        model.add(Conv2D(filters= 16, kernel_size= 5))
        model.add(Activation(activation))

        if pooling == "avg":
            model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        else:
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Flatten -> FC 120 -> Dropout -> Activation
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dropout(0.5))
        model.add(Activation(activation))

        # FC 84 -> Dropout -> Activation
        model.add(Dense(84))
        model.add(Dropout(0.5))
        model.add(Activation(activation))

        # FC 4-> Softmax
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        return model

BS = 120
LR = 0.01
EPOCHS = 10
opt = SGD(lr=LR)

# First model with max pooling
model = LeNet.build(3, IMG_SIZE, IMG_SIZE, 4, pooling="max")
model.compile(loss= "categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

# Тренируем модель
H1 = model.fit(trainX, trainY, validation_data= (testX, testY), batch_size= BS,
              epochs= EPOCHS, verbose=1)

# Оцениваем точности на обучающих данных и на тестовых
scores_train = model.evaluate(trainX, trainY, verbose= 1)
scores_test = model.evaluate(testX, testY, verbose= 1)

print("\nModel with Max Pool Accuracy on Train Data: %.2f%%" % (scores_train[1]*100))
print("Model with Max Pool Accuracy on Test Data: %.2f%%" % (scores_test[1]*100))

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

plt.figure(figsize=(15,5))
plt.plot(np.arange(0, EPOCHS), H1.history["acc"], label="Max Pool Train Acc")
plt.plot(np.arange(0, EPOCHS), H1.history["val_acc"], label="Max Pool Test Acc")
plt.title("Comparing Models Train\Test Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="upper left")

# Функция для загрузки файлов
def upload_file():
  from google.colab import files
  uploaded = files.upload()
  for k, v in uploaded.items():
    open(k, 'wb').write(v)
  return list(uploaded.keys())

# словарь соответствий
labels = {0: "Circle", 1: "Square", 2: "Triangle", 3: "0"}

# считываем картику, помещенную в форму, ресайзим ее и даем результат, что там за фигура
imgg = cv2.imread(upload_file()[0])
imgg = cv2.resize(imgg, (IMG_SIZE, IMG_SIZE))
imgg = np.reshape(imgg, [1, IMG_SIZE, IMG_SIZE, 3])
print(f" \n\n\nPredicted figure: {labels[model.predict_classes(imgg)[0]]}")







