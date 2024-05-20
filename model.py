import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_folder = "D:\\train\\"
X_train = []
y_train = []
labels = 0
emotions = []

for dir in os.listdir(train_folder):
    path = os.path.join(train_folder, dir)
    emotions.append(dir)
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)  
        img = cv2.resize(img, (48, 48))
        image_norm = np.array(img) / 255   
        X_train.append(image_norm)
        y_train.append(labels)
    labels += 1



test_folder="D:\\test\\"
X_test = []
y_test = []
labels = 0
emotions=[]
for dir in os.listdir(train_folder):
    path = train_folder+dir +'/'
    emotions.append(dir)
    for filename in os.listdir(path):
        img = cv2.imread(path + filename)   
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray,(48,48))
        image_norm = np.array(img)/255   
        X_test.append(image_norm)
        y_test.append(labels)
    labels+=1

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_train)
y_test = np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('D:\\Emotion_2.keras', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
callbacks = [checkpoint, reduce_lr]

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность на этапах проверки и обучения')
plt.ylabel('Точность')
plt.xlabel('Эпохи')
plt.legend(['Точность на этапе обучения', 'Точность на этапе проверки'], loc='upper left')
plt.show()
