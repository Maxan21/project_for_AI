
import telebot
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np


k=0
token=''
bot=telebot.TeleBot(token)

haar = cv2.CascadeClassifier(r'D:\Python\Python312\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Отправьте фото.")
@bot.message_handler(content_types=['photo'])
def photo(message):  
    global k  # Убедитесь, что переменная k инициализирована где-то в коде
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    # Сохраняем изображение локально
    image_path = f"image_pos_{k}.jpg"
    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    
    # Читаем изображение и преобразуем в градации серого
    img = cv2.imread(image_path)
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
           # определяем координаты лица
    face_coordinates = haar.detectMultiScale(grayscaled_img)
    
    for (x,y,w,h) in face_coordinates:
        # изменяем размер изображения и нормируем значения
        roi_gray = grayscaled_img[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
                directory='happy'
                image_path_1=f'D:/train/{directory}'

                cv2.imwrite(f"{image_path_1}/image_pos_{k}.jpg",roi_gray)
    k += 1
bot.polling(none_stop=True, interval=0)

