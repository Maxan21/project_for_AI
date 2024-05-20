import telebot
import cv2
import tensorflow as tf
import time
token=''
bot=telebot.TeleBot(token)
model=tf.keras.models.load_model("D:\\Emotion_2.keras")
class_emotion=['happy', 'negative', 'neutral', 'surprise']
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Отправьте фото.")
@bot.message_handler(content_types=['photo'])
def photo(message):  
    global model, class_emotion, haar
    fileID = message.photo[-1].file_id   
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)
    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)
    image_path = "image.jpg"
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resized_image = cv2.resize(gray_image, (48, 48), interpolation=cv2.INTER_AREA)

    roi = resized_image.reshape(1, 48, 48, 1).astype('float') / 255.0
    # im = Image.fromarray(resized_image) было создано, чтобы проверить, как работает с фото
    # im.show()
    pred=model.predict(roi)[0]
    label=class_emotion[pred.argmax()]
    bot.send_message(message.chat.id, label)
if __name__ == '__main__':
    while True:
        try: #добавляем try для бесперебойной работы
            bot.polling(none_stop=True) #запуск бота
        except:
            time.sleep(10) #в случае падения