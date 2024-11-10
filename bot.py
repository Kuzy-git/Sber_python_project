import os
import asyncio

import numpy as np
from aiogram import Bot, Dispatcher, types, F
from boto3 import client
from dramatiq import actor
from googletrans import Translator
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.utils import load_img, img_to_array

# Настройки Telegram и Yandex Cloud S3
BUCKET_NAME = 'images4'
ENDPOINT_URL = 'https://storage.yandexcloud.net'

# Инициализация бота и клиента Yandex S3
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
s3_client = client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    endpoint_url=ENDPOINT_URL,
    region_name='ru-central1',  # Укажите свой регион
)

# Загрузка предобученной модели для классификации
model = ResNet50(weights='imagenet')

# Инициализация переводчика
translator = Translator()


def classify_image(filename):
    """Классифицирует изображение с использованием ResNet50 и переводит метку на русский."""
    try:
        img = load_img(filename, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=1)[0][0]
        label, confidence = decoded_preds[1], decoded_preds[2]

        # Замена подчеркивания на пробел и перевод метки на русский
        label_for_translation = label.replace("_", " ")
        translated_label = translator.translate(label_for_translation, src='en', dest='ru').text
        return translated_label, confidence
    except Exception as e:
        print(f"Ошибка при классификации изображения: {e}")
        return "Не удалось классифицировать", 0


@actor
def classify_and_respond(chat_id, file_path):
    """Фоновая задача для классификации изображения и отправки ответа в Telegram."""
    print(f"Задача началась: классификация изображения {file_path}")
    
    label, confidence = classify_image(file_path)
    confidence_percent = round(confidence * 100, 2)

    # Отправляем результат пользователю
    print(f"Результат классификации: {label} с уверенностью {confidence_percent}%")
    
    asyncio.run(bot.send_message(
        chat_id=chat_id,
        text=f"На изображении: {label} с уверенностью {confidence_percent}%"
    ))

    os.remove(file_path)


@dp.message(F.text == "/start")
async def start_command(message: types.Message):
    """Отправляет приветственное сообщение при команде /start."""
    await message.answer("Привет! Отправь мне изображение, и я его классифицирую.")


@dp.message(F.photo)
async def handle_image(message: types.Message):
    """Обрабатывает изображение: загружает в Yandex S3 и отправляет на классификацию."""
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    filename = os.path.basename(file_info.file_path)

    # Скачиваем изображение
    file = await bot.download_file(file_info.file_path)
    with open(filename, 'wb') as f:
        f.write(file.read())

    # Загружаем изображение в Yandex S3
    s3_client.upload_file(Filename=filename, Bucket=BUCKET_NAME, Key=filename)

    # Отправляем задачу на классификацию в фоновом режиме
    classify_and_respond.send(message.chat.id, filename)

    await message.reply("Изображение загружено в облако. Начинаю классификацию...")


async def main():
    """Запускает бота."""
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())