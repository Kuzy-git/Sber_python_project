# Sber_python_project
# Telegram Bot для Классификации Изображений с Использованием Yandex Cloud S3 и ResNet50

Этот проект представляет собой Telegram-бот, который позволяет пользователям отправлять изображения для классификации. Бот обрабатывает изображения, загружает их в Yandex Cloud S3, классифицирует с использованием предобученной модели ResNet50 и возвращает пользователю результат на русском языке.

## Стек технологий
- **Python** — основной язык программирования
- **Telegram Bot API** с библиотекой **aiogram** — для создания бота и обработки сообщений
- **Dramatiq** — для обработки задач в фоне
- **Yandex Cloud S3** — хранилище для загруженных изображений
- **Keras / TensorFlow** — для загрузки и применения модели ResNet50
- **Googletrans** — для перевода на русский язык
- **Redis** — брокер сообщений для Dramatiq

## Установка и Запуск

### Предварительные требования
- Python 3.8+
- Установленный и запущенный **Redis** для работы с Dramatiq
- Зарегистрированный бот в Telegram и полученный **API Token**
- **Yandex Cloud** аккаунт и настроенный S3-бакет

### Установка зависимостей
Проект использует **Poetry** для управления зависимостями. Установите Poetry, если он не установлен: