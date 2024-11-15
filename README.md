# Sber_python_project
# Telegram Bot для Классификации Изображений с Использованием Yandex Cloud S3 и ResNet50

Этот проект представляет собой Telegram-бот, который позволяет пользователям отправлять изображения для классификации. Бот обрабатывает изображения, загружает их в Yandex Cloud S3, классифицирует с использованием предобученной модели ResNet50 и возвращает пользователю результат на русском языке.

## Стек технологий
- **Python** — основной язык программирования
- **Telegram Bot API** с библиотекой **aiogram** — для создания бота и обработки сообщений
- **Dramatiq** — для обработки фоновых задач
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

### Проект использует **Poetry** для управления зависимостями. Установите Poetry, если он не установлен.

### 1. Клонирование репозитория 

```git clone https://github.com/Kuzy-git/Sber_python_project.git```

### 2. Переход в директорию Sber_python_project 

```cd Sber_python_project```

### 3. Создание виртуального окружения

```poetry install```

### 4. Активация виртуального окружения

```Poetry shell```

### 5. Установка и запуск Redis
#### Для работы с фоновыми задачами через dramatiq нужен брокер сообщений. В этом проекте используется Redis. Установите и запустите Redis. Я устанавливала через Ubuntu:

```sudo apt-get install redis-server```

```redis-server```

### 6. Настройка переменных окружения
#### Создайте файл .env в корне проекта и добавьте следующие переменные:

```API_TOKEN=<Ваш_Token_для_Telegram_бота>```

```AWS_ACCESS_KEY_ID=<Ваш_ключ_доступа_AWS>```

```AWS_SECRET_ACCESS_KEY=<Ваш_секретный_ключ_AWS>```

### 7. Запуск проекта
#### После того как все зависимости установлены и переменные окружения настроены, запустите проект:

```dramatiq bot```

```python bot.py```

### 8. Использование
#### Откройте Telegram и найдите вашего бота.
#### Напишите команду /start, чтобы начать взаимодействие с ботом.
#### Отправьте изображение, и бот классифицирует его с использованием модели ResNet50, а затем отправит результат обратно.
