# Тестовое задание на стажировку "Гринатом"


![Метрики](https://raw.githubusercontent.com/genalll/test_ml/main/rating/static/3.png)





### Данные: открытый набор данных, который содержит в себе отзывы о фильмах, а также соответствующие им оценки рейтинга.

### Разработать веб-сервис для оценки комментариев, рецензий,  отзывов к фильмам.
#                                                                 Выполнено
####  Произведена предобработка данных. Удалены стоп-слова из отзывов, произведены лемматизаци, токинезация, векторизация учебной и тестовой выборки, стандартизация, отбор признаков.
#### Сформированы датасеты для обучения.
#### На сформированные датасетах обучены и сохранены для последующего создания прототипа приложения: векторизатор текста(CountVectorizer), стандартизатор(StandardScaler), система отбора признаков(SelectKBest).
#### Произведено обучение моделей: GaussianNB,RandomForestClassifier, нейросети tensorflow в режиме многоклассовой классификации и бинарной классификации. Наилучшее результаты - за нейросетью.
#### Создано приложение-прототип на фреймворке django, куда интегрированы ранее обученные модели.
#### Код проекта выложен в публичный репозиторий github
#### Приложение развернуто на VPS по адресу: http://178.21.8.213:8888/      .
#### Подготовлен отчет о работе с оценкой точности полученного результата на тестовой выборке.


### технологический стек: python,pandas,sklearn,nltk,tensorflow.


* Метрики обучения на тестовой выборке

Для оценки позитивная и негативная направленность рецензии.

![Метрики](https://github.com/genalll/test_ml/raw/main/rating/static/2.png)


Для прогнозирования рейтинга фильма на основании рецензии

![Метрики](https://raw.githubusercontent.com/genalll/test_ml/main/rating/static/1.png)

##### Ссылка на презентацию проекта:
##### Ссылка на развернутое приложение django http://178.21.8.213:8888/
##### Блокнот с обучением моделей и подготовкой данных в корневой папке проекта klass.ipynb  https://github.com/genalll/test_ml/blob/main/klass.ipynb


* Для запуска приложения локально в папке выполните команду python3 manage.py runserver Проект будет доступен локально http://127.0.0.1:8000/ .
* В папке files - предобученные модели.