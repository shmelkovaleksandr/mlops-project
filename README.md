# Распознавание рукописных символов

Этот проект представляет собой полный MLOps конвейер для обучения модели распознавания рукописных символов на основе датасета EMNIST и создания веб-приложения для демонстрации работы модели.

## Установка

# Для установки и запуска проекта выполните следующие команды:

# Клонирование репозитория
```bash
git clone https://github.com/<ваш_юзернейм>/<название_репозитория>.git
cd <название_репозитория>
```

# Установка зависимостей
```bash
pip install -r requirements.txt
```

# Обучение модели
Для обучения модели и экспорта в формат ONNX выполните:
```bash
python src/models/modeling/train.py
```

Скрипт автоматически:
1. Загрузит данные EMNIST
2. Обучит несколько моделей с разными гиперпараметрами
3. Выберет лучшую модель (если указано несколько параметров src/config.py)
4. Экспортирует лучшую модель в формат ONNX

# Запуск веб-приложения
После обучения и экспорта модели вы можете запустить веб-приложение:
```bash
python app/app.py
```
После запуска приложение будет доступно по адресу http://localhost:7860

## Структура проекта

Проект организован согласно Cookiecutter Data Science и имеет следующую структуру:

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

