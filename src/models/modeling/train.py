import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import os
import sys
from pathlib import Path

# Добавляем путь к корню проекта
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import load_data, get_dataset_info
from src.config import BASE_CONFIG, HYPERPARAMETERS, ONNX_CONFIG


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        """
        Инициализация модели многослойного перцептрона
        
        Parameters:
        input_size (int): Размер входного слоя
        hidden_size1 (int): Размер первого скрытого слоя
        hidden_size2 (int): Размер второго скрытого слоя
        output_size (int): Размер выходного слоя
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        # Определение слоев как атрибуты модуля
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        """
        Прямой проход через сеть
        
        Parameters:
        x (torch.Tensor): Входной тензор
        
        Returns:
        torch.Tensor: Выходной тензор
        """
        # Преобразование входных данных в плоский вектор
        x = x.view(-1, self.input_size)
        # Первый скрытый слой
        x = F.relu(self.fc1(x))
        # Второй скрытый слой
        x = F.relu(self.fc2(x))
        # Выходной слой
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
    
    def train_model(self, train_dataloader, test_dataloader, learning_rate, epochs):
        """
        Обучение модели
        
        Parameters:
        train_dataloader: DataLoader для обучающей выборки
        test_dataloader: DataLoader для тестовой выборки
        learning_rate (float): Скорость обучения
        epochs (int): Количество эпох
        
        Returns:
        float: Лучшая достигнутая точность
        """
        # Определяем оптимизатор
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # Определяем функцию потерь
        criterion = nn.NLLLoss()
        
        best_accuracy = 0
        
        # Основной цикл обучения
        for epoch in range(epochs):
            # Метрики для обучающей выборки
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Устанавливаем модель в режим обучения
            self.train()
            
            # Проход по батчам обучающей выборки
            for batch_idx, (X, y) in enumerate(train_dataloader):
                # Обнуляем градиенты
                optimizer.zero_grad()
                
                # Прямой проход
                outputs = self(X)
                # Вычисление функции потерь
                loss = criterion(outputs, y)
                # Обратное распространение ошибки
                loss.backward()
                # Обновление параметров
                optimizer.step()
                
                # Накопление статистики
                running_loss += loss.item()
                correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
                total += y.size(0)
                
                # Вывод статистики каждые 100 батчей
                if batch_idx % 100 == 99:
                    print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
            
            # Вычисление точности на обучающей выборке
            train_accuracy = 100 * correct / total
            print(f'Эпоха {epoch + 1} Точность на обучающей выборке: {train_accuracy:.2f}%')
            
            # Оценка на тестовой выборке
            test_accuracy = self.evaluate(test_dataloader)
            print(f'Эпоха {epoch + 1} Точность на тестовой выборке: {test_accuracy:.2f}% \n')
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
        
        return best_accuracy
    
    def evaluate(self, dataloader):
        """
        Оценка модели на наборе данных
        
        Parameters:
        dataloader: DataLoader для оценки
        
        Returns:
        float: Точность на наборе данных (%)
        """
        # Переключение в режим оценки
        self.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in dataloader:
                outputs = self(X)
                correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
                total += y.size(0)
        
        return 100 * correct / total
    
    def export_to_onnx(self, filename, input_shape=(1, 1, 28, 28)):
        """
        Экспорт модели в формат ONNX
        
        Parameters:
        filename (str): Путь для сохранения файла ONNX
        input_shape (tuple): Форма входного тензора
        """
        # Устанавливаем модель в режим оценки
        self.eval()
        
        # Создаем пример входных данных
        dummy_input = torch.randn(input_shape)
        
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Экспорт модели
        torch.onnx.export(
            self,  # модель
            dummy_input,  # входной тензор
            filename,  # путь для сохранения
            export_params=True,  # сохранить параметры модели
            opset_version=12,  # версия ONNX
            do_constant_folding=True,  # оптимизация констант
            input_names=ONNX_CONFIG['input_names'],  
            output_names=ONNX_CONFIG['output_names'],
            dynamic_axes=ONNX_CONFIG['dynamic_axes']
        )
        print(f"Модель успешно экспортирована в {filename}")


def run_experiment(params):
    """
    Запуск эксперимента с заданными параметрами
    
    Parameters:
    params (dict): Параметры эксперимента
    
    Returns:
    tuple: (MLP model, accuracy)
    """
    # Загрузка данных
    train_dataloader, test_dataloader, train_set, test_set = load_data(
        root=BASE_CONFIG['root'],
        split=BASE_CONFIG['split'],
        batch_size=params['batch_size']
    )
    
    # Получение информации о датасете
    dataset_info = get_dataset_info(train_set, test_set)
    
    # Создание и обучение модели
    model = MLP(
        input_size=dataset_info['input_size'],
        hidden_size1=params['hidden_size1'],
        hidden_size2=params['hidden_size2'],
        output_size=dataset_info['classes_len']
    )
    
    accuracy = model.train_model(
        train_dataloader,
        test_dataloader,
        params['learning_rate'],
        params['epochs']
    )
    
    return model, accuracy


def find_best_model():
    """
    Поиск лучшей модели среди всех экспериментов
    
    Returns:
    tuple: (best_model, best_params, best_accuracy)
    """
    results = []
    models = []
    
    for params in HYPERPARAMETERS:
        print(f"\nТестирование параметров: {params}")
        model, accuracy = run_experiment(params)
        results.append((params, accuracy))
        models.append(model)
        print(f"Точность: {accuracy:.2f}%")
    
    # Вывод результатов всех экспериментов
    print("\nРезультаты всех экспериментов:")
    for idx, (params, accuracy) in enumerate(results):
        print(f"Параметры: {params}")
        print(f"Точность: {accuracy:.2f}%")
        print("-" * 50)
    
    # Поиск лучшей модели
    best_idx = max(range(len(results)), key=lambda i: results[i][1])
    best_params, best_accuracy = results[best_idx]
    best_model = models[best_idx]
    
    print("\nЛучшие параметры:")
    print(f"Параметры: {best_params}")
    print(f"Точность: {best_accuracy:.2f}%")
    
    return best_model, best_params, best_accuracy


def main():
    """
    Основная функция для запуска обучения
    """
    print("Начало обучения модели...")
    best_model, best_params, best_accuracy = find_best_model()
    
    # Экспорт лучшей модели в ONNX
    best_model.export_to_onnx(ONNX_CONFIG['filename'])
    
    return best_model, best_params, best_accuracy


if __name__ == "__main__":
    main()