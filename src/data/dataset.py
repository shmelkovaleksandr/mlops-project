import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def load_data(root='./data', split='balanced', batch_size=32):
    """
    Загрузка данных EMNIST
    
    Parameters:
    root (str): Путь для сохранения датасетов
    split (str): Используемый split датасета
    batch_size (int): Размер пакета данных
    
    Returns:
    tuple: (train_dataloader, test_dataloader, train_set, test_set)
    """
    # Загрузка данных
    train_set = datasets.EMNIST(
        root=root, 
        split=split, 
        train=True, 
        download=True, 
        transform=ToTensor()
    )
    
    test_set = datasets.EMNIST(
        root=root, 
        split=split, 
        train=False, 
        download=True, 
        transform=ToTensor()
    )
    
    # Создаем загрузчики данных
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader, train_set, test_set

def get_dataset_info(train_set, test_set):
    """
    Получение информации о датасете
    
    Parameters:
    train_set: Обучающий набор данных
    test_set: Тестовый набор данных
    
    Returns:
    dict: Информация о датасете
    """
    train_samples_len = len(train_set)
    test_samples_len = len(test_set)
    image_shape = train_set[0][0].shape
    classes_len = len(train_set.classes)
    classes_labels = train_set.classes
    
    return {
        'train_samples_len': train_samples_len,
        'test_samples_len': test_samples_len,
        'image_shape': image_shape,
        'classes_len': classes_len,
        'classes_labels': classes_labels,
        'input_size': image_shape[1] * image_shape[2]
    }