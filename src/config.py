"""
Конфигурации для проекта
"""

# Базовая конфигурация
BASE_CONFIG = {
    'root': './data',
    'split': 'balanced',
    'batch_size': 32,
    'learning_rate': 1e-3,
    'hidden_size1': 256,
    'hidden_size2': 128,
    'epochs': 10
}

# Конфигурации для экспериментов
HYPERPARAMETERS = [
    {"batch_size": 32, "learning_rate": 1e-3, "hidden_size1": 256, "hidden_size2": 128, "epochs": 4}
]

# Конфигурация для ONNX экспорта
ONNX_CONFIG = {
    'filename': 'models/emnist_model.onnx',
    'input_names': ['input'],
    'output_names': ['output'],
    'dynamic_axes': {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
}