import torch
import numpy as np

def xavier_normal(F_in, F_out):
    """
    Инициализация весов методом Ксавье
    
    Parameters:
    F_in (int): размер входного слоя
    F_out (int): размер выходного слоя
    
    Returns:
    torch.Tensor: тензор весов, инициализированный методом Ксавье
    """
    limit = np.sqrt(6 / float(F_in + F_out))
    W = np.random.uniform(low=-limit, high=limit, size=(F_out, F_in))  
    return torch.from_numpy(W).type(torch.float32).requires_grad_()

def initialize_parameters(input_size, hidden_size1, hidden_size2, output_size):
    """
    Инициализация параметров модели (weights и biases)
    
    Parameters:
    input_size (int): Размер входного слоя
    hidden_size1 (int): Размер первого скрытого слоя
    hidden_size2 (int): Размер второго скрытого слоя
    output_size (int): Размер выходного слоя
    
    Returns:
    list: Список параметров модели [W1, b1, W2, b2, W3, b3]
    """
    W1 = xavier_normal(input_size, hidden_size1)
    b1 = torch.randn(hidden_size1, requires_grad=True)
    W2 = xavier_normal(hidden_size1, hidden_size2)
    b2 = torch.randn(hidden_size2, requires_grad=True)
    W3 = xavier_normal(hidden_size2, output_size)
    b3 = torch.randn(output_size, requires_grad=True)
    
    return [W1, b1, W2, b2, W3, b3]

def update_parameters(parameters, learning_rate):
    """
    Обновление параметров модели
    
    Parameters:
    parameters: список параметров модели
    learning_rate: скорость обучения
    """
    with torch.no_grad():
        for param in parameters:
            param -= learning_rate * param.grad
            param.grad.zero_()