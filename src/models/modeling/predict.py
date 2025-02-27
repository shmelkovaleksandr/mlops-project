import torch
import numpy as np
import onnxruntime as ort
import sys
import logging
from pathlib import Path

# Настраиваем логирование
logger = logging.getLogger(__name__)

# Добавляем путь к корню проекта
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import ONNX_CONFIG


def predict_with_torch_model(model, input_data):
    """
    Предсказание с использованием PyTorch модели
    
    Parameters:
    model: Модель PyTorch
    input_data (torch.Tensor): Входные данные
    
    Returns:
    tuple: (predicted_class, confidence)
    """
    # Переключаем модель в режим оценки
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_data)
        probabilities = torch.exp(outputs)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return predicted_class, confidence


def load_onnx_model(model_path):
    """
    Загрузка ONNX модели
    
    Parameters:
    model_path (str): Путь к файлу модели ONNX
    
    Returns:
    onnxruntime.InferenceSession: Сессия для инференса
    """
    logger.info(f"Loading ONNX model from {model_path}")
    try:
        # Проверяем, существует ли файл
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        session = ort.InferenceSession(model_path)
        
        # Выводим информацию о входах и выходах модели
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        logger.info(f"Model loaded. Input names: {input_names}, Output names: {output_names}")
        
        return session
    except Exception as e:
        logger.error(f"Error loading ONNX model: {str(e)}", exc_info=True)
        raise


def predict_with_onnx(session, input_data, input_name='input'):
    """
    Предсказание с использованием ONNX модели
    
    Parameters:
    session: Сессия ONNX
    input_data (numpy.ndarray): Входные данные
    input_name (str): Имя входного слоя
    
    Returns:
    tuple: (predicted_class, confidence)
    """
    try:
        # Преобразование входных данных в нужный формат
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
        
        logger.debug(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")
        
        # Удостоверимся, что имена входов правильные
        actual_input_name = session.get_inputs()[0].name
        if input_name != actual_input_name:
            logger.warning(f"Provided input name '{input_name}' doesn't match model's input name '{actual_input_name}'. Using '{actual_input_name}'.")
            input_name = actual_input_name
        
        # Получение предсказания
        logger.debug(f"Running ONNX inference with input name: {input_name}")
        outputs = session.run(None, {input_name: input_data})
        
        log_probs = outputs[0]
        logger.debug(f"Output log_probs shape: {log_probs.shape}")
        
        # Преобразование log_softmax в вероятности
        probabilities = np.exp(log_probs)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = float(probabilities[0, predicted_class])
        
        logger.debug(f"Predicted class: {predicted_class}, confidence: {confidence:.4f}")
        
        return predicted_class, confidence
    
    except Exception as e:
        logger.error(f"Error in ONNX prediction: {str(e)}", exc_info=True)
        raise


def preprocess_image(image, target_size=(28, 28)):
    """
    Предобработка изображения для подачи в модель
    
    Parameters:
    image: Исходное изображение
    target_size (tuple): Целевой размер (высота, ширина)
    
    Returns:
    numpy.ndarray: Предобработанное изображение
    """
    try:
        logger.debug(f"Original image shape: {image.shape}")
        
        # Преобразование в numpy, если это не numpy array
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Преобразование к нужному размеру, если необходимо
        if image.shape[-2:] != target_size:
            logger.debug(f"Resizing image from {image.shape[-2:]} to {target_size}")
            # Используем PIL для изменения размера
            from PIL import Image
            if len(image.shape) == 3 and image.shape[0] == 1:  # (1, H, W)
                img = Image.fromarray(image[0].astype(np.uint8))
            elif len(image.shape) == 2:  # (H, W)
                img = Image.fromarray(image.astype(np.uint8))
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
                
            img = img.resize(target_size, Image.LANCZOS)
            image = np.array(img)
        
        # Нормализация значений пикселей в диапазон [0, 1]
        if image.max() > 1.0:
            logger.debug(f"Normalizing image with max value: {image.max()}")
            image = image / 255.0
        
        # Добавление размерности батча, если её нет
        if len(image.shape) == 2:
            logger.debug("Adding batch dimension")
            image = np.expand_dims(image, axis=0)
        if len(image.shape) == 3 and image.shape[0] != 1:
            image = np.expand_dims(image, axis=0)
        
        logger.debug(f"Preprocessed image shape: {image.shape}")
        return image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}", exc_info=True)
        raise