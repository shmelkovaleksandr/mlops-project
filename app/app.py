import gradio as gr
import numpy as np
import os
import sys
from pathlib import Path
import logging
from PIL import Image

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EMNIST-APP")

project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Added project root to sys.path: {project_root}")

try:
    import onnxruntime as ort
    print("Successfully imported onnxruntime")
except ImportError as e:
    print(f"ERROR importing onnxruntime: {e}")
    print("Please install onnxruntime: pip install onnxruntime")
    sys.exit(1)

EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]

model_path = os.path.join(project_root, 'models', 'emnist_model.onnx')

def create_dummy_model():
    print(f"Creating dummy model at {model_path}")
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        import torch
        import torch.nn as nn
        
        dummy_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(EMNIST_CLASSES)),
            nn.LogSoftmax(dim=1)
        )
        
        dummy_input = torch.randn(1, 1, 28, 28)
        torch.onnx.export(
            dummy_model,
            dummy_input,
            model_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Dummy model created successfully at {model_path}")
        return True
    except Exception as e:
        print(f"Error creating dummy model: {e}")
        return False

def load_model():
    print(f"Trying to load model from {model_path}")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        if not create_dummy_model():
            print("Failed to create dummy model, exiting")
            return None
    
    try:
        session = ort.InferenceSession(model_path)
        print(f"Model loaded successfully")
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        print(f"Model input names: {input_names}")
        print(f"Model output names: {output_names}")
        return session
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

onnx_session = load_model()
if onnx_session is None:
    print("Failed to load ONNX model, exiting")
    sys.exit(1)

def predict_character(image):
    print("\n\n===== PREDICT CHARACTER CALLED =====")
    print(f"Debug: Тип изображения: {type(image)}")
    
    # Если входные данные — словарь, извлекаем изображение из 'composite'
    if isinstance(image, dict):
        print(f"Debug: Ключи словаря: {image.keys()}")
        if 'composite' in image:
            image = image['composite']  # Извлекаем композитное изображение
            print(f"Debug: Тип извлеченного изображения: {type(image)}")
            print(f"Debug: Форма извлеченного изображения: {image.shape}")
        else:
            return "Ошибка: в словаре нет ключа 'composite'"
    
    # Проверяем, является ли изображение массивом NumPy
    if not isinstance(image, np.ndarray):
        return "Ошибка: изображение не в формате NumPy массива"
    
    try:
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        
        # Преобразование в grayscale
        if len(image.shape) == 3 and image.shape[2] in [3, 4]:  # RGB или RGBA
            print("Converting to grayscale")
            gray_image = np.mean(image[:, :, :3], axis=2).astype(np.float32)
        elif len(image.shape) == 2:  # Уже grayscale
            print("Image already in grayscale")
            gray_image = image.astype(np.float32)
        else:
            return "Ошибка: неожиданная форма изображения"
        
        # Добавляем отладку: минимальное и максимальное значение в grayscale
        print(f"Gray image min: {np.min(gray_image)}, max: {np.max(gray_image)}, mean: {np.mean(gray_image)}")
        
        # Инверсия изображения, если фон светлый
        if np.mean(gray_image) > 128:
            print("Inverting image")
            gray_image = 255 - gray_image
        else:
            print("Not inverting image")
        
        # Добавляем отладку после инверсии
        print(f"Processed image min: {np.min(gray_image)}, max: {np.max(gray_image)}, mean: {np.mean(gray_image)}")
        
        # Масштабирование до 28x28
        pil_image = Image.fromarray(gray_image.astype(np.uint8))
        resized_image = pil_image.resize((28, 28), Image.LANCZOS)
        resized_array = np.array(resized_image).astype(np.float32)
        
        # Добавляем отладку для масштабированного изображения
        print(f"Resized image min: {np.min(resized_array)}, max: {np.max(resized_array)}, mean: {np.mean(resized_array)}")
        
        # Нормализация
        normalized_image = resized_array / 255.0
        
        # Подготовка для модели
        model_input = np.expand_dims(np.expand_dims(normalized_image, axis=0), axis=0)
        outputs = onnx_session.run(None, {'input': model_input})
        probabilities = np.exp(outputs[0])
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = float(probabilities[0, predicted_class])
        character = EMNIST_CLASSES[predicted_class]
        
        return f"Символ: {character}, Уверенность: {round(confidence*100)} %"
    
    except Exception as e:
        return f"Ошибка распознавания: {str(e)}"

def predict_sketch(sketch_data):
    return predict_character(sketch_data)

def predict_image(image_data):
    return predict_character(image_data)

def create_app():
    with gr.Blocks(title="Распознавание рукописных символов") as demo:
        gr.Markdown("# Распознавание рукописных символов")
        
        with gr.Row():
            #with gr.Column():
                #ketch = gr.Sketchpad(label="Нарисовать символ")
                #predict_sketch_btn = gr.Button("Распознать (холст)")
                
            with gr.Column():
                image_input = gr.Image(image_mode="L", label="Загрузить изображение")
                predict_image_btn = gr.Button("Распознать (файл)")
                
        result = gr.Textbox(label="Результат распознавания")
        clear_btn = gr.Button("Очистить все поля")
        
        #predict_sketch_btn.click(fn=predict_sketch, inputs=sketch, outputs=result)
        predict_image_btn.click(fn=predict_image, inputs=image_input, outputs=result)
        
        clear_btn.click(fn=lambda: None, inputs=None, outputs=[image_input]) # [sketch, image_input]
        
        gr.Markdown("### Информация о модели")
        gr.Markdown(f"Путь к модели: {model_path}")
        
    return demo

if __name__ == "__main__":
    print("Starting Gradio app")
    app = create_app()
    app.launch(share=False)
    print("Gradio app closed")
