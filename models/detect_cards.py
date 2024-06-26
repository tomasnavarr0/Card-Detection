import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np


def get_color_for_class(class_name):
    if 'O' in class_name:
        return (255, 255, 0)  # Amarillo
    elif 'E' in class_name:
        return (0, 0, 255)  # Azul
    elif 'B' in class_name:
        return (0, 255, 0)  # Verde
    elif 'C' in class_name:
        return (255, 0, 0)  # Rojo
    elif 'J' in class_name:
        return (255, 0, 255)  # Violeta
    else:
        return (255, 255, 255)  # Blanco por defecto


class DetectCards:
    model_path: str = "models\yolov8_100epochs.pt"
    model = YOLO(model_path)

    @staticmethod
    def detect_and_show_cards_in_image(
            image_path: str, model_path: str = model_path, class_names: dict[int,str] = model.names, confidence_threshold: float = 0.35
            ) -> list[str]:
        cards_found = []
        # Cargar el modelo YOLOv8
        model = YOLO(model_path)
        
        # Cargar la imagen
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Realizar la detección
        results = model.predict(image_rgb, conf=confidence_threshold)
        
        # Dibujar los bounding boxes en la imagen
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, class_id = result
            if conf >= confidence_threshold:  # Filtrar por umbral de confianza
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_id = int(class_id)
                class_name = class_names[class_id]
                label = f'{class_name} {conf:.2f}'
                cards_found.append(class_name)
                color = get_color_for_class(class_name)
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Mostrar la imagen con los bounding boxes
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

        return cards_found


