import cv2
from ultralytics import YOLO
import numpy as np
import time
import pygame

# Inicializar pygame para reproducir el sonido
pygame.mixer.init()

# Cargar sonido de "envido"
pygame.mixer.music.load('envido.mp3')

# Define las clases y los colores correspondientes
class_names = {
    0: '1O', 1: '1C', 2: '1E', 3: '1B', 4: '2O', 5: '2C', 6: '2E', 7: '2B', 
    8: '3O', 9: '3C', 10: '3E', 11: '3B', 12: '4O', 13: '4C', 14: '4E', 15: '4B', 
    16: '5O', 17: '5C', 18: '5E', 19: '5B', 20: '6O', 21: '6C', 22: '6E', 23: '6B', 
    24: '7O', 25: '7C', 26: '7E', 27: '7B', 28: '8O', 29: '8C', 30: '8E', 31: '8B', 
    32: '9O', 33: '9C', 34: '9E', 35: '9B', 36: '10O', 37: '10C', 38: '10E', 39: '10B', 
    40: '11O', 41: '11C', 42: '11E', 43: '11B', 44: '12O', 45: '12C', 46: '12E', 47: '12B', 
    48: 'J'
}

# Obtener el color para una clase dada
def get_color_for_class(class_name):
    if 'O' in class_name:
        return (0, 255, 255)  # Amarillo
    elif 'E' in class_name:
        return (255, 0, 0)  # Azul
    elif 'B' in class_name:
        return (0, 255, 0)  # Verde
    elif 'C' in class_name:
        return (0, 0, 255)  # Rojo
    elif 'J' in class_name:
        return (255, 0, 255)  # Violeta
    else:
        return (255, 255, 255)  # Blanco por defecto

# Función para detección en tiempo real y verificar "envido"
def detect_and_show_cards_real_time(model_path):
    # Cargar el modelo YOLOv8
    model = YOLO(model_path)

    # Capturar video de la cámara
    cap = cv2.VideoCapture(0)  # Usa 0 para la cámara por defecto
    
    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return

    card_detection_times = {}
    envido_called = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede recibir frame (fin del stream?)")
            break

        # Realizar la detección
        results = model(frame)
        
        current_time = time.time()
        detected_suits = []

        # Dibujar los bounding boxes en la imagen
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, class_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            class_name = class_names[class_id]
            label = f'{class_name} {conf:.2f}'
            color = get_color_for_class(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            suit = class_name[-1]
            detected_suits.append(suit)

            if suit not in card_detection_times:
                card_detection_times[suit] = current_time

        # Verificar si hay dos cartas del mismo palo detectadas por 3 segundos
        for suit in card_detection_times.keys():
            if detected_suits.count(suit) >= 2:
                if current_time - card_detection_times[suit] >= 3:
                    if not envido_called:
                        print("¡Envido!")
                        pygame.mixer.music.play()
                        envido_called = True
            else:
                card_detection_times[suit] = current_time

        # Mostrar el frame con los bounding boxes
        cv2.imshow('Detección de Cartas en Tiempo Real', frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar el recurso de captura y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()

# Ejemplo de uso
model_path = '100epochs_yolov8.pt'
detect_and_show_cards_real_time(model_path)
