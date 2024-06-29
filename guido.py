import cv2
from ultralytics import YOLO

# Definir las clases para las que fueron entrenados los modelos
classes_suits = ['O', 'C', 'E', 'B', 'J']
classes_numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'J']

# Colores para los palos
color_map = {
    'O': (0, 255, 255),   # Amarillo para oro
    'C': (0, 0, 255),     # Rojo para copas
    'E': (255, 0, 0),     # Azul para espadas
    'B': (0, 255, 0),     # Verde para bastos
    'J': (255, 255, 255)  # Blanco para joker
}

# Cargar el modelo YOLO entrenado para suits y numbers
model_suits = YOLO("suits.pt")
model_numbers = YOLO("numbers.pt")

# Inicializar la captura de la cámara
cap = cv2.VideoCapture(0)  # El argumento 0 indica que se usará la cámara predeterminada (puedes cambiarlo según tu configuración)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Obtener la resolución máxima soportada por la cámara
max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir el tamaño de la ventana
window_name = 'YOLO Camera'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

def calculate_iou(box1, box2):
    """ Calcular el Intersection over Union (IoU) de dos bounding boxes. """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


# Procesar los fotogramas de la cámara
while True:
    # Capturar un fotograma
    ret, frame = cap.read()
    
    # Verificar si se capturó correctamente
    if not ret:
        print("Error: No se pudo capturar el fotograma.")
        break
    
    # Realizar la detección de objetos con el modelo suits.pt
    results_suits = model_suits(frame)
    results_numbers = model_numbers(frame)
    
    # Obtener las predicciones de suits.pt
    predictions_suits = results_suits[0].boxes.xyxy.cpu().numpy()  # Obtener las cajas delimitadoras como numpy array
    classes_suits = results_suits[0].boxes.cls.cpu().numpy()  # Obtener las clases predichas
    scores_suits = results_suits[0].boxes.conf.cpu().numpy()  # Obtener las puntuaciones de confianza

    # Obtener las predicciones de numbers.pt
    predictions_numbers = results_numbers[0].boxes.xyxy.cpu().numpy()  # Obtener las cajas delimitadoras como numpy array
    classes_numbers = results_numbers[0].boxes.cls.cpu().numpy()  # Obtener las clases predichas
    scores_numbers = results_numbers[0].boxes.conf.cpu().numpy()  # Obtener las puntuaciones de confianza

    # Combinar las predicciones en un solo bounding box
    combined_boxes = []
    for box_suits, cls_suits, score_suits in zip(predictions_suits, classes_suits, scores_suits):
        xmin_suits, ymin_suits, xmax_suits, ymax_suits = box_suits
        suit_name = model_suits.names[int(cls_suits)]
        
        best_iou = 0
        best_box_numbers = None
        best_cls_numbers = None
        best_score_numbers = None
        
        for box_numbers, cls_numbers, score_numbers in zip(predictions_numbers, classes_numbers, scores_numbers):
            iou = calculate_iou(box_suits, box_numbers)
            if iou > best_iou:
                best_iou = iou
                best_box_numbers = box_numbers
                best_cls_numbers = cls_numbers
                best_score_numbers = score_numbers
        
        if best_box_numbers is not None:
            xmin_num, ymin_num, xmax_num, ymax_num = best_box_numbers
            number_name = model_numbers.names[int(best_cls_numbers)]
            
            xmin_combined = min(xmin_suits, xmin_num)
            ymin_combined = min(ymin_suits, ymin_num)
            xmax_combined = max(xmax_suits, xmax_num)
            ymax_combined = max(ymax_suits, ymax_num)
            
            combined_boxes.append((xmin_combined, ymin_combined, xmax_combined, ymax_combined, suit_name, number_name, (score_suits + best_score_numbers) / 2))
    
    # Dibujar las cajas delimitadoras combinadas y etiquetas en el fotograma
    for xmin, ymin, xmax, ymax, suit_name, number_name, score in combined_boxes:
        color = color_map[suit_name]  # Obtener el color según el palo
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(frame, f"{suit_name}{number_name} {score:.2f}", (int(xmin), int(ymin) - 10),
                    cv2.QT_FONT_NORMAL, 0.5, color, 2)
    
    # Mostrar el fotograma con las detecciones
    cv2.imshow(window_name, frame)
    
    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()