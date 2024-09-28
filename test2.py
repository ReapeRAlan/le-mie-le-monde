import torch
import cv2
import numpy as np
import pathlib
from ultralytics import YOLO  # YOLOv8, una versión más rápida y eficiente
from deep_sort_realtime.deepsort_tracker import DeepSort  # Rastreador de objetos
from sklearn.cluster import KMeans  # Para clasificación de comportamiento
from scipy.signal import savgol_filter  # Para suavizado de trayectorias
from pykalman import KalmanFilter  # Filtro de Kalman para suavizar trayectorias
from collections import defaultdict

# Corregir la ruta de pathlib si es necesario
pathlib.PosixPath = pathlib.WindowsPath

# Cargar modelo YOLOv8 (nano para objetos pequeños)
model = YOLO('yolov8n.pt')  # Puedes usar 'yolov8n.pt' o 'yolov8s.pt' para mayor precisión en detección de objetos pequeños.

# Inicializar DeepSORT para el seguimiento de las abejas
tracker = DeepSort(max_age=25, n_init=5, nms_max_overlap=1.0)

# Abrir el video de manera segura
video_path = "AbejasDeteccion.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Error al abrir el video: {video_path}")

# Obtener el tamaño del video original para mantener la resolución
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definir el codec de salida y el archivo de video de salida (opcional si deseas guardar el video procesado)
output_path = 'output_video.avi'  # Cambiar ruta si se necesita guardar
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID mantiene una alta calidad
out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))  # Mayor FPS para un video más fluido

# Diccionario para almacenar las trayectorias de las abejas
trajectories = defaultdict(list)  # Usar defaultdict para manejar mejor las trayectorias

# Lista de nombres de las clases (ajústala según tu modelo)
class_names = ["bee", "drone", "pollenbee", "queen"]

# Parámetros de suavizado
window_size = 7  # Tamaño de la ventana del suavizado (más pequeño para más sensibilidad)
poly_order = 3   # Orden del polinomio para el suavizado

# Filtro de Kalman avanzado
def kalman_smooth(trajectory):
    kf = KalmanFilter(initial_state_mean=[0, 0], n_dim_obs=2)
    smoothed_trajectory, _ = kf.smooth(np.array(trajectory))
    return smoothed_trajectory

# Función para analizar la trayectoria y predecir comportamiento
def analyze_trajectory(track_points):
    if len(track_points) < 2:
        return "Indeterminado"
    
    velocities = np.linalg.norm(np.diff(track_points, axis=0), axis=1)
    
    if len(velocities) > 1:
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(velocities.reshape(-1, 1))
        label = kmeans.predict([[np.mean(velocities)]])[0]
        behaviors = ["Exploración", "Reclutamiento", "Defensa"]
        return behaviors[label]
    
    return "Indeterminado"

# Procesar el video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mantener la calidad aplicando ecualización de histograma para mejorar contraste
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Ejecutar inferencia con el modelo YOLOv8
    results = model(frame, conf=0.3, iou=0.45)  # Confianza baja para detectar más objetos pequeños

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Caja delimitadora
            conf = box.conf[0].item()  # Confianza
            cls = int(box.cls[0])  # Clase
            if conf > 0.3:  # Umbral de confianza ajustado
                width, height = x2 - x1, y2 - y1
                detections.append(([x1, y1, width, height], conf, cls))

    # Convertir detecciones al formato esperado por DeepSORT
    if detections:
        bboxes = np.array([d[0] for d in detections])
        confidences = np.array([d[1] for d in detections])
        dets = [([x, y, w, h], conf) for (x, y, w, h), conf in zip(bboxes, confidences)]
        
        tracked_objects = tracker.update_tracks(dets, frame=frame)
        
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltwh()
            x1, y1, w, h = bbox
            center_x, center_y = int(x1 + w / 2), int(y1 + h / 2)
            
            # Dibujar el bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Añadir punto a la trayectoria y suavizarla
            trajectories[track_id].append((center_x, center_y))
            smoothed_trajectory = kalman_smooth(trajectories[track_id])

            # Dibujar trayectoria suavizada
            for i in range(1, len(smoothed_trajectory)):
                cv2.line(frame, tuple(map(int, smoothed_trajectory[i - 1])), tuple(map(int, smoothed_trajectory[i])), (0, 255, 0), 2)
            
            # Mostrar tipo de abeja
            cls_id = int(detections[0][2])  # Obtener el índice de clase correctamente
            class_label = class_names[cls_id]  # Obtener nombre de la clase
            cv2.putText(frame, class_label, (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Analizar comportamiento si hay suficientes puntos
            if len(smoothed_trajectory) > 5:
                behavior = analyze_trajectory(smoothed_trajectory)
                cv2.putText(frame, behavior, (int(x1), int(y2) + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Mostrar el frame con detecciones
    cv2.imshow("Detections", frame)

    # Guardar el frame procesado en el video de salida
    out.write(frame)

    # Controlar velocidad de visualización
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
