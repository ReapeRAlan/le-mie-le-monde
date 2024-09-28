import torch
import cv2
import numpy as np
import pathlib
from deep_sort_realtime.deepsort_tracker import DeepSort  # Rastreador de objetos
from sklearn.cluster import KMeans  # Para clasificación de comportamiento
from scipy.signal import savgol_filter  # Para suavizado de trayectorias

# Librería para filtro de Kalman
from pykalman import KalmanFilter

# Corregir la ruta de pathlib si es necesario
pathlib.PosixPath = pathlib.WindowsPath

# Cargar modelo personalizado entrenado para abejas (best.pt)
model_path = "best (2).pt"  # Ruta a tu modelo personalizado
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Ajustar el IOU para mejorar NMS
model.iou = 0.45  # Reducir el umbral IOU para permitir más detecciones solapadas

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
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG es un buen codec para mantener calidad
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Diccionario para almacenar las trayectorias de las abejas
trajectories = {}

# Lista de nombres de las clases (ajústala según tu modelo)
class_names = ["bee", "drone", "pollenbee", "queen"]

# Parámetros de suavizado
window_size = 7  # Tamaño de la ventana del suavizado (más pequeño para más sensibilidad)
poly_order = 3   # Orden del polinomio para el suavizado

# Umbral para detección mínima basada en tamaño
min_width, min_height = 5, 5  # Reducir el tamaño mínimo para detectar abejas más pequeñas

# Función para análisis de comportamiento
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

# Función para suavizar trayectorias con filtro de Kalman
def kalman_smooth(trajectory):
    trajectory = np.array(trajectory)
    
    if len(trajectory) < 2:
        return trajectory  # Si no hay suficientes puntos, no suavizar
    
    # Definir el filtro de Kalman
    kf = KalmanFilter(initial_state_mean=trajectory[0], n_dim_obs=2)
    
    # Definir las matrices de transición y observación (estado simple de posición)
    kf = kf.em(trajectory, n_iter=5)
    
    # Aplicar el filtro de Kalman a la trayectoria
    smoothed_trajectory, _ = kf.smooth(trajectory)
    
    return smoothed_trajectory

# Procesar el video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Asegurarse de que el frame tenga la profundidad correcta de bits y tipo de datos (8 bits, rango 0-255)
    frame = cv2.convertScaleAbs(frame)  # Esto asegura que los valores estén en el rango correcto para cv2

    # Ejecutar inferencia con el modelo personalizado
    results = model(frame)

    # Extraer las detecciones con umbral de confianza más bajo y tamaño mínimo reducido
    detections = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        width, height = int(box[2] - box[0]), int(box[3] - box[1])
        if conf > 0.35 and width > min_width and height > min_height:  # Umbral de confianza reducido y tamaño mínimo ajustado
            x1, y1, x2, y2 = map(int, box)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))

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
            
            # Dibujar el bounding box (grosor 2, color azul)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Añadir punto a la trayectoria y suavizarla
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((center_x, center_y))
            
            smoothed_trajectory = kalman_smooth(trajectories[track_id])

            # Dibujar trayectoria suavizada (línea verde, grosor 2)
            for i in range(1, len(smoothed_trajectory)):
                cv2.line(frame, tuple(map(int, smoothed_trajectory[i - 1])), tuple(map(int, smoothed_trajectory[i])), (0, 255, 0), 2)
            
            # Mostrar tipo de abeja
            cls_id = int(detections[0][2])  # Obtener el índice de clase correctamente
            class_label = class_names[cls_id]  # Obtener nombre de la clase
            cv2.putText(frame, class_label, (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Analizar comportamiento si hay suficientes puntos
            if len(smoothed_trajectory) > 5:
                behavior = analyze_trajectory(smoothed_trajectory)
                cv2.putText(frame, behavior, (int(x1), int(y2) + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Etiqueta de comportamiento más clara

    # Mostrar el frame con detecciones
    cv2.imshow("Detections", frame)

    # Guardar el frame procesado en el video de salida
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()  # Asegurarse de liberar el archivo de salida si guardas el video
cv2.destroyAllWindows()
