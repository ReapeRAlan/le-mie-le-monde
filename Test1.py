import torch
import cv2
import numpy as np
import pathlib
from deep_sort_realtime.deepsort_tracker import DeepSort  # Rastreador de objetos
from sklearn.cluster import KMeans  # Para clasificación de comportamiento

# Corregir la ruta de pathlib si es necesario
pathlib.PosixPath = pathlib.WindowsPath

# Cargar modelo personalizado entrenado para abejas (best.pt)
model_path = "best (2).pt"  # Ruta a tu modelo personalizado
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Inicializar DeepSORT para el seguimiento de las abejas
tracker = DeepSort(max_age=20, n_init=3, nms_max_overlap=1.0)

# Abrir el video
video_path = "AbejasDeteccion.mp4"
cap = cv2.VideoCapture(video_path)

# Diccionario para almacenar las trayectorias de las abejas
trajectories = {}

# Lista de nombres de las clases (ajústala según tu modelo)
class_names = ["bee", "drone", "pollenbee", "queen"]

# Función para analizar la trayectoria y predecir comportamiento
def analyze_trajectory(track_points):
    velocities = []
    for i in range(1, len(track_points)):
        x1, y1 = track_points[i - 1]
        x2, y2 = track_points[i]
        velocity = np.sqrt((x2 - x1)*2 + (y2 - y1)*2)
        velocities.append(velocity)
    
    if len(velocities) > 1:
        velocities = np.array(velocities).reshape(-1, 1)
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(velocities)
        label = kmeans.predict([[np.mean(velocities)]])
        if label == 0:
            return "Exploración"
        elif label == 1:
            return "Reclutamiento"
        else:
            return "Defensa"
    return "Indeterminado"

# Procesar el video sin redimensionar para mantener la precisión del modelo personalizado
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ejecutar inferencia con el modelo personalizado
    results = model(frame)

    # Extraer las detecciones
    detections = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        if conf > 0.3:  # Umbral de confianza ajustado
            x1, y1, x2, y2 = map(int, box)
            width, height = x2 - x1, y2 - y1
            detections.append(([x1, y1, width, height], conf, int(cls)))

    # Convertir detecciones al formato esperado por DeepSORT
    if len(detections) > 0:
        bboxes = np.array([d[0] for d in detections])
        confidences = np.array([d[1] for d in detections])
        classes = np.array([d[2] for d in detections])
        
        # Crear la lista de detecciones en el formato correcto
        dets = [([x, y, w, h], conf) for (x, y, w, h), conf in zip(bboxes, confidences)]
        
        tracked_objects = tracker.update_tracks(dets, frame=frame)
        
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltwh()
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            # Obtener el centro del bounding box
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Dibujar el bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Añadir el punto a la trayectoria
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((center_x, center_y))
            
            # Dibujar la trayectoria
            for i in range(1, len(trajectories[track_id])):
                if trajectories[track_id][i - 1] is None or trajectories[track_id][i] is None:
                    continue
                cv2.line(frame, trajectories[track_id][i - 1], trajectories[track_id][i], (0, 255, 0), 2)
            
            # Mostrar el tipo de abeja (etiqueta de clase)
            cls_id = int(classes[0])  # Obtener la clase de la detección actual
            class_label = class_names[cls_id]  # Obtener el nombre de la clase
            cv2.putText(frame, class_label, (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Añadir etiqueta de clase

            # Analizar la trayectoria y predecir el comportamiento
            if len(trajectories[track_id]) > 5:  # Si hay suficientes puntos en la trayectoria
                behavior = analyze_trajectory(trajectories[track_id])
                cv2.putText(frame, behavior, (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Añadir comportamiento

    # Mostrar el frame con las detecciones, tipos de abejas y comportamiento
    cv2.imshow("Detections", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()