import torch
import cv2
import numpy as np
import pathlib
from deep_sort_realtime.deepsort_tracker import DeepSort  # Importamos el rastreador


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Cargar el modelo YOLOv5 entrenado
model_path = "best (2).pt"  # Ruta al archivo de pesos en tu máquina local
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path) #torch.hub.load:Este es un método de la librería PyTorch que te permite descargar e instanciar modelos de una biblioteca en GitHub o de un repositorio compatible. 
#ultralytics/yolov5':Este es el repositorio en GitHub del que se está cargando el modelo. En este caso, se refiere al repositorio de YOLOv5, que es mantenido por la organización Ultralytics
#custom':Este argumento especifica qué tipo de modelo quieres cargar. La palabra 'custom' se usa cuando tienes un modelo personalizado, es decir, un modelo que has entrenado tú mismo en un conjunto de datos específico. De esta manera, en lugar de cargar un modelo preentrenado (como 'yolov5s' o 'yolov5m'), estás cargando un modelo entrenado por ti.



# Inicializar DeepSORT para el seguimiento de las abejas
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
#DeepSort utiliza tanto información de la detección (posición, tamaño) como características visuales para seguir objetos con mayor precisión.
#max_age=30 30 fotogramas es el límite; si un objeto no es detectado en 30 fotogramas consecutivos, su seguimiento se detendrá.
#n_init=3 indica cuántas detecciones consecutivas deben observarse antes de considerar un objeto como una pista válida.  ayuda a eevitar falsos
#nms_max_overlap=1.0:Este parámetro controla el umbral de superposición máxima permitido para la Supresión de Múltiples Detecciones (Non-Maximum Suppression, NMS). Un valor de 1.0 significa que la NMS permitirá superposiciones completas entre las cajas delimitadoras de objetos.






# Abrir el video
video_path = "AbejasDeteccion.mp4"
cap = cv2.VideoCapture(video_path)

# Diccionario para almacenar las trayectorias de las abejas
trajectories = {}
#trajectories = {}:Aquí se inicializa un diccionario vacío llamado trajectories, que se va a utilizar para almacenar las trayectorias de los objetos detectados o rastreados. En este caso, cada abeja rastreada podría estar asociado a una clave (ID del objeto), y el valor sería una lista de puntos o coordenadas que representan la trayectoria (su recorrido a lo largo del tiempo en el video).
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    #while cap.isOpened()::Esto es un bucle while que se ejecuta siempre que la captura de video (cap) esté abierta. La función cap.isOpened() devuelve True si el archivo de video o la cámara está lista para ser leída, por lo que este bucle controla el procesamiento de fotograma
    #ret, frame = cap.read():Aquí se están leyendo los fotogramas del video usando el método cap.read(). Esta función devuelve dos valores:ret: Es un valor booleano que indica si la lectura del frame fue exitosa. Si es True, significa que un nuevo frame fue leído correctamente.frame: Es el propio frame del video (una imagen), que se utilizará para hacer el procesamiento posterior, como la detección de objetos o el seguimiento.
    
    
    
    
    # Ejecutar inferencia con el modelo YOLOv5 entrenado
    results = model(frame)
    
    # Extraer las detecciones
    detections = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        if conf > 0.5:  # Umbral de confianza
            x1, y1, x2, y2 = map(int, box)
            width, height = x2 - x1, y2 - y1
            detections.append(([x1, y1, width, height], conf, int(cls)))
    #detections = []:Se inicializa una lista vacía llamada detections. Aquí se almacenarán las detecciones válidas (aquellas que superan el umbral de confianza establecido).for *box, conf, cls in results.xyxy[0].cpu().numpy()::Este es un bucle for que recorre cada una de las detecciones realizadas por el modelo YOLOv5.results.xyxy[0] contiene las coordenadas y los valores de confianza para las detecciones en formato tensor. cpu().numpy(): Se convierte el tensor a un array de NumPy para poder manipularlo fácilmente. La conversión a la CPU es necesaria porque, inicialmente, los resultados suelen estar en la GPU (si es utilizada).*box, conf, cls: Se desglosan los valores en:*box: Las coordenadas de la caja delimitadora (x1, y1, x2, y2).conf: El valor de confianza de la detección (qué tan seguro está el modelo de que el objeto detectado es correcto). cls: La clase del objeto detectado (qué tipo de objeto es, por ejemplo, persona, coche, etc.). if conf > 0.5::Se verifica si el valor de confianza de la detección es mayor que 0.5. Esto significa que solo se considerarán las detecciones cuya confianza sea superior al 50%. Las detecciones con una confianza inferior se descartan. x1, y1, x2, y2 = map(int, box):Se extraen y convierten las coordenadas de la caja delimitadora (x1, y1, x2, y2) a enteros. Estas coordenadas representan el borde superior izquierdo (x1, y1) y el borde inferior derecho (x2, y2) de la caja que encierra al objeto detectado. width, height = x2 - x1, y2 - y1: Se calculan el ancho y alto de la caja delimitadora usando las coordenadas extraídas. Esto se hace restando las coordenadas de los bordes. detections.append(([x1, y1, width, height], conf, int(cls))): Se agrega la detección a la lista detections. Cada elemento de la lista contiene: [x1, y1, width, height]: Las coordenadas del borde superior izquierdo de la caja, junto con su ancho y alto. conf: El valor de confianza de la detección. int(cls): La clase del objeto detectado, convertida en un entero

    # Convertir detecciones al formato esperado por DeepSORT
    if len(detections) > 0:
        bboxes = np.array([d[0] for d in detections])
        confidences = np.array([d[1] for d in detections])
    #if len(detections) > 0::Se verifica si la lista detections contiene algún elemento (es decir, si hay alguna detección válida). Si la lista tiene un tamaño mayor que cero, significa que al menos una detección ha sido realizada. Si no hay detecciones, el bloque de código dentro de este if no se ejecuta
    #bboxes = np.array([d[0] for d in detections]):Se crea un array de NumPy llamado bboxes que contiene solo las cajas delimitadoras (bounding boxes) de cada detección.[d[0] for d in detections] es una comprensión de listas que toma el primer elemento (d[0]) de cada detección en la lista detections. El primer elemento de cada detección en la lista es la caja delimitadora, que contiene las coordenadas y dimensiones del objeto detectado (como [x1, y1, width, height]). El resultado es una lista de cajas delimitadoras, que luego se convierte en un array de NumPy usando np.array() para facilitar el procesamiento posterior
    #confidences = np.array([d[1] for d in detections]):De manera similar, se crea un array de NumPy llamado confidences que contiene los valores de confianza de cada detección[d[1] for d in detections] toma el segundo elemento (d[1]) de cada detección en la lista detections, que corresponde al valor de confianza de la detección. Este valor representa qué tan seguro está el modelo de que la detección es correcta, y se almacena en un array de NumPy para facilitar su manipulación    
        
        
        
        # Crear la lista de detecciones en el formato correcto
        dets = [([x, y, w, h], conf) for (x, y, w, h), conf in zip(bboxes, confidences)]
        
        tracked_objects = tracker.update_tracks(dets, frame=frame)
        #tracker: Es una instancia de un rastreador de objetos, como DeepSort en tu caso. Este rastreador es responsable de seguir el movimiento de los objetos a través de los fotogramas del video. update_tracks: Es un método del rastreador que se utiliza para actualizar el seguimiento de los objetos en función de las nuevas detecciones. dets: Es una lista o array de las detecciones actuales, donde cada detección incluye una caja delimitadora (bounding box) y otros detalles como la confianza y la clase del objeto detectado. Estas detecciones son las entradas para actualizar el rastreador. frame=frame: Es el fotograma actual del video que se está procesando. El rastreador utiliza este fotograma para actualizar las posiciones y el estado de los objetos rastreados. tracked_objects: Es el resultado de la llamada al método update_tracks. Contiene la información actualizada sobre los objetos que el rastreador sigue. Esta información generalmente incluye:
        
        
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltwh()
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            # Obtener el centro del bounding box x1, y1, x2, y2: Representan las coordenadas de la caja delimitadora (bounding box) de un objeto detectado
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Dibujar el bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2) dibuja un rectángulo azul alrededor de la caja delimitadora detectada en el frame. frame es la imagen en la que se dibuja el rectángulo. (int(x1), int(y1)) son las coordenadas de la esquina superior izquierda del rectángulo, y (int(x2), int(y2)) son las coordenadas de la esquina inferior derecha. (255, 0, 0) define el color azul del rectángulo en formato BGR. 2 es el grosor de la línea del rectángulo.
            #cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) añade un texto verde en la imagen. frame es la imagen en la que se dibuja el texto. f'ID: {int(track_id)}' es el texto que muestra el identificador del objeto (track_id), convertido a entero. (int(x1), int(y1) - 10) son las coordenadas donde se coloca el texto, un poco arriba de la esquina superior izquierda del rectángulo. cv2.FONT_HERSHEY_SIMPLEX es el tipo de fuente, 0.5 es la escala del texto, (0, 255, 0) es el color verde del texto, y 2 es el grosor del texto.
            
            
            # Añadir el punto a la trayectoria
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((center_x, center_y))
            
            #if track_id not in trajectories: verifica si el track_id del objeto no está ya en el diccionario trajectories. Si track_id no se encuentra como una clave en trajectories, significa que aún no hemos registrado ninguna trayectoria para este identificador.
            #trajectories[track_id] = [] inicializa una nueva entrada en el diccionario trajectories con la clave track_id y le asigna una lista vacía. Esto prepara una lista para almacenar las coordenadas del centro del objeto a medida que se mueven a través de los frames.
            #trajectories[track_id].append((center_x, center_y)) añade las coordenadas (center_x, center_y) del centro del objeto a la lista asociada con el track_id. Esto actualiza la trayectoria del objeto con su nueva posición en el frame actual.

 
            
            # Dibujar la trayectoria
            for i in range(1, len(trajectories[track_id])):
                if trajectories[track_id][i - 1] is None or trajectories[track_id][i] is None:
                    continue
                cv2.line(frame, trajectories[track_id][i - 1], trajectories[track_id][i], (0, 255, 0), 2)
                
            #`for i in range(1, len(trajectories[track_id])):` itera sobre los índices de la lista de coordenadas para el objeto con `track_id`. Comienza en 1 porque se compara cada punto con el anterior.
            #if trajectories[track_id][i - 1] is None or trajectories[track_id][i] is None:` verifica si el punto anterior o el punto actual en la trayectoria es `None`. Si cualquiera de los puntos es `None`, se omite el paso de dibujo de la línea para evitar errores.
          #`cv2.line(frame, trajectories[track_id][i - 1], trajectories[track_id][i], (0, 255, 0), 2)` dibuja una línea verde en el `frame` entre el punto anterior `(trajectories[track_id][i - 1])` y el punto actual `(trajectories[track_id][i])`. El color verde es especificado por `(0, 255, 0)`, y el grosor de la línea es 2 píxeles.

    # Mostrar el frame con las detecciones y trayectorias
    cv2.imshow("Detections", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #para salir del video
        break

cap.release()
cv2.destroyAllWindows()
