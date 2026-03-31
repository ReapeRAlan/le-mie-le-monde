# 🐝 Le Mie, Le Monde — Detección, Seguimiento y Análisis de Comportamiento de Abejas

> *"Le mie, le monde"* — Las abejas, el mundo. Un sistema de visión por computadora para la detección en tiempo real, el seguimiento multi-objeto y la clasificación de comportamiento de abejas melíferas usando modelos YOLO y Deep Learning.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Características Principales](#-características-principales)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Stack Tecnológico](#-stack-tecnológico)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Dataset](#-dataset)
- [Scripts del Proyecto](#-scripts-del-proyecto)
- [Requisitos e Instalación](#-requisitos-e-instalación)
- [Uso](#-uso)
- [Clasificación de Comportamiento](#-clasificación-de-comportamiento)
- [Pipeline de Procesamiento](#-pipeline-de-procesamiento)
- [Proyección y Roadmap](#-proyección-y-roadmap)
- [Licencia](#-licencia)

---

## 🎯 Descripción del Proyecto

**Le Mie, Le Monde** es un proyecto de visión por computadora enfocado en la **detección, seguimiento y análisis de comportamiento de abejas melíferas** a partir de video. Utiliza modelos de detección de objetos de la familia YOLO (YOLOv5 y YOLOv8), combinados con el algoritmo de seguimiento multi-objeto **DeepSORT** y técnicas de Machine Learning (KMeans) para clasificar patrones de movimiento en tres categorías de comportamiento: **Exploración**, **Reclutamiento** y **Defensa**.

El proyecto incluye un **dataset etiquetado en formato YOLO** con 4 clases de abejas (obrera, zángano, abeja con polen y reina), y múltiples iteraciones del pipeline de procesamiento con mejoras progresivas en detección, suavizado de trayectorias y calidad de salida.

---

## ✨ Características Principales

- 🔍 **Detección multi-clase:** Identifica 4 tipos de abejas — `bee` (obrera), `drone` (zángano), `pollenbee` (con polen) y `queen` (reina)
- 🎯 **Seguimiento multi-objeto:** DeepSORT mantiene IDs consistentes entre fotogramas usando características visuales
- 📊 **Análisis de comportamiento:** Clasificación automática en Exploración, Reclutamiento o Defensa mediante KMeans
- 🔄 **Suavizado de trayectorias:** Filtro de Kalman y Savitzky-Golay para reducir ruido
- 📹 **Procesamiento de video en tiempo real** con visualización de bounding boxes, trayectorias e IDs
- 💾 **Exportación de video procesado** con todas las anotaciones superpuestas
- 🖼️ **Ecualización de histograma** en espacio LAB para mejorar el contraste

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Video Input   │────▶│  YOLO Detection  │────▶│  DeepSORT       │
│ (AbejasDetec..  │     │  (YOLOv5/v8)     │     │  Tracking       │
│      .mp4)      │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐     ┌────────▼────────┐
                        │  Behavior        │◀────│  Trajectory     │
                        │  Classification  │     │  Smoothing      │
                        │  (KMeans)        │     │  (Kalman)       │
                        └────────┬─────────┘     └─────────────────┘
                                 │
                        ┌────────▼─────────┐
                        │  Visualization   │
                        │  & Video Output  │
                        │  (.avi)          │
                        └──────────────────┘
```

---

## 🛠️ Stack Tecnológico

| Tecnología | Uso |
|---|---|
| **PyTorch** | Framework de Deep Learning, carga de modelos YOLO |
| **YOLOv5** | Detección de objetos con pesos personalizados (`best (2).pt`) |
| **YOLOv8 (Ultralytics)** | Detección optimizada de objetos pequeños |
| **DeepSORT** | Seguimiento multi-objeto en tiempo real |
| **OpenCV** | Procesamiento de video, visualización, ecualización de histograma |
| **NumPy** | Operaciones numéricas y vectoriales |
| **Scikit-learn (KMeans)** | Clasificación no supervisada de comportamiento |
| **SciPy (Savitzky-Golay)** | Suavizado de señales y trayectorias |
| **PyKalman** | Filtro de Kalman para estabilización de trayectorias |
| **Roboflow** | Origen del dataset etiquetado |

---

## 📁 Estructura del Repositorio

```
le-mie-le-monde/
├── main.py                  # Pipeline principal: detección + tracking + comportamiento
├── Abejas.py                # Versión educativa con documentación detallada
├── Test1.py                 # Variante con etiquetas de clase (bee/drone/pollenbee/queen)
├── test.py                  # Versión avanzada con YOLOv8 + Kalman + ecualización
├── test2.py                 # Versión producción con Kalman + Savitzky-Golay + video output
├── setup.py                 # Configuración de paquete (setuptools)
├── bee_dataset/             # Dataset YOLO (Roboflow Honey Bee Detection v4)
│   ├── data.yaml            # Configuración de clases y rutas del dataset
│   ├── train/               # 1,738 imágenes de entrenamiento + etiquetas
│   │   ├── images/
│   │   └── labels/
│   ├── valid/               # 176 imágenes de validación + etiquetas
│   │   ├── images/
│   │   └── labels/
│   └── test/                # 122 imágenes de prueba + etiquetas
│       ├── images/
│       └── labels/
├── yolov5/                  # Directorio YOLOv5 (descargado por torch.hub)
├── yolov9/                  # Directorio reservado para YOLOv9 (futuro)
├── .gitignore               # Archivos excluidos (modelos .pt, videos, venv)
└── .gitattributes           # Git LFS para archivos multimedia
```

---

## 📦 Dataset

El proyecto utiliza el dataset **Honey Bee Detection Model v4** de [Roboflow Universe](https://universe.roboflow.com/matt-nudi/honey-bee-detection-model-zgjnb/dataset/4), bajo licencia **CC BY 4.0**.

| Parámetro | Valor |
|---|---|
| **Imágenes base** | 883 |
| **Total con augmentación** | ~4,575 |
| **Formato** | YOLO v5 PyTorch |
| **Clases** | 4 (`bee`, `drone`, `pollenbee`, `queen`) |
| **Train / Valid / Test** | 1,738 / 176 / 122 imágenes |

### Augmentaciones Aplicadas
- Volteo horizontal (50% de probabilidad)
- Ajuste de brillo aleatorio (±20%)
- Desenfoque gaussiano aleatorio (0-10 px)
- Ruido salt-and-pepper (5% de píxeles)

---

## 📜 Scripts del Proyecto

### `main.py` — Pipeline Principal
- **Modelo:** YOLOv5 con pesos personalizados (`best (2).pt`)
- **Confianza:** Dinámica (base 0.2, ajustada por resolución)
- **Tracking:** DeepSORT (max_age=20, n_init=3)
- **Comportamiento:** KMeans sobre velocidades de trayectoria
- **Salida:** Visualización en tiempo real

### `Abejas.py` — Versión Educativa
- Idéntica funcionalidad a `main.py`
- **Confianza:** Estática (0.5)
- **Tracking:** DeepSORT (max_age=30, n_init=3)
- Incluye documentación detallada en español explicando cada componente

### `Test1.py` — Con Etiquetas de Clase
- Añade etiquetas de tipo de abeja (`bee`, `drone`, `pollenbee`, `queen`)
- **Confianza:** 0.3
- Muestra tipo + comportamiento por cada abeja rastreada

### `test2.py` — Versión con Kalman y Video Output
- **Mejoras:** Filtro de Kalman, Savitzky-Golay, normalización de escala
- **Modelo IOU:** 0.45 para más detecciones superpuestas
- **Tamaño mínimo:** 5×5 px para objetos pequeños
- **Tracking:** DeepSORT (max_age=25, n_init=5)
- **Salida:** `output_video.avi` (MJPG, 20 FPS)

### `test.py` — Versión Más Avanzada (YOLOv8)
- **Modelo:** YOLOv8 nano (`yolov8n.pt`)
- **Ecualización:** Histograma en espacio de color LAB
- **Tracking:** DeepSORT (max_age=25, n_init=5)
- **Salida:** `output_video.avi` (XVID, 30 FPS)
- Usa `defaultdict` para manejo eficiente de trayectorias

---

## ⚙️ Requisitos e Instalación

### Requisitos del Sistema
- Python 3.8+
- GPU con CUDA (recomendado para inferencia en tiempo real)
- Webcam o archivo de video (`.mp4`)

### Dependencias

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install deep-sort-realtime
pip install scikit-learn
pip install scipy
pip install pykalman
pip install ultralytics          # Para YOLOv8 (test.py)
```

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/ReapeRAlan/le-mie-le-monde.git
cd le-mie-le-monde

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install torch torchvision opencv-python numpy deep-sort-realtime scikit-learn scipy pykalman ultralytics

# 4. Colocar archivos necesarios en la raíz del proyecto:
#    - best (2).pt       → Pesos del modelo YOLOv5 entrenado
#    - AbejasDeteccion.mp4  → Video de entrada para procesamiento
```

---

## 🚀 Uso

### Ejecución Básica (YOLOv5 + DeepSORT)

```bash
python main.py
```

### Ejecución Avanzada (YOLOv8 + Kalman + Video Output)

```bash
python test.py
```

### Ejecución con Suavizado Kalman y Exportación de Video

```bash
python test2.py
```

> **Nota:** Presionar `q` para cerrar la ventana de visualización durante la ejecución.

---

## 🧠 Clasificación de Comportamiento

El sistema analiza las trayectorias de cada abeja rastreada y clasifica su comportamiento usando **KMeans clustering** sobre las velocidades de movimiento:

| Comportamiento | Descripción | Patrón de Velocidad |
|---|---|---|
| 🔵 **Exploración** | La abeja se mueve lentamente, explorando el entorno | Velocidad baja |
| 🟡 **Reclutamiento** | Movimiento moderado, patrón de comunicación (danza) | Velocidad media |
| 🔴 **Defensa** | Movimientos rápidos y agresivos, comportamiento defensivo | Velocidad alta |

El análisis se activa cuando una abeja tiene **más de 5 puntos** en su trayectoria, asegurando suficientes datos para una clasificación significativa.

---

## 🔄 Pipeline de Procesamiento

```
1. 📹 Lectura de frame del video
       │
2. 🖼️ Pre-procesamiento (ecualización de histograma LAB)
       │
3. 🔍 Detección con YOLO (YOLOv5/v8)
       │
4. 🎯 Filtrado por confianza (0.2 - 0.5)
       │
5. 📍 Seguimiento con DeepSORT (asignación de IDs)
       │
6. 📈 Cálculo de trayectoria y suavizado (Kalman / Savitzky-Golay)
       │
7. 🧠 Clasificación de comportamiento (KMeans)
       │
8. 🎨 Visualización (bounding boxes, IDs, trayectorias, etiquetas)
       │
9. 💾 Escritura a video de salida (.avi)
       │
10. 🖥️ Mostrar en ventana en tiempo real
```

---

## 🔮 Proyección y Roadmap

### Fase 1 — Mejoras Inmediatas
- [ ] **Refactorización del código:** Unificar los 5 scripts en un único pipeline modular con argumentos de línea de comandos
- [ ] **Corrección de bugs:** Corregir error de exponenciación en `Test1.py` (línea 34: `*2` → `**2`)
- [ ] **Configuración centralizada:** Archivo `config.yaml` para umbrales, rutas, hiperparámetros
- [ ] **Requirements.txt:** Archivo de dependencias con versiones fijadas
- [ ] **Rutas relativas:** Eliminar rutas absolutas de Windows en `data.yaml`

### Fase 2 — Mejoras de Modelo y Rendimiento
- [ ] **Integración de YOLOv9:** Implementar soporte para YOLOv9 (directorio ya reservado)
- [ ] **Fine-tuning de YOLOv8:** Entrenar YOLOv8 con el dataset de abejas para mejorar la detección específica
- [ ] **Optimización GPU:** Implementar inferencia por lotes (batch inference) para mayor velocidad
- [ ] **Modelo de comportamiento entrenado:** Reemplazar KMeans no supervisado con un clasificador supervisado (Random Forest / Red Neuronal) entrenado con datos etiquetados de comportamiento real
- [ ] **Detección de danza waggle:** Algoritmo específico para identificar la danza de reclutamiento de las abejas

### Fase 3 — Funcionalidades Avanzadas
- [ ] **Conteo automático de abejas:** Estadísticas de entrada/salida de la colmena por periodo de tiempo
- [ ] **Análisis de flujo de tráfico:** Mapa de calor de rutas frecuentes en la colmena
- [ ] **Detección de anomalías:** Alertas automáticas ante comportamientos inusuales (posible enjambre, enfermedad, depredador)
- [ ] **Soporte de cámara en vivo:** Integración con cámaras USB/IP para monitoreo continuo
- [ ] **Dashboard web:** Interfaz web con visualización en tiempo real, gráficas de tendencias y alertas (Flask/Streamlit)
- [ ] **API REST:** Endpoints para integración con otros sistemas de monitoreo apícola

### Fase 4 — Escalabilidad y Producción
- [ ] **Despliegue en edge:** Optimización con TensorRT/ONNX para Raspberry Pi o NVIDIA Jetson
- [ ] **Base de datos temporal:** Almacenamiento de trayectorias y comportamientos en base de datos para análisis histórico
- [ ] **Multi-cámara:** Soporte para múltiples colmenas monitoreadas simultáneamente
- [ ] **Modelo de salud de colmena:** Machine Learning predictivo que correlacione patrones de comportamiento con indicadores de salud
- [ ] **Integración IoT:** Sensores de temperatura, humedad y peso combinados con datos de visión
- [ ] **Aplicación móvil:** Notificaciones y monitoreo remoto para apicultores

### Fase 5 — Investigación y Ciencia
- [ ] **Publicación académica:** Paper describiendo la metodología de clasificación de comportamiento
- [ ] **Dataset ampliado:** Contribuir al dataset de Roboflow con más imágenes etiquetadas
- [ ] **Modelos pre-entrenados públicos:** Publicar pesos optimizados para la comunidad apícola
- [ ] **Colaboración con biólogos:** Validación científica de las clasificaciones de comportamiento
- [ ] **Análisis de polinización:** Extensión del sistema para rastrear abejas en campos de cultivo y medir actividad polinizadora

---

## 📊 Evolución de los Scripts

El proyecto muestra una **progresión iterativa** clara:

```
Abejas.py (educativo)
    │
    ▼
main.py (base funcional + confianza dinámica)
    │
    ▼
Test1.py (+ etiquetas de clase)
    │
    ▼
test2.py (+ Kalman + video output + tamaño mínimo)
    │
    ▼
test.py (+ YOLOv8 + ecualización LAB + XVID 30fps)
```

---

## 📄 Licencia

- **Dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Honey Bee Detection Model por [Matt Nudi / Roboflow](https://universe.roboflow.com/matt-nudi/honey-bee-detection-model-zgjnb/dataset/4)
- **YOLOv5/v8:** [AGPL-3.0](https://github.com/ultralytics/yolov5/blob/master/LICENSE) — Ultralytics
- **Código del proyecto:** Consultar con los autores

---

<p align="center">
  <i>Desarrollado con 🐝 para la conservación y estudio de las abejas melíferas</i>
</p>