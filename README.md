# Audio Classification 

Sistema de clasificación de audio que identifica sonidos ambientales usando un modelo CNN ResNet.

## Qué hace

Clasifica archivos de audio en 50 categorías del dataset ESC-50 (ladridos de perro, lluvia, bocinas de auto, etc).

## Cómo funciona

- Convierte audio a espectrogramas mel
- Usa red neuronal ResNet para clasificación
- Retorna predicciones principales con puntajes de confianza

## Instalación

Instalar dependencias con uv:

```bash
uv sync
```

## Entrenamiento

Entrenar el modelo en Modal cloud:

```bash
modal run ops/train.py
```

## Inferencia

Desplegar servidor de inferencia:

```bash
modal run ops/inference.py
```

Probar el endpoint:

```bash
INFERENCE_ENDPOINT_URL=<tu-endpoint-url> pytest tests/test_inference_endpoint.py -v
```

## Estructura

- `src/` - Código principal del modelo y entrenamiento
- `ops/` - Scripts de despliegue en Modal  
- `tests/` - Archivos de pruebas
- `data/` - Dataset ESC-50
