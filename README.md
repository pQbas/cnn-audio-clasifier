# Audio Classification 

Sistema de clasificación de audio que identifica sonidos ambientales usando un modelo CNN ResNet.

**Qué hace?** 

- Clasifica archivos de audio en 50 categorías del dataset ESC-50 (ladridos de perro, lluvia, bocinas de auto, etc).

**Cómo funciona?**

- Convierte audio a espectrogramas mel
- Usa red neuronal ResNet para clasificación
- Retorna predicciones principales con puntajes de confianza

## Uso

1. Instalar dependencias con uv:

```bash
uv sync
```

2. Entrenar el modelo en Modal cloud:

```bash
make train
```

3. Desplegar servidor de inferencia:

```bash
make deploy
```

4. Probar el endpoint:

```bash
make test
```

## Estructura

- `src/` - Código principal del modelo y entrenamiento
- `ops/` - Scripts de despliegue en Modal  
- `tests/` - Archivos de pruebas
- `data/` - Dataset ESC-50
