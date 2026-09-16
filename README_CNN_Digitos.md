# Reconocimiento de Dígitos Manuscritos con Redes Neuronales Convolutivas (CNN)

Clasificador de imágenes construido con **TensorFlow/Keras** que reconoce dígitos manuscritos (0–9) a partir de imágenes de 8×8 píxeles, incluyendo una segunda iteración del modelo para comparar el efecto real de ajustar hiperparámetros.

## 🎯 Problema

A partir del dataset de dígitos manuscritos (1.797 imágenes de 8×8 píxeles en escala de grises, 10 clases balanceadas), el objetivo es entrenar una red neuronal convolutiva capaz de identificar qué dígito representa cada imagen, aprendiendo automáticamente las características relevantes (bordes, curvas, trazos) en lugar de definirlas manualmente.

El notebook parte con una explicación conceptual propia de qué es una CNN y qué hace cada tipo de capa (convolución, pooling, flatten, densa) antes de pasar a la implementación — pensado tanto para documentar el resultado como para explicar el razonamiento detrás de la arquitectura elegida.

## 🛠️ Stack y metodología

- **Framework:** TensorFlow / Keras (`Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`)
- **Preprocesamiento:** normalización de píxeles (rango original 0–16 → 0–1), `reshape` de vector plano (64,) a tensor de imagen (8, 8, 1)
- **División de datos:** 80% entrenamiento / 20% prueba con `stratify` para mantener la proporción de clases, más un 20% adicional del set de entrenamiento usado como validación durante el fit
- **Arquitectura:** dos bloques Conv2D+MaxPooling seguidos de Flatten, una capa densa de 128 neuronas, `Dropout(0.3)` para regularización, y salida softmax de 10 clases
- **Entrenamiento:** `Adam`, `sparse_categorical_crossentropy`, `EarlyStopping` monitoreando `val_loss` para evitar sobreentrenamiento
- **Evaluación:** accuracy y loss en test, matriz de confusión, `classification_report` (precision/recall/F1 por clase) y visualización de predicciones individuales
- **Experimentación:** un segundo modelo ("optimizado") con learning rate más bajo, batch size mayor, más épocas y una capa de pooling menos, comparado objetivamente contra el primero

## 📊 Resultados

**Modelo base:**

| Métrica | Valor |
|---|---|
| Accuracy (test) | 97.22% |
| Loss (test) | 0.0635 |
| F1-score (macro avg) | 0.9719 |

Solo 10 errores sobre 360 imágenes de prueba, concentrados principalmente en el dígito 8 (confundido ocasionalmente con 1, 2 y 7).

**Modelo optimizado (segunda iteración):**

| Métrica | Valor |
|---|---|
| Accuracy (test) | 95.28% |
| Loss (test) | 0.1274 |

**Conclusión del propio experimento:** el modelo optimizado obtuvo un desempeño *inferior* al original (−1.94 puntos porcentuales de accuracy). Este resultado se documenta explícitamente en el notebook como evidencia de que ajustar hiperparámetros no garantiza una mejora — el modelo más simple (con ambas capas de pooling) generalizó mejor que la variante con más épocas y una capa de reducción espacial menos.

## 🚀 Cómo ejecutarlo

```bash
# Clonar el repositorio
git clone <https://github.com/haroldrodriguezadm-png/Reconocimiento-de-D-gitos-Manuscritos-con-Redes-Neuronales-Convolutivas-CNN-/blob/main/README_CNN_Digitos.md>
cd <Reconocimiento-de-Digitos-Manuscritos-con-Redes-Neuronales-Convolutivas-CNN>

# Instalar dependencias
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn openpyxl

# Ejecutar el notebook
jupyter notebook "Reconocimiento de imágenes con redes neuronales convolutivas.ipynb"
```

El notebook espera el archivo de datos en la ruta indicada en la variable `archivo` (por defecto apunta a una ruta de Google Colab). Ajústala a la ubicación local antes de ejecutar.

**Requisitos:** Python 3.9+, TensorFlow 2.x

## 📁 Estructura del repositorio

```
├── Reconocimiento de imágenes con redes neuronales convolutivas.ipynb   # Notebook principal
├── data/
│   └── digitos_mnist_simple.xlsx                                        # Dataset de entrada
└── README.md
```

## 🔭 Limitaciones y próximos pasos

- El dataset usa imágenes de baja resolución (8×8), útil para fines didácticos pero muy por debajo de MNIST estándar (28×28); los resultados no son directamente comparables con benchmarks públicos.
- La comparación de hiperparámetros exploró una sola variante adicional; un `GridSearch`/`KerasTuner` sistemático permitiría explorar más combinaciones (número de filtros, tasa de dropout, arquitectura) de forma menos manual.
- Próximos pasos: aumentar la resolución de entrada o probar con un dataset de imágenes reales (no sintéticas de 8×8), aplicar data augmentation, y guardar el modelo entrenado (`model.save()`) para inferencia posterior.

## 👤 Autor

Harold Rodríguez B. — [LinkedIn](https://www.linkedin.com/in/harold-rodriguez-boisset/) 
