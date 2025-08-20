# Modelo de clasificación de imágenes de perros y gatos 

En este proyecto se ha implementado un red neuronal (CNN) en Python que gracias al uso de la librería **PyTorch**. Para poder entrenar el modelo se han usado dos tipos de imágenes divididos en carpetas.

## Índice
 * [Estructura del proyecto](#Estructura-del-proyecto)
 * [Instrucciones de uso](#Instrucciones-de-uso)
 * [Resumen del código](#Resumen-del-código)


## Estructura del proyecto 

Id_Imagen/\
│── data/\
│   └── train/\
│       ├── cats/\
│       └── dogs/\
│── src/\
│   └── app.py\
│── modelo.pth

## Instrucciones de uso 

### 1. Clona el repositorio en tú editor
git clone https://github.com/iriaprados/Modelo-clasificaci-n-de-im-genes.git\
cd Id_Imagen

### 2.Instalas las librerías
pip install torch torchvision numpy

### 3. Organiza y descarga las imagenes como se indica en la estructura del proyecto
Puedes descargas tu dataset con las imagenes directamente de esta página web → https://www.kaggle.com/competitions/perros-y-gatos
 
### 4. Entrena tú modelo 
python app.py

### 5. Comprueba que tu modelo se ha guardado automáticamente
En la carpeta src, debe aparecer un archivo llamado **modelo.pth**

## Resumen del código

### Transformaciones de imagen
* Redimensiona la imagen a 128x128
* Convierte la imagen en tensores

### Creación de los modelos de clasificación 
* Conv 1 + pool → detecta patrones sencillos
* Conv2 + Pool → detecta patrones más complejos
* FC1 → densidad que representa la imagen, uso de 128 neuronas
* FC2 → identifica si la salida de la imagen es un gato o un perro, uso de 2 neuronas
* Activación Relu

### Entrenamiento 
Se realiza un entrenamiento gracias a la optimización y el critério de pérdida en cada episodio. Esta información de recalacula cada 5 episodios. 

### Validación y entrenamiento
El modelo ha sido configurado para que trabaje en un 80% en el entrenamiento y en un 20% en la validación de las imagenas clasificadas. De forma adicional, el código indica la preción que presenta el modelo. 

### Guardado del modelo 

### Ejemplo de información de salida 
> Epoch 1/5, Perdida: 0.6821, Precisión de la evaluación: 58.50% \
Epoch 2/5, Perdida: 0.5934, Precisión de la evaluación: 72.10% \
...\
Modelo guardado como modelo.pth






