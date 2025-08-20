# Modelo de clasificación de imágenes de perros y gatos 

En este proyecto se ha implementado un red neuronal (CNN) en Python que gracias al uso de la librería **PyTorch**. Para poder entrenar el modelo se han usado dos tipos de imágenes divididos en carpetas.

## Índice
*[Estructura del proyecto](#Estructura del proyecto)
*[Instrucciones de uso](#Instrucciones de uso)

## Estructura del proyecto 

Id_Imagen/
│── data/
│   └── train/
│       ├── cats/
│       └── dogs/
│── src/
│   └── app.py
│── modelo.pth

## Instrucciones de uso 

### 1. Clona el repositorio en tú editor
git clone https://github.com/iriaprados/Modelo-clasificaci-n-de-im-genes.git
cd Id_Imagen

### 2.Instalas las librerías
pip install torch torchvision numpy

### 3. Organiza y descarga las imagenes como se indica en la estructura del proyecto
Puedes descargas tu dataset con las imagenes directamente de esta página web → https://www.kaggle.com/competitions/perros-y-gatos
 
### 4. Entrena tú modelo 
python app.py

### 5. Comprueba que tu modelo se ha guardado automáticamente
En la carpeta src, debe aparecer un archivo llamado **modelo.pth**
