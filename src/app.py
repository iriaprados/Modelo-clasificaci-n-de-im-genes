 
import numpy as np
import torch
import torch.utils.data as dataset # Para definir el dataset
from torchvision.datasets import ImageFolder # Cargar imagenes del dataset
from torchvision import transforms # Procesar y redimensionar imagenes
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim # Importar optimizadores para el modelo
from torch.utils.data import random_split # División de imagenes en entrenamiento y validación

# Definir las transformaciones de las imagenes para que puedan ser procesadas 
transformations= transforms.Compose([
    transforms.Resize((128, 128)), # Redimensionar imagenes al mismo tamaño 
    transforms.ToTensor(), # Convertir imagenes a tensores (forma vectorial de las imagenes)
])

# Introducir la informacion del dataset
dataset = ImageFolder(
    root= r'C:\Users\IRIAP\OneDrive\Documentos\Proyectos\Id_Imagen\data\train',
    transform=transformations) 

# Entrenar por batch de fotos el modelo 
dataLoader = DataLoader(
    dataset, # Dataset a cargar
    batch_size=32, # Tamaño del batch (32 fotos )
    shuffle=True, # Mezclar aleatoriamente las imagenes 
)

# Prueba el DataLoader
# for batch_idx, (images, labels) in enumerate(dataLoader):
#     print(f"Lote {batch_idx + 1}:")
#     print(f"  Tamaño del batch: {images.size()}")
#     print(f"  Etiquetas: {labels}")
#     break  # Solo para probar el primer lote
# Definir el modelo 


# Modelo de clasificacion
class clasification(nn.Module):

    def __init__(self):
        super(clasification, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) # Crear 16 filtros para detectar patrones
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # Reducir el tamaño de la imagen, para detectar patrones mas complejos
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Crea 32 nuevos filtros para detectar patrones
        self.fc1 = nn.Linear(32 * 32 * 32, 128) # Convertir imagenes a un vector de 128 dimensiones (nueronas)
        self.fc2 = nn.Linear(128, 2) # Dar dos salidas posibles a las neuroas (dos tipos de imagenes)
        self.relu = nn.ReLU() # Función de activación ReLU, normalización de los resultados
    
    # Que hace el modelo con cada imagen
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # Reduce el tamaño de la imagen, pasada por el primer filtro
        x = self.pool(self.relu(self.conv2(x))) # Reduce el tamaño de la imagen, pasada por el segundo filtro
        x = torch.flatten(x, 1)  # Aplanar la imagen para convertirla en un vector
        x = self.relu(self.fc1(x)) # Pasar el vector por la primera capa de neuronas
        x = self.fc2(x) # Pasar el vector por la segunda capa de neuronas
        return x  

# Definir el modelo 
model = clasification()
criterion = nn.CrossEntropyLoss() # Perdida del modelo 
optimizer = optim.Adam(model.parameters(), lr=0.001) 
epchs = 5 

for epoch in range(epchs):
    runLoss = 0.0 # Inicializar la perdida 

    # print(f"\n Entrenando Epoch {epoch + 1}/{epchs}...")

    for images, labels in dataLoader:
        optimizer.zero_grad() 
        output = model(images) # Pasar las imagenes por el modelo
        # print(f" Salida: {output.shape}")
        loss = criterion(output, labels) # Perdida de los esperado menos lo predicho
        loss.backward() 
        optimizer.step() # Actualizar los pesos del modelo segun la perdida

        runLoss += loss.item() # Sumar la perdida de cada uno de los batches
    
    mediaLoss = runLoss / len(dataLoader) # Calcular la perdida media de cada episiodio 

    # print(f"Epoch completado {epoch + 1}, Perdida media: {mediaLoss:.4f}")

# Dividir el dataset en entrenamiento y validación
train= int(0.8 * len(dataset))
val= len(dataset) - train

train_dataset, val_dataset = random_split(dataset, [train, val]) # División aleatoria del contenido entre entrenamiento y evaluación

print(f"Total imágenes: {len(dataset)}")
print(f"Imágenes de entrenamiento: {len(train_dataset)}")
print(f"Imágenes de validación: {len(val_dataset)}")


# Dividir en  batches las imágenes del entrenamiento y de la evaluación
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Cálculo de la precisión de la validación, modelo en modo evaluación 
for epoch in range(epchs):

    # ---- Entrenamiento ----
    model.train()
    runLoss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        runLoss += loss.item()

    train_loss = runLoss / len(train_loader)

    # ---- Validación ----
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    prescision = 100 * correct / total

    print(f"Epoch {epoch+1}/{epchs}, Perdida: {train_loss:.4f}, Precisión de la evaluación: {prescision:.2f}%")

torch.save(model.state_dict(), "modelo.pth")
print("Modelo guardado como modelo.pth")