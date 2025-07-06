#!/usr/bin/env python3
# 🦷 Entrenamiento de clasificación para dental_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import sys
from datetime import datetime

def create_data_loaders(data_dir, batch_size=32):
    """Crea data loaders para entrenamiento."""
    
    # Verificar que existan los directorios
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Si no existen train/val directamente, buscar en subdirectorios
    if not os.path.exists(train_dir):
        # Buscar el primer subdirectorio que contenga train/val
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        for subdir in subdirs:
            potential_train = os.path.join(data_dir, subdir, 'train')
            potential_val = os.path.join(data_dir, subdir, 'val')
            if os.path.exists(potential_train) and os.path.exists(potential_val):
                train_dir = potential_train
                val_dir = potential_val
                print(f"📁 Usando dataset en: {subdir}")
                break
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"❌ Directorio de entrenamiento no encontrado: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"❌ Directorio de validación no encontrado: {val_dir}")
    
    print(f"📂 Train: {train_dir}")
    print(f"📂 Val: {val_dir}")
    
    # Transformaciones
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, train_dataset.classes

def create_model(num_classes):
    """Crea modelo ResNet para clasificación."""
    model = models.resnet50(pretrained=True)
    
    # Congelar capas iniciales
    for param in model.parameters():
        param.requires_grad = False
    
    # Reemplazar clasificador final
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=50):
    """Entrena el modelo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Directorio de salida
    output_dir = f"./logs/dental_dataset_classification"
    os.makedirs(output_dir, exist_ok=True)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # Entrenamiento
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 
                      os.path.join(output_dir, 'best_model.pth'))
        
        scheduler.step()
        print()
    
    print(f"Mejor precisión de validación: {best_acc:.4f}")
    return model

def main():
    print("🦷 Iniciando entrenamiento de clasificación para dental_dataset")
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"🚀 Usando GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ Usando CPU - el entrenamiento será más lento")
    
    # Usar rutas relativas desde el directorio de ejecución
    data_dir = "../datasets/classification"
    
    print(f"📁 Directorio de datos: {data_dir}")
    
    # Verificar que existe el directorio
    if not os.path.exists(data_dir):
        print(f"❌ Error: No se encontró el directorio de datos: {data_dir}")
        print("💡 Ejecute desde training/ y asegúrese de que existe:")
        print("   ../datasets/classification/{train,val}/")
        return
    
    batch_size = 32
    num_epochs = 50
    
    try:
        # Crear data loaders
        train_loader, val_loader, classes = create_data_loaders(data_dir, batch_size)
        print(f"Clases encontradas: {classes}")
        print(f"Número de clases: {len(classes)}")
        
        # Crear modelo
        model = create_model(len(classes))
        
        # Entrenar
        trained_model = train_model(model, train_loader, val_loader, num_epochs)
        
        print("✅ Entrenamiento completado")
        print(f"📁 Modelo guardado en: ./logs/dental_dataset_classification/")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        print("💡 Verifica que:")
        print("   - El dataset existe en la ruta especificada")
        print("   - Las carpetas train/ y val/ contienen subdirectorios de clases")
        print("   - Hay suficientes imágenes en cada clase")
        sys.exit(1)

if __name__ == "__main__":
    main()
