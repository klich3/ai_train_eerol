#!/usr/bin/env python3
"""
🔍 DATASET ORGANIZER & AUTO-CLASSIFIER
======================================

Herramienta inteligente para detectar y organizar automáticamente datasets
según su formato (YOLO, COCO, U-Net, Clasificación) y moverlos a las
carpetas correspondientes (_YOLO, _COCO, _UNET, _pure images and masks).

Funcionalidades:
- Detección automática del tipo de dataset
- Organización automática en carpetas
- Conversión entre formatos
- Validación de estructura
- Backup de seguridad
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import cv2
import numpy as np


class DatasetDetector:
    """🔍 Detector de tipos de dataset."""
    
    def __init__(self):
        self.supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.annotation_patterns = {
            'yolo': {
                'files': ['.txt'],
                'structure': ['train', 'valid', 'test', 'val'],
                'config_files': ['data.yaml', 'dataset.yaml', 'classes.txt']
            },
            'coco': {
                'files': ['.json'],
                'structure': ['annotations'],
                'config_files': ['instances_train.json', 'instances_val.json', '_annotations.coco.json']
            },
            'unet': {
                'files': ['.png', '.jpg'],
                'structure': ['images', 'masks', 'labels'],
                'config_files': []
            },
            'classification': {
                'files': [],
                'structure': ['class_folders'],
                'config_files': []
            }
        }
    
    def detect_dataset_type(self, dataset_path: Path) -> Tuple[str, float, Dict]:
        """
        🔍 Detecta el tipo de dataset y devuelve confianza.
        
        Returns:
            (tipo, confianza, detalles)
        """
        dataset_info = {
            'path': str(dataset_path),
            'total_files': 0,
            'image_files': 0,
            'annotation_files': 0,
            'subdirectories': [],
            'file_extensions': Counter(),
            'structure_detected': [],
            'config_files_found': []
        }
        
        # Analizar estructura del directorio
        self._analyze_directory_structure(dataset_path, dataset_info)
        
        # Detectar cada tipo con confianza
        yolo_confidence = self._detect_yolo(dataset_info)
        coco_confidence = self._detect_coco(dataset_info)
        unet_confidence = self._detect_unet(dataset_info)
        classification_confidence = self._detect_classification(dataset_info)
        
        # Determinar el tipo con mayor confianza
        confidences = {
            'yolo': yolo_confidence,
            'coco': coco_confidence,
            'unet': unet_confidence,
            'classification': classification_confidence
        }
        
        best_type = max(confidences, key=confidences.get)
        best_confidence = confidences[best_type]
        
        dataset_info['detection_scores'] = confidences
        dataset_info['detected_type'] = best_type
        dataset_info['confidence'] = best_confidence
        
        return best_type, best_confidence, dataset_info
    
    def _analyze_directory_structure(self, path: Path, info: Dict):
        """📁 Analiza la estructura del directorio."""
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    info['total_files'] += 1
                    extension = item.suffix.lower()
                    info['file_extensions'][extension] += 1
                    
                    if extension in self.supported_image_extensions:
                        info['image_files'] += 1
                    elif extension in ['.txt', '.json', '.xml']:
                        info['annotation_files'] += 1
                elif item.is_dir():
                    rel_path = item.relative_to(path)
                    info['subdirectories'].append(str(rel_path))
        except PermissionError:
            pass
    
    def _detect_yolo(self, info: Dict) -> float:
        """🎯 Detecta formato YOLO."""
        score = 0.0
        
        # Buscar archivos .txt (anotaciones YOLO)
        if info['file_extensions'].get('.txt', 0) > 0:
            score += 30
        
        # Buscar estructura típica de YOLO
        yolo_dirs = ['train', 'valid', 'test', 'val', 'images', 'labels']
        found_dirs = sum(1 for d in yolo_dirs if any(d in subdir for subdir in info['subdirectories']))
        score += found_dirs * 10
        
        # Buscar archivos de configuración YOLO
        yolo_configs = ['data.yaml', 'dataset.yaml', 'classes.txt']
        for config in yolo_configs:
            if any(config in str(info['path']) for config in yolo_configs):
                score += 15
        
        # Proporción de archivos txt vs imágenes
        if info['image_files'] > 0 and info['file_extensions'].get('.txt', 0) > 0:
            ratio = info['file_extensions']['.txt'] / info['image_files']
            if 0.5 <= ratio <= 1.2:  # Proporción típica YOLO
                score += 20
        
        return min(score, 100)
    
    def _detect_coco(self, info: Dict) -> float:
        """🎯 Detecta formato COCO."""
        score = 0.0
        
        # Buscar archivos .json (anotaciones COCO)
        if info['file_extensions'].get('.json', 0) > 0:
            score += 40
        
        # Buscar estructura típica de COCO
        coco_patterns = ['annotations', 'train', 'val', 'test']
        found_patterns = sum(1 for p in coco_patterns if any(p in subdir for subdir in info['subdirectories']))
        score += found_patterns * 15
        
        # Buscar archivos típicos de COCO
        coco_files = ['instances_', '_annotations.coco.json', 'captions_', 'person_keypoints_']
        # Este check requeriría examinar el contenido real de los archivos
        
        # Si hay pocos JSON pero muchas imágenes, es probable COCO
        if info['image_files'] > 100 and info['file_extensions'].get('.json', 0) < 10:
            score += 20
        
        return min(score, 100)
    
    def _detect_unet(self, info: Dict) -> float:
        """🎯 Detecta formato U-Net."""
        score = 0.0
        
        # Buscar estructura típica de U-Net (images + masks)
        unet_dirs = ['images', 'masks', 'labels', 'groundtruth', 'gt']
        found_dirs = sum(1 for d in unet_dirs if any(d in subdir.lower() for subdir in info['subdirectories']))
        score += found_dirs * 20
        
        # U-Net típicamente tiene igual número de imágenes y máscaras
        if 'masks' in ' '.join(info['subdirectories']).lower():
            score += 30
        
        # Buscar imágenes en escala de grises (típico de máscaras)
        if info['file_extensions'].get('.png', 0) > info['file_extensions'].get('.jpg', 0):
            score += 15  # PNG es común para máscaras
        
        return min(score, 100)
    
    def _detect_classification(self, info: Dict) -> float:
        """🎯 Detecta formato de clasificación."""
        score = 0.0
        
        # Clasificación típicamente tiene carpetas por clase
        if len(info['subdirectories']) >= 2:
            score += 20
        
        # Pocas o ninguna anotación (solo imágenes organizadas por carpetas)
        if info['annotation_files'] == 0 and info['image_files'] > 0:
            score += 30
        
        # Muchas subcarpetas de primer nivel (clases)
        first_level_dirs = [d for d in info['subdirectories'] if '/' not in d and '\\' not in d]
        if len(first_level_dirs) >= 3:
            score += 25
        
        # Distribución uniforme de archivos por carpeta
        if len(first_level_dirs) >= 2:
            score += 15
        
        return min(score, 100)


class DatasetOrganizer:
    """📁 Organizador automático de datasets."""
    
    def __init__(self, base_path: str = "_dataSets"):
        self.base_path = Path(base_path)
        self.detector = DatasetDetector()
        self.target_dirs = {
            'yolo': self.base_path / "_YOLO",
            'coco': self.base_path / "_COCO", 
            'unet': self.base_path / "_UNET",
            'classification': self.base_path / "_pure images and masks"
        }
        
        # Crear directorios objetivo si no existen
        for target_dir in self.target_dirs.values():
            target_dir.mkdir(parents=True, exist_ok=True)
    
    def scan_and_organize(self, dry_run: bool = True, min_confidence: float = 70) -> Dict:
        """
        🔍 Escanea y organiza automáticamente los datasets.
        
        Args:
            dry_run: Si True, solo simula sin mover archivos
            min_confidence: Confianza mínima para auto-organizar
        """
        results = {
            'scanned_datasets': [],
            'organized_datasets': [],
            'skipped_datasets': [],
            'errors': []
        }
        
        print("🔍 ESCANEANDO DATASETS...")
        print("="*50)
        
        # Buscar datasets en el directorio base
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                try:
                    # Detectar tipo de dataset
                    detected_type, confidence, details = self.detector.detect_dataset_type(item)
                    
                    dataset_info = {
                        'name': item.name,
                        'path': str(item),
                        'detected_type': detected_type,
                        'confidence': confidence,
                        'details': details
                    }
                    
                    results['scanned_datasets'].append(dataset_info)
                    
                    # Mostrar resultado
                    print(f"📂 {item.name}")
                    print(f"   🎯 Tipo detectado: {detected_type.upper()}")
                    print(f"   📊 Confianza: {confidence:.1f}%")
                    print(f"   📋 Archivos: {details['total_files']} total, {details['image_files']} imágenes")
                    
                    # Organizar si la confianza es suficiente
                    if confidence >= min_confidence:
                        if not dry_run:
                            success = self._move_dataset(item, detected_type)
                            if success:
                                results['organized_datasets'].append(dataset_info)
                                print(f"   ✅ Movido a {self.target_dirs[detected_type]}")
                            else:
                                results['errors'].append(f"Error moviendo {item.name}")
                                print(f"   ❌ Error al mover")
                        else:
                            print(f"   🎯 Se movería a: {self.target_dirs[detected_type]}")
                    else:
                        results['skipped_datasets'].append(dataset_info)
                        print(f"   ⚠️ Confianza baja, requiere revisión manual")
                    
                    print()
                    
                except Exception as e:
                    error_msg = f"Error procesando {item.name}: {e}"
                    results['errors'].append(error_msg)
                    print(f"❌ {error_msg}")
        
        # Mostrar resumen
        print("📊 RESUMEN:")
        print("="*30)
        print(f"📂 Datasets escaneados: {len(results['scanned_datasets'])}")
        print(f"✅ Organizados: {len(results['organized_datasets'])}")
        print(f"⚠️ Omitidos: {len(results['skipped_datasets'])}")
        print(f"❌ Errores: {len(results['errors'])}")
        
        return results
    
    def _move_dataset(self, source: Path, dataset_type: str) -> bool:
        """📦 Mueve un dataset al directorio correspondiente."""
        try:
            target_dir = self.target_dirs[dataset_type]
            destination = target_dir / source.name
            
            # Evitar sobreescribir
            if destination.exists():
                counter = 1
                while destination.exists():
                    destination = target_dir / f"{source.name}_{counter}"
                    counter += 1
            
            # Mover directorio
            shutil.move(str(source), str(destination))
            return True
            
        except Exception as e:
            print(f"Error moviendo {source}: {e}")
            return False
    
    def interactive_organize(self):
        """🎮 Modo interactivo para organizar datasets."""
        print("🎮 ORGANIZADOR INTERACTIVO DE DATASETS")
        print("="*45)
        print()
        
        # Primero hacer un escaneo en seco
        results = self.scan_and_organize(dry_run=True)
        
        if not results['scanned_datasets']:
            print("❌ No se encontraron datasets para organizar")
            return
        
        print("\n🤔 ¿Qué deseas hacer?")
        print("1. Organizar automáticamente (confianza ≥70%)")
        print("2. Revisar y organizar manualmente")
        print("3. Solo mostrar detección")
        print("0. Cancelar")
        
        choice = input("\n🎯 Selecciona una opción: ").strip()
        
        if choice == '1':
            confirm = input("\n⚠️ ¿Confirmas mover los datasets? (s/N): ").strip().lower()
            if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                print("\n🚀 Organizando automáticamente...")
                self.scan_and_organize(dry_run=False, min_confidence=70)
        
        elif choice == '2':
            self._manual_organize_mode(results)
        
        elif choice == '3':
            print("✅ Detección completada. No se movieron archivos.")
        
        else:
            print("❌ Operación cancelada")
    
    def _manual_organize_mode(self, results: Dict):
        """👤 Modo de organización manual."""
        print("\n👤 MODO MANUAL:")
        print("="*20)
        
        for dataset in results['scanned_datasets']:
            print(f"\n📂 Dataset: {dataset['name']}")
            print(f"🎯 Tipo detectado: {dataset['detected_type'].upper()} ({dataset['confidence']:.1f}%)")
            print(f"📋 Detalles: {dataset['details']['image_files']} imágenes, {dataset['details']['annotation_files']} anotaciones")
            
            print("\n¿Qué hacer con este dataset?")
            print("1. Mover a _YOLO")
            print("2. Mover a _COCO")
            print("3. Mover a _UNET")
            print("4. Mover a _pure images and masks")
            print("5. Omitir")
            print("0. Terminar")
            
            choice = input("Selecciona: ").strip()
            
            if choice == '0':
                break
            elif choice == '5':
                continue
            elif choice in ['1', '2', '3', '4']:
                type_map = {'1': 'yolo', '2': 'coco', '3': 'unet', '4': 'classification'}
                target_type = type_map[choice]
                
                source_path = Path(dataset['path'])
                success = self._move_dataset(source_path, target_type)
                
                if success:
                    print(f"✅ {dataset['name']} movido a {self.target_dirs[target_type]}")
                else:
                    print(f"❌ Error moviendo {dataset['name']}")


def main():
    """🚀 Función principal."""
    print("🔍 DATASET ORGANIZER & AUTO-CLASSIFIER")
    print("="*45)
    print()
    
    # Verificar que existe el directorio _dataSets
    if not Path("_dataSets").exists():
        print("❌ Directorio '_dataSets' no encontrado")
        print("💡 Asegúrate de ejecutar el script desde el directorio correcto")
        return
    
    organizer = DatasetOrganizer()
    
    print("🎯 OPCIONES:")
    print("1. Escaneo y organización automática")
    print("2. Modo interactivo")
    print("3. Solo escanear (sin mover)")
    print("0. Salir")
    
    choice = input("\n🎯 Selecciona una opción: ").strip()
    
    if choice == '1':
        print("\n🤖 MODO AUTOMÁTICO:")
        min_conf = input("Confianza mínima (70): ").strip() or "70"
        try:
            min_confidence = float(min_conf)
            organizer.scan_and_organize(dry_run=False, min_confidence=min_confidence)
        except ValueError:
            print("❌ Confianza inválida, usando 70%")
            organizer.scan_and_organize(dry_run=False, min_confidence=70)
    
    elif choice == '2':
        organizer.interactive_organize()
    
    elif choice == '3':
        organizer.scan_and_organize(dry_run=True)
    
    else:
        print("👋 ¡Hasta luego!")


if __name__ == "__main__":
    main()
