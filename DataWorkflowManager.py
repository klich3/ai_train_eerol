"""
█▀ █▄█ █▀▀ █░█ █▀▀ █░█
▄█ ░█░ █▄▄ █▀█ ██▄ ▀▄▀

Author: <Anton Sychev> (anton at sychev dot xyz)
DataWorkflowManager.py (c) 2025
Created:  2025-07-05 18:30:00 
Desc: Workflow Manager para unificación, balanceo y preparación de datasets dentales
Docs: Implementa estrategia completa de mezcla de datos para entrenar modelos robustos
"""

import os
import json
import yaml
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class DentalDataWorkflowManager:
    """
    🎛️ Gestor de flujo de trabajo para datasets dentales
    Implementa unificación, balanceo y preparación de datos para entrenamiento robusto
    """
    
    def __init__(self, base_path: str = None, output_path: str = None):
        # Usar rutas relativas desde donde se ejecuta el script
        self.base_path = Path(base_path) if base_path else Path("_dataSets")
        
        # Estructura dental-ai organizada
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = Path("dental-ai")
        
        # Archivo de análisis relativo
        self.analysis_file = Path("dental_dataset_analysis.json")
        
        # Configuración de la estructura dental-ai
        self.dental_ai_structure = {
            'datasets': {
                'detection_combined': 'YOLO format fusionados',
                'segmentation_coco': 'COCO format unificado', 
                'segmentation_bitmap': 'Máscaras para U-Net',
                'classification': 'Clasificación por carpetas'
            },
            'models': {
                'yolo_detect': 'Modelos YOLO detección',
                'yolo_segment': 'Modelos YOLO segmentación',
                'unet_teeth': 'Modelos U-Net dientes',
                'cnn_classifier': 'Clasificadores CNN'
            },
            'training': {
                'scripts': 'Scripts de entrenamiento',
                'configs': 'Configuraciones de entrenamiento',
                'logs': 'Logs de entrenamiento'
            },
            'api': {
                'main.py': 'API principal',
                'models': 'Modelos para la API',
                'utils': 'Utilidades'
            },
            'docs': 'Documentación'
        }
        
        # Logging configuration
        self.log_entries = []
        
        # Configuración de seguridad
        self.safety_config = {
            'backup_enabled': True,
            'read_only_source': True,
            'verify_copy': True,
            'preserve_original_structure': True
        }
        
        # Configuración de formatos objetivo
        self.target_formats = {
            'detection': 'YOLO',
            'segmentation': 'COCO',
            'classification': 'folders'
        }
        
        # Configuración de resoluciones estándar
        self.standard_resolutions = {
            'yolo': (640, 640),
            'coco': (1024, 1024),
            'unet': (512, 512)
        }
        
        # Mapeo de clases unificadas
        self.unified_classes = {
            'caries': ['caries', 'Caries', 'CARIES', 'cavity', 'decay', 'Q1_Caries', 'Q2_Caries', 'Q3_Caries', 'Q4_Caries'],
            'tooth': ['tooth', 'teeth', 'Tooth', 'TOOTH', 'diente', 'molar', 'premolar', 'canine', 'incisor'],
            'filling': ['filling', 'Filling', 'Fillings', 'FILLING', 'restoration', 'RESTORATION'],
            'crown': ['crown', 'Crown', 'CROWN', 'CROWN AND BRIDGE'],
            'implant': ['implant', 'Implant', 'IMPLANT'],
            'root_canal': ['Root Canal Treatment', 'ROOT CANAL TREATED TOOTH', 'root canal'],
            'bone_loss': ['Bone Loss', 'BONE LOSS', 'VERTICAL BONE LOSS'],
            'impacted': ['impacted', 'Impacted', 'IMPACTED TOOTH', 'Q1_Impacted', 'Q2_Impacted', 'Q3_Impacted', 'Q4_Impacted'],
            'periapical_lesion': ['Periapical lesion', 'Q1_Periapical_Lesion', 'Q2_Periapical_Lesion', 'Q3_Periapical_Lesion', 'Q4_Periapical_Lesion'],
            'maxillary_sinus': ['maxillary sinus', 'MAXILLARY SINUS', 'MAXILLARY  SINUS'],
            'mandible': ['Mandible', 'mandible', 'RAMUS OF MANDIBLE', 'INFERIOR BORDER OF MANDIBLE'],
            'maxilla': ['Maxilla', 'maxilla']
        }
        
        self.workflow_config = {
            'train_ratio': 0.7,
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            'min_samples_per_class': 10,
            'max_augmentation_factor': 5,
            'class_balance_threshold': 0.1  # Si una clase tiene menos del 10% del promedio, se balancea
        }
        
        # Crear directorio de salida
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def log_message(self, message: str, file_path: str = None):
        """Función de logging con timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        self.log_entries.append(formatted_message)
        
        if file_path:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(formatted_message + "\n")
    
    def create_dental_ai_structure(self):
        """🏗️ Crea la estructura completa de dental-ai si no existe."""
        print("\n🏗️ INICIALIZANDO ESTRUCTURA DENTAL-AI...")
        
        structure_created = []
        
        # Crear directorios principales
        for main_dir, content in self.dental_ai_structure.items():
            main_path = self.output_path / main_dir
            main_path.mkdir(parents=True, exist_ok=True)
            structure_created.append(str(main_path))
            
            if isinstance(content, dict):
                for sub_dir, description in content.items():
                    sub_path = main_path / sub_dir
                    sub_path.mkdir(parents=True, exist_ok=True)
                    structure_created.append(str(sub_path))
                    
                    # Crear archivo README para cada subdirectorio
                    readme_path = sub_path / 'README.md'
                    if not readme_path.exists():
                        with open(readme_path, 'w', encoding='utf-8') as f:
                            f.write(f"# {sub_dir.replace('_', ' ').title()}\n\n")
                            f.write(f"{description}\n\n")
                            f.write(f"Creado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            elif main_dir == 'training':
                # Crear subdirectorios específicos de training
                for sub_dir in ['scripts', 'configs', 'logs']:
                    sub_path = main_path / sub_dir
                    sub_path.mkdir(parents=True, exist_ok=True)
                    structure_created.append(str(sub_path))
        
        # Crear archivo principal de documentación
        self._create_main_documentation()
        
        # Crear requirements.txt principal
        self._create_requirements_file()
        
        # Crear archivo de configuración principal
        self._create_main_config()
        
        print(f"✅ Estructura dental-ai creada con {len(structure_created)} directorios")
        return structure_created
    
    def _create_main_documentation(self):
        """Crea documentación principal del proyecto dental-ai."""
        readme_path = self.output_path / 'README.md'
        
        readme_content = f"""# 🦷 Dental AI - Sistema de Análisis Dental con IA

Proyecto completo para análisis dental usando deep learning, generado automáticamente por DataWorkflowManager.

## 📁 Estructura del Proyecto

```
dental-ai/
├── datasets/           # Datasets procesados y listos para entrenamiento
│   ├── detection_combined/     # Datasets YOLO fusionados para detección
│   ├── segmentation_coco/      # Datasets COCO para segmentación
│   ├── segmentation_bitmap/    # Máscaras para U-Net
│   └── classification/         # Datasets para clasificación
├── models/             # Modelos entrenados
│   ├── yolo_detect/           # Modelos YOLO para detección
│   ├── yolo_segment/          # Modelos YOLO para segmentación
│   ├── unet_teeth/            # Modelos U-Net para dientes
│   └── cnn_classifier/        # Clasificadores CNN
├── training/           # Scripts y configuraciones de entrenamiento
│   ├── scripts/               # Scripts de entrenamiento automatizados
│   ├── configs/               # Configuraciones específicas
│   └── logs/                  # Logs de entrenamiento
├── api/                # API REST para inferencia
├── docs/               # Documentación adicional
└── README.md          # Este archivo
```

## 🚀 Inicio Rápido

### 1. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 2. Entrenamiento de Modelos
```bash
cd training/scripts
bash train_[nombre_dataset].sh
```

### 3. Uso de la API
```bash
cd api
python main.py
```

## 📊 Datasets Disponibles

Los datasets están organizados por tipo de tarea:

- **Detección**: Datasets YOLO para detectar estructuras dentales
- **Segmentación**: Datasets COCO y máscaras bitmap para segmentación precisa
- **Clasificación**: Datasets organizados por carpetas para clasificación de patologías

## 🔧 Configuración

Todos los parámetros de entrenamiento están en `training/configs/`.

## 📝 Logs y Monitoreo

Los logs de entrenamiento se guardan en `training/logs/` con timestamps.

## 🛡️ Protección de Datos

Este proyecto utiliza un sistema de seguridad que:
- ✅ NUNCA modifica los datos originales
- ✅ Crea copias de solo lectura
- ✅ Verifica la integridad de los archivos copiados
- ✅ Mantiene logs completos de todas las operaciones

## 📈 Desarrollo

Generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} por DataWorkflowManager.

Para regenerar o actualizar datasets, utiliza el DataWorkflowManager en el directorio padre.
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _create_requirements_file(self):
        """Crea archivo de requirements para el proyecto dental-ai."""
        req_path = self.output_path / 'requirements.txt'
        
        requirements = [
            "# 🦷 Dental AI Requirements",
            "# Core ML",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "ultralytics>=8.0.0",
            "detectron2",
            "",
            "# Computer Vision",
            "opencv-python>=4.8.0",
            "Pillow>=9.5.0",
            "albumentations>=1.3.0",
            "",
            "# Data Science",
            "numpy>=1.24.0",
            "pandas>=2.0.0", 
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "scikit-learn>=1.3.0",
            "",
            "# API",
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.22.0",
            "pydantic>=2.0.0",
            "",
            "# Utils",
            "tqdm>=4.65.0",
            "pyyaml>=6.0",
            "requests>=2.31.0",
            "",
            "# Development",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0"
        ]
        
        with open(req_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(requirements))
    
    def _create_main_config(self):
        """Crea configuración principal del proyecto."""
        config_path = self.output_path / 'config.yaml'
        
        config = {
            'project': {
                'name': 'dental-ai',
                'version': '1.0.0',
                'created': datetime.now().isoformat(),
                'description': 'Sistema de análisis dental con IA'
            },
            'training': {
                'default_epochs': 100,
                'default_batch_size': 16,
                'default_img_size': 640,
                'patience': 20,
                'save_period': 10
            },
            'paths': {
                'datasets': 'datasets/',
                'models': 'models/',
                'training': 'training/',
                'api': 'api/',
                'logs': 'training/logs/'
            },
            'safety': {
                'preserve_originals': True,
                'verify_copies': True,
                'create_backups': True,
                'read_only_source': True
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def safe_copy_file(self, src: Path, dst: Path, verify: bool = True) -> bool:
        """Copia un archivo de forma segura con verificaciones."""
        try:
            # Verificar que existe el archivo fuente
            if not src.exists():
                self.log_message(f"❌ Archivo fuente no existe: {src}")
                return False
            
            # Verificar permisos de lectura
            if not os.access(src, os.R_OK):
                self.log_message(f"❌ Sin permisos de lectura: {src}")
                return False
            
            # Crear directorio padre si no existe
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Copiar archivo
            shutil.copy2(src, dst)
            
            # Hacer de solo lectura (protección adicional)
            if self.safety_config['read_only_source']:
                dst.chmod(0o444)  # Solo lectura
            
            # Verificar integridad si se solicita
            if verify and self.safety_config['verify_copy']:
                if not self._verify_file_integrity(src, dst):
                    self.log_message(f"❌ Verificación de integridad falló: {src} -> {dst}")
                    return False
            
            return True
            
        except Exception as e:
            self.log_message(f"❌ Error copiando {src} -> {dst}: {e}")
            return False
    
    def _verify_file_integrity(self, src: Path, dst: Path) -> bool:
        """Verifica la integridad de un archivo copiado."""
        try:
            # Comparar tamaños
            if src.stat().st_size != dst.stat().st_size:
                return False
            
            # Comparar hash MD5 para archivos pequeños (< 100MB)
            if src.stat().st_size < 100 * 1024 * 1024:
                import hashlib
                
                with open(src, 'rb') as f1, open(dst, 'rb') as f2:
                    hash1 = hashlib.md5(f1.read()).hexdigest()
                    hash2 = hashlib.md5(f2.read()).hexdigest()
                    return hash1 == hash2
            
            return True  # Para archivos grandes, solo verificar tamaño
            
        except Exception:
            return False
    
    def load_analysis_data(self) -> Dict[str, Any]:
        """Carga los datos del análisis previo."""
        try:
            with open(self.analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: No se encontró {self.analysis_file}")
            print("Ejecuta primero el script de análisis")
            return {}
    
    def unify_class_names(self, original_name: str) -> str:
        """Unifica nombres de clases similares."""
        original_lower = original_name.lower().strip()
        
        for unified_name, variations in self.unified_classes.items():
            if any(var.lower() in original_lower for var in variations):
                return unified_name
        
        # Si no se encuentra, devolver el nombre limpio
        return original_name.replace(' ', '_').lower()
    
    def analyze_class_distribution(self, data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """🧱 1. Análisis estadístico de clases por tipo de dataset."""
        print("\n📊 ANALIZANDO DISTRIBUCIÓN DE CLASES...")
        
        class_distribution = {
            'YOLO': defaultdict(int),
            'COCO': defaultdict(int),
            'UNET': defaultdict(int),
            'Unknown': defaultdict(int)
        }
        
        total_samples = defaultdict(int)
        
        for dataset_type, dataset_info in data.items():
            if not dataset_type.startswith('_'):
                continue
                
            format_type = self._get_format_type(dataset_info.get('type', ''))
            
            for dataset_name, dataset_data in dataset_info.get('datasets', {}).items():
                class_dist = dataset_data.get('class_distribution', {})
                
                for class_name, count in class_dist.items():
                    unified_name = self.unify_class_names(class_name)
                    class_distribution[format_type][unified_name] += count
                    total_samples[unified_name] += count
        
        # Mostrar estadísticas
        print("\n📈 DISTRIBUCIÓN DE CLASES UNIFICADAS:")
        for format_type, classes in class_distribution.items():
            if classes:
                print(f"\n🔹 {format_type}:")
                sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
                for class_name, count in sorted_classes[:10]:  # Top 10
                    print(f"   • {class_name}: {count:,} muestras")
        
        return dict(class_distribution), dict(total_samples)
    
    def _get_format_type(self, type_string: str) -> str:
        """Determina el tipo de formato basándose en la descripción."""
        if 'YOLO' in type_string:
            return 'YOLO'
        elif 'COCO' in type_string:
            return 'COCO'
        elif 'U-Net' in type_string:
            return 'UNET'
        else:
            return 'Unknown'
    
    def identify_imbalanced_classes(self, total_samples: Dict[str, int]) -> Dict[str, str]:
        """Identifica clases desbalanceadas que necesitan augmentación."""
        if not total_samples:
            return {}
        
        avg_samples = np.mean(list(total_samples.values()))
        threshold = avg_samples * self.workflow_config['class_balance_threshold']
        
        imbalanced_classes = {}
        
        print(f"\n⚖️ ANÁLISIS DE BALANCE DE CLASES (Promedio: {avg_samples:.0f} muestras)")
        print(f"🚨 Umbral crítico: {threshold:.0f} muestras")
        
        for class_name, count in sorted(total_samples.items(), key=lambda x: x[1]):
            if count < threshold:
                factor_needed = int(avg_samples / count) if count > 0 else self.workflow_config['max_augmentation_factor']
                factor_needed = min(factor_needed, self.workflow_config['max_augmentation_factor'])
                imbalanced_classes[class_name] = f"Necesita {factor_needed}x augmentación"
                print(f"   🔴 {class_name}: {count} muestras → Necesita {factor_needed}x")
            elif count < avg_samples * 0.5:
                imbalanced_classes[class_name] = "Ligeramente desbalanceada"
                print(f"   🟡 {class_name}: {count} muestras → Ligeramente baja")
            else:
                print(f"   🟢 {class_name}: {count} muestras → Bien balanceada")
        
        return imbalanced_classes
    
    def recommend_dataset_fusion(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """🔄 3. Recomienda fusión de datasets similares."""
        print("\n🔗 RECOMENDACIONES DE FUSIÓN DE DATASETS...")
        
        fusion_groups = {
            'dental_detection_panoramic': [],
            'dental_detection_periapical': [],
            'dental_segmentation_coco': [],
            'dental_classification': []
        }
        
        for dataset_type, dataset_info in data.items():
            if not dataset_type.startswith('_'):
                continue
                
            format_type = self._get_format_type(dataset_info.get('type', ''))
            
            for dataset_name, dataset_data in dataset_info.get('datasets', {}).items():
                categories = [cat.lower() for cat in dataset_data.get('categories', [])]
                
                # Clasificar por tipo y contenido
                if format_type == 'YOLO':
                    if any('panoramic' in cat for cat in categories) or 'panoramic' in dataset_name.lower():
                        fusion_groups['dental_detection_panoramic'].append(dataset_name)
                    elif any('periapical' in cat for cat in categories) or 'periapical' in dataset_name.lower():
                        fusion_groups['dental_detection_periapical'].append(dataset_name)
                    else:
                        fusion_groups['dental_detection_panoramic'].append(dataset_name)  # Default a panoramic
                
                elif format_type == 'COCO':
                    fusion_groups['dental_segmentation_coco'].append(dataset_name)
                
                elif format_type == 'Unknown' and any('classification' in cat for cat in categories):
                    fusion_groups['dental_classification'].append(dataset_name)
        
        # Mostrar recomendaciones
        print("\n💡 GRUPOS DE FUSIÓN RECOMENDADOS:")
        for group_name, datasets in fusion_groups.items():
            if datasets:
                print(f"\n🔹 {group_name.replace('_', ' ').title()}:")
                for dataset in datasets[:5]:  # Mostrar top 5
                    print(f"   • {dataset}")
                if len(datasets) > 5:
                    print(f"   ... y {len(datasets) - 5} más")
        
        return fusion_groups
    
    def create_unified_yolo_dataset(self, data: Dict[str, Any], datasets_to_merge: List[str], 
                                   output_name: str = "panoramic_detection", 
                                   target_type: str = "detection_combined") -> str:
        """🧱 Crea dataset YOLO unificado en la estructura dental-ai."""
        print(f"\n🔨 CREANDO DATASET YOLO EN DENTAL-AI: {output_name}")
        print("🛡️ MODO SEGURO: Los datos originales NO serán modificados")
        
        # Crear estructura dental-ai si no existe
        self.create_dental_ai_structure()
        
        # Directorio de destino en dental-ai
        dataset_dir = self.output_path / 'datasets' / target_type / output_name
        
        # Crear estructura YOLO estándar
        for split in ['train', 'val', 'test']:
            (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Crear directorio de logs
        log_dir = self.output_path / 'training' / 'logs' / f'{output_name}_creation'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        unified_classes = set()
        all_images = []
        processing_stats = {
            'datasets_processed': 0,
            'images_found': 0,
            'images_copied': 0,
            'labels_processed': 0,
            'errors': [],
            'skipped_files': []
        }
        
        # Log de procesamiento
        log_file = log_dir / f'{output_name}_processing.log'
        
        def log_message(message: str):
            print(f"   {message}")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - {message}\n")
        
        log_message("🚀 Iniciando creación de dataset en dental-ai")
        log_message(f"📂 Origen: {self.base_path}")
        log_message(f"📁 Destino: {dataset_dir}")
        
        # Recopilar todas las imágenes y anotaciones (SOLO LECTURA)
        for dataset_type, dataset_info in data.items():
            if not dataset_type.startswith('_'):
                continue
                
            format_type = self._get_format_type(dataset_info.get('type', ''))
            if format_type != 'YOLO':
                continue
            
            for dataset_name, dataset_data in dataset_info.get('datasets', {}).items():
                if dataset_name not in datasets_to_merge:
                    continue
                
                log_message(f"📂 Procesando dataset: {dataset_name}")
                dataset_path = Path(dataset_data['path'])
                
                # Verificar accesibilidad del dataset
                if not dataset_path.exists():
                    error_msg = f"❌ Dataset no encontrado: {dataset_path}"
                    log_message(error_msg)
                    processing_stats['errors'].append(error_msg)
                    continue
                
                if not os.access(dataset_path, os.R_OK):
                    error_msg = f"❌ Sin permisos de lectura: {dataset_path}"
                    log_message(error_msg)
                    processing_stats['errors'].append(error_msg)
                    continue
                
                # Buscar imágenes (SOLO LECTURA)
                image_files = []
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    found_files = list(dataset_path.rglob(f'*{ext}'))
                    image_files.extend(found_files)
                    log_message(f"   🖼️ Encontradas {len(found_files)} imágenes {ext}")
                
                processing_stats['images_found'] += len(image_files)
                
                # Obtener clases del dataset
                classes = dataset_data.get('classes', [])
                if classes:
                    unified_classes.update([self.unify_class_names(cls) for cls in classes])
                    log_message(f"   🏷️ Clases: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
                
                # Procesar cada imagen
                for img_file in image_files:
                    label_file = img_file.with_suffix('.txt')
                    
                    if label_file.exists() and os.access(label_file, os.R_OK):
                        all_images.append({
                            'image_path': img_file,
                            'label_path': label_file,
                            'dataset': dataset_name,
                            'classes': classes,
                            'original_dataset_path': dataset_path
                        })
                    else:
                        processing_stats['skipped_files'].append(str(img_file))
                
                processing_stats['datasets_processed'] += 1
        
        if not all_images:
            error_msg = "❌ No se encontraron imágenes válidas con anotaciones"
            log_message(error_msg)
            print(error_msg)
            return str(dataset_dir)
        
        log_message(f"📊 Total de imágenes a procesar: {len(all_images):,}")
        log_message(f"🏷️ Clases unificadas encontradas: {len(unified_classes)}")
        
        # Crear mapeo de clases unificado
        class_mapping = {cls: idx for idx, cls in enumerate(sorted(unified_classes))}
        
        # Guardar archivo de clases
        classes_file = dataset_dir / 'classes.txt'
        with open(classes_file, 'w', encoding='utf-8') as f:
            for cls in sorted(unified_classes):
                f.write(f"{cls}\n")
        
        log_message(f"💾 Archivo de clases guardado: {classes_file}")
        
        # Crear data.yaml compatible con YOLO
        yaml_data = {
            'path': str(dataset_dir.resolve()),  # Ruta absoluta para YOLO
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': len(unified_classes),
            'names': list(sorted(unified_classes)),
            
            # Metadatos adicionales
            'source_datasets': datasets_to_merge,
            'created': datetime.now().isoformat(),
            'total_images': len(all_images),
            'project': 'dental-ai',
            'task': 'detection'
        }
        
        yaml_file = dataset_dir / 'data.yaml'
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        
        log_message(f"📝 Configuración YAML guardada: {yaml_file}")
        
        # División estratificada
        random.seed(42)
        random.shuffle(all_images)
        
        total_images = len(all_images)
        train_count = int(total_images * self.workflow_config['train_ratio'])
        val_count = int(total_images * self.workflow_config['val_ratio'])
        
        train_imgs = all_images[:train_count]
        val_imgs = all_images[train_count:train_count + val_count]
        test_imgs = all_images[train_count + val_count:]
        
        # Copiar archivos de forma segura
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        for split_name, images in splits.items():
            log_message(f"📁 Copiando {len(images)} imágenes a {split_name}")
            
            for i, img_data in enumerate(tqdm(images, desc=f"Copiando {split_name}")):
                try:
                    # Nombres únicos para evitar conflictos
                    img_ext = img_data['image_path'].suffix
                    new_img_name = f"{img_data['dataset']}_{i:06d}{img_ext}"
                    new_label_name = f"{img_data['dataset']}_{i:06d}.txt"
                    
                    # Rutas de destino
                    new_img_path = dataset_dir / split_name / 'images' / new_img_name
                    new_label_path = dataset_dir / split_name / 'labels' / new_label_name
                    
                    # Copiar imagen de forma segura
                    if self.safe_copy_file(img_data['image_path'], new_img_path):
                        processing_stats['images_copied'] += 1
                    else:
                        error_msg = f"Error copiando imagen: {img_data['image_path']}"
                        processing_stats['errors'].append(error_msg)
                        continue
                    
                    # Procesar etiquetas de forma segura
                    if self._convert_yolo_labels_safe(
                        img_data['label_path'], 
                        new_label_path,
                        img_data['classes'], 
                        class_mapping,
                        log_message
                    ):
                        processing_stats['labels_processed'] += 1
                    else:
                        error_msg = f"Error procesando etiqueta: {img_data['label_path']}"
                        processing_stats['errors'].append(error_msg)
                
                except Exception as e:
                    error_msg = f"Error procesando {img_data['image_path']}: {e}"
                    log_message(f"❌ {error_msg}")
                    processing_stats['errors'].append(error_msg)
        
        # Guardar estadísticas
        stats_file = log_dir / f'{output_name}_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(processing_stats, f, indent=2, ensure_ascii=False, default=str)
        
        # Crear script de entrenamiento específico
        self._create_training_script_for_dataset(output_name, target_type)
        
        # Resumen final
        log_message("✅ DATASET CREADO EN DENTAL-AI")
        log_message(f"📊 Estadísticas finales:")
        log_message(f"   • Datasets procesados: {processing_stats['datasets_processed']}")
        log_message(f"   • Imágenes copiadas: {processing_stats['images_copied']}")
        log_message(f"   • Etiquetas procesadas: {processing_stats['labels_processed']}")
        log_message(f"   • Errores: {len(processing_stats['errors'])}")
        
        print(f"\n✅ Dataset YOLO creado en dental-ai: {dataset_dir}")
        print(f"📊 Estadísticas: {stats_file}")
        print(f"📋 Log completo: {log_file}")
        print(f"🛡️ Datos originales intactos en: {self.base_path}")
        print(f"\n🚀 Para entrenar:")
        print(f"   cd dental-ai/training")
        print(f"   bash train_{output_name}.sh")
        
        if processing_stats['errors']:
            print(f"⚠️ Se encontraron {len(processing_stats['errors'])} errores. Revisa el log.")
        
        return str(dataset_dir)
    
    def _create_training_script_for_dataset(self, dataset_name: str, target_type: str):
        """Crea script de entrenamiento específico para el dataset."""
        training_dir = self.output_path / 'training'
        script_file = training_dir / f'train_{dataset_name}.sh'
        
        script_content = f"""#!/bin/bash
# 🦷 Entrenamiento específico para {dataset_name}

echo "🦷 Iniciando entrenamiento para {dataset_name}..."

# Configuración específica
MODEL="yolov8n.pt"  # Cambiar según necesidades
DATA="../datasets/{target_type}/{dataset_name}/data.yaml"
EPOCHS=100
BATCH=16
IMG_SIZE=640

# Crear directorio de salida
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./logs/{dataset_name}_${{TIMESTAMP}}"
mkdir -p $OUTPUT_DIR

echo "📁 Guardando en: $OUTPUT_DIR"
echo "📊 Dataset: $DATA"

# Verificar que existe el dataset
if [ ! -f "$DATA" ]; then
    echo "❌ Error: No se encontró $DATA"
    exit 1
fi

# Entrenamiento YOLO
yolo detect train \\
    model=$MODEL \\
    data=$DATA \\
    epochs=$EPOCHS \\
    batch=$BATCH \\
    imgsz=$IMG_SIZE \\
    project=$OUTPUT_DIR \\
    name="{dataset_name}" \\
    save_period=10 \\
    patience=20 \\
    device=0 \\
    workers=8 \\
    cache=True

echo "✅ Entrenamiento completado"
echo "📁 Modelo guardado en: $OUTPUT_DIR/{dataset_name}/weights/"
echo "📊 Métricas en: $OUTPUT_DIR/{dataset_name}/"

# Copiar mejor modelo a directorio de modelos
cp "$OUTPUT_DIR/{dataset_name}/weights/best.pt" "../models/yolo_detect/{dataset_name}_best.pt"
echo "📦 Modelo copiado a: ../models/yolo_detect/{dataset_name}_best.pt"
"""
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Hacer ejecutable
        import stat
        script_file.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        print(f"📝 Script de entrenamiento creado: {script_file}")
    
    def _convert_yolo_labels_safe(self, src_label: Path, dst_label: Path, 
                                 original_classes: List[str], class_mapping: Dict[str, int],
                                 log_function) -> bool:
        """Convierte etiquetas YOLO de forma segura con verificación completa."""
        try:
            # Verificar acceso de lectura al archivo fuente
            if not os.access(src_label, os.R_OK):
                log_function(f"❌ Sin permisos de lectura: {src_label}")
                return False
            
            # Leer archivo original (SOLO LECTURA)
            with open(src_label, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                log_function(f"⚠️ Archivo de etiquetas vacío: {src_label}")
                # Crear archivo vacío en destino
                dst_label.touch()
                return True
            
            converted_lines = []
            conversion_stats = {'converted': 0, 'skipped': 0, 'errors': 0}
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    log_function(f"⚠️ Línea malformada {line_num} en {src_label}: {line}")
                    conversion_stats['errors'] += 1
                    continue
                
                try:
                    class_id = int(parts[0])
                    
                    # Verificar que el class_id es válido
                    if class_id < 0 or class_id >= len(original_classes):
                        log_function(f"⚠️ ID de clase inválido {class_id} en línea {line_num} de {src_label}")
                        conversion_stats['errors'] += 1
                        continue
                    
                    # Convertir coordenadas y verificar rangos
                    try:
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Verificar que las coordenadas están en rango válido [0, 1]
                        if not all(0 <= coord <= 1 for coord in [x_center, y_center, width, height]):
                            log_function(f"⚠️ Coordenadas fuera de rango en línea {line_num} de {src_label}")
                            conversion_stats['errors'] += 1
                            continue
                            
                    except ValueError:
                        log_function(f"⚠️ Coordenadas inválidas en línea {line_num} de {src_label}: {parts[1:5]}")
                        conversion_stats['errors'] += 1
                        continue
                    
                    # Mapear clase a nombre unificado
                    original_class = original_classes[class_id]
                    unified_class = self.unify_class_names(original_class)
                    new_class_id = class_mapping.get(unified_class, 0)
                    
                    # Crear nueva línea con clase mapeada
                    new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                    converted_lines.append(new_line)
                    conversion_stats['converted'] += 1
                    
                except ValueError:
                    log_function(f"⚠️ ID de clase no numérico en línea {line_num} de {src_label}: {parts[0]}")
                    conversion_stats['errors'] += 1
                    continue
            
            # Crear directorio destino si no existe
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            
            # Escribir archivo convertido
            with open(dst_label, 'w', encoding='utf-8') as f:
                f.writelines(converted_lines)
            
            # Establecer permisos de solo lectura
            os.chmod(dst_label, 0o444)
            
            # Log de estadísticas de conversión
            if conversion_stats['errors'] > 0:
                log_function(f"⚠️ {src_label.name}: {conversion_stats['converted']} convertidas, {conversion_stats['errors']} errores")
            
            return conversion_stats['converted'] > 0 or len(lines) == 0
            
        except Exception as e:
            log_function(f"❌ Error procesando etiquetas {src_label}: {e}")
            return False
    
    def create_unified_coco_dataset(self, data: Dict[str, Any], datasets_to_merge: List[str], 
                                   output_name: str = "dental_segmentation", 
                                   target_type: str = "segmentation_coco") -> str:
        """🧱 Crea dataset COCO unificado para segmentación en la estructura dental-ai."""
        print(f"\n🔨 CREANDO DATASET COCO EN DENTAL-AI: {output_name}")
        print("🛡️ MODO SEGURO: Los datos originales NO serán modificados")
        
        # Crear estructura dental-ai si no existe
        self.create_dental_ai_structure()
        
        # Directorio de destino en dental-ai
        dataset_dir = self.output_path / 'datasets' / target_type / output_name
        
        # Crear estructura COCO estándar
        for split in ['train', 'val', 'test']:
            (dataset_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Crear directorio de logs
        log_dir = self.output_path / 'training' / 'logs' / f'{output_name}_creation'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar estructura COCO
        coco_annotations = {
            'train': {'images': [], 'annotations': [], 'categories': []},
            'val': {'images': [], 'annotations': [], 'categories': []},
            'test': {'images': [], 'annotations': [], 'categories': []}
        }
        
        unified_categories = {}
        all_images = []
        processing_stats = {
            'datasets_processed': 0,
            'images_found': 0,
            'images_copied': 0,
            'annotations_processed': 0,
            'errors': []
        }
        
        log_file = log_dir / f'{output_name}_processing.log'
        
        def log_message(message: str):
            print(f"   {message}")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - {message}\n")
        
        log_message("🚀 Iniciando creación de dataset COCO en dental-ai")
        log_message(f"📂 Origen: {self.base_path}")
        log_message(f"📁 Destino: {dataset_dir}")
        
        # Procesar datasets COCO
        for dataset_type, dataset_info in data.items():
            if not dataset_type.startswith('_'):
                continue
                
            format_type = self._get_format_type(dataset_info.get('type', ''))
            if format_type != 'COCO':
                continue
            
            for dataset_name, dataset_data in dataset_info.get('datasets', {}).items():
                if dataset_name not in datasets_to_merge:
                    continue
                
                log_message(f"📂 Procesando dataset COCO: {dataset_name}")
                dataset_path = Path(dataset_data['path'])
                
                # Buscar archivo de anotaciones COCO
                annotation_files = list(dataset_path.rglob('*.json'))
                if not annotation_files:
                    log_message(f"❌ No se encontraron archivos JSON COCO en {dataset_path}")
                    processing_stats['errors'].append(f"Sin anotaciones COCO: {dataset_name}")
                    continue
                
                # Procesar cada archivo de anotaciones
                for ann_file in annotation_files:
                    try:
                        with open(ann_file, 'r', encoding='utf-8') as f:
                            coco_data = json.load(f)
                        
                        # Agregar categorías unificadas
                        if 'categories' in coco_data:
                            for cat in coco_data['categories']:
                                unified_name = self.unify_class_names(cat['name'])
                                if unified_name not in unified_categories:
                                    new_id = len(unified_categories) + 1
                                    unified_categories[unified_name] = {
                                        'id': new_id,
                                        'name': unified_name,
                                        'supercategory': cat.get('supercategory', 'dental')
                                    }
                        
                        # Procesar imágenes y anotaciones
                        if 'images' in coco_data and 'annotations' in coco_data:
                            for img in coco_data['images']:
                                img_path = dataset_path / img['file_name']
                                if img_path.exists():
                                    all_images.append({
                                        'coco_image': img,
                                        'image_path': img_path,
                                        'annotations': [ann for ann in coco_data['annotations'] 
                                                      if ann['image_id'] == img['id']],
                                        'dataset': dataset_name,
                                        'categories': coco_data.get('categories', [])
                                    })
                                    processing_stats['images_found'] += 1
                        
                        processing_stats['datasets_processed'] += 1
                        log_message(f"✅ Procesado: {ann_file.name}")
                        
                    except Exception as e:
                        error_msg = f"Error procesando {ann_file}: {e}"
                        log_message(f"❌ {error_msg}")
                        processing_stats['errors'].append(error_msg)
        
        if not all_images:
            error_msg = "❌ No se encontraron imágenes válidas con anotaciones COCO"
            log_message(error_msg)
            print(error_msg)
            return str(dataset_dir)
        
        log_message(f"📊 Total de imágenes a procesar: {len(all_images):,}")
        log_message(f"🏷️ Categorías unificadas: {len(unified_categories)}")
        
        # División estratificada
        random.seed(42)
        random.shuffle(all_images)
        
        total_images = len(all_images)
        train_count = int(total_images * self.workflow_config['train_ratio'])
        val_count = int(total_images * self.workflow_config['val_ratio'])
        
        splits = {
            'train': all_images[:train_count],
            'val': all_images[train_count:train_count + val_count],
            'test': all_images[train_count + val_count:]
        }
        
        # Configurar categorías para todos los splits
        categories_list = list(unified_categories.values())
        for split in coco_annotations:
            coco_annotations[split]['categories'] = categories_list
        
        # Procesar cada split
        for split_name, images in splits.items():
            log_message(f"📁 Procesando {len(images)} imágenes para {split_name}")
            
            image_id_counter = 1
            annotation_id_counter = 1
            
            for img_data in tqdm(images, desc=f"Procesando {split_name}"):
                try:
                    # Copiar imagen
                    img_ext = img_data['image_path'].suffix
                    new_img_name = f"{img_data['dataset']}_{image_id_counter:06d}{img_ext}"
                    new_img_path = dataset_dir / split_name / new_img_name
                    
                    if self.safe_copy_file(img_data['image_path'], new_img_path):
                        processing_stats['images_copied'] += 1
                        
                        # Crear entrada de imagen COCO
                        coco_image = {
                            'id': image_id_counter,
                            'file_name': new_img_name,
                            'height': img_data['coco_image']['height'],
                            'width': img_data['coco_image']['width']
                        }
                        coco_annotations[split_name]['images'].append(coco_image)
                        
                        # Procesar anotaciones
                        for ann in img_data['annotations']:
                            # Mapear categoría
                            original_cat_id = ann['category_id']
                            original_cat = next((cat for cat in img_data['categories'] 
                                               if cat['id'] == original_cat_id), None)
                            
                            if original_cat:
                                unified_name = self.unify_class_names(original_cat['name'])
                                new_cat_id = unified_categories[unified_name]['id']
                                
                                # Crear nueva anotación
                                new_annotation = {
                                    'id': annotation_id_counter,
                                    'image_id': image_id_counter,
                                    'category_id': new_cat_id,
                                    'bbox': ann['bbox'],
                                    'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                                    'iscrowd': ann.get('iscrowd', 0)
                                }
                                
                                # Agregar segmentación si existe
                                if 'segmentation' in ann:
                                    new_annotation['segmentation'] = ann['segmentation']
                                
                                coco_annotations[split_name]['annotations'].append(new_annotation)
                                annotation_id_counter += 1
                                processing_stats['annotations_processed'] += 1
                        
                        image_id_counter += 1
                    
                except Exception as e:
                    error_msg = f"Error procesando imagen {img_data['image_path']}: {e}"
                    log_message(f"❌ {error_msg}")
                    processing_stats['errors'].append(error_msg)
        
        # Guardar archivos de anotaciones COCO
        for split_name in ['train', 'val', 'test']:
            ann_file = dataset_dir / split_name / 'annotations.json'
            with open(ann_file, 'w', encoding='utf-8') as f:
                json.dump(coco_annotations[split_name], f, indent=2, ensure_ascii=False)
            log_message(f"💾 Anotaciones {split_name} guardadas: {ann_file}")
        
        # Crear script de entrenamiento para segmentación
        self._create_segmentation_training_script(output_name, target_type)
        
        # Guardar estadísticas
        stats_file = log_dir / f'{output_name}_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(processing_stats, f, indent=2, ensure_ascii=False, default=str)
        
        # Resumen final
        log_message("✅ DATASET COCO CREADO EN DENTAL-AI")
        print(f"\n✅ Dataset COCO creado en dental-ai: {dataset_dir}")
        print(f"📊 Estadísticas: {stats_file}")
        print(f"📋 Log completo: {log_file}")
        
        return str(dataset_dir)
    
    def _create_segmentation_training_script(self, dataset_name: str, target_type: str):
        """Crea script de entrenamiento para segmentación."""
        training_dir = self.output_path / 'training'
        script_file = training_dir / f'train_seg_{dataset_name}.py'
        
        script_content = f'''#!/usr/bin/env python3
# 🦷 Entrenamiento de segmentación para {dataset_name}

import os
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

def setup_config():
    """Configuración para entrenamiento de segmentación."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Configuración del dataset
    cfg.DATASETS.TRAIN = ("{dataset_name}_train",)
    cfg.DATASETS.TEST = ("{dataset_name}_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Configuración de entrenamiento
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = (2000,)
    cfg.SOLVER.GAMMA = 0.1
    
    # Configuración del modelo
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("{dataset_name}_train").thing_classes)
    
    # Directorio de salida
    cfg.OUTPUT_DIR = f"./logs/{dataset_name}_segmentation"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def register_datasets():
    """Registra los datasets COCO."""
    dataset_path = "../datasets/{target_type}/{dataset_name}"
    
    # Registrar splits
    for split in ["train", "val", "test"]:
        dataset_name_split = f"{dataset_name}_{split}"
        if dataset_name_split not in DatasetCatalog.list():
            register_coco_instances(
                dataset_name_split,
                {{}},
                f"{dataset_path}/{split}/annotations.json",
                f"{dataset_path}/{split}"
            )

def main():
    setup_logger()
    print("🦷 Iniciando entrenamiento de segmentación para {dataset_name}")
    
    # Registrar datasets
    register_datasets()
    
    # Configurar entrenamiento
    cfg = setup_config()
    
    # Crear trainer y entrenar
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print("✅ Entrenamiento completado")
    print(f"📁 Modelo guardado en: {{cfg.OUTPUT_DIR}}")

if __name__ == "__main__":
    main()
'''
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Hacer ejecutable
        import stat
        script_file.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        print(f"📝 Script de entrenamiento de segmentación creado: {script_file}")
    
    def create_unified_classification_dataset(self, data: Dict[str, Any], datasets_to_merge: List[str], 
                                            output_name: str = "dental_classification", 
                                            target_type: str = "classification") -> str:
        """🧱 Crea dataset de clasificación unificado en la estructura dental-ai."""
        print(f"\n🔨 CREANDO DATASET DE CLASIFICACIÓN EN DENTAL-AI: {output_name}")
        print("🛡️ MODO SEGURO: Los datos originales NO serán modificados")
        
        # Crear estructura dental-ai si no existe
        self.create_dental_ai_structure()
        
        # Directorio de destino en dental-ai
        dataset_dir = self.output_path / 'datasets' / target_type / output_name
        
        # Crear directorio de logs
        log_dir = self.output_path / 'training' / 'logs' / f'{output_name}_creation'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        unified_classes = set()
        all_images = []
        processing_stats = {
            'datasets_processed': 0,
            'images_found': 0,
            'images_copied': 0,
            'classes_found': 0,
            'errors': []
        }
        
        log_file = log_dir / f'{output_name}_processing.log'
        
        def log_message(message: str):
            print(f"   {message}")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - {message}\n")
        
        log_message("🚀 Iniciando creación de dataset de clasificación en dental-ai")
        log_message(f"📂 Origen: {self.base_path}")
        log_message(f"📁 Destino: {dataset_dir}")
        
        # Procesar datasets de clasificación
        for dataset_type, dataset_info in data.items():
            if not dataset_type.startswith('_'):
                continue
            
            for dataset_name, dataset_data in dataset_info.get('datasets', {}).items():
                if dataset_name not in datasets_to_merge:
                    continue
                
                log_message(f"📂 Procesando dataset: {dataset_name}")
                dataset_path = Path(dataset_data['path'])
                
                # Buscar estructura de carpetas por clase
                class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
                
                if not class_dirs:
                    # Buscar imágenes con etiquetas en nombres o metadatos
                    log_message(f"⚠️ No se encontraron carpetas de clases en {dataset_path}")
                    continue
                
                for class_dir in class_dirs:
                    class_name = self.unify_class_names(class_dir.name)
                    unified_classes.add(class_name)
                    
                    # Buscar imágenes en la carpeta de clase
                    image_files = []
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        found_files = list(class_dir.rglob(f'*{ext}'))
                        image_files.extend(found_files)
                    
                    for img_file in image_files:
                        all_images.append({
                            'image_path': img_file,
                            'class_name': class_name,
                            'original_class': class_dir.name,
                            'dataset': dataset_name
                        })
                        processing_stats['images_found'] += 1
                    
                    log_message(f"   🏷️ Clase '{class_name}': {len(image_files)} imágenes")
                
                processing_stats['datasets_processed'] += 1
                processing_stats['classes_found'] += len(class_dirs)
        
        if not all_images:
            error_msg = "❌ No se encontraron imágenes válidas para clasificación"
            log_message(error_msg)
            print(error_msg)
            return str(dataset_dir)
        
        log_message(f"📊 Total de imágenes: {len(all_images):,}")
        log_message(f"🏷️ Clases unificadas: {len(unified_classes)}")
        
        # Crear estructura de carpetas para cada split y clase
        for split in ['train', 'val', 'test']:
            for class_name in unified_classes:
                class_dir = dataset_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
        
        # División estratificada por clase
        from collections import defaultdict
        images_by_class = defaultdict(list)
        
        for img_data in all_images:
            images_by_class[img_data['class_name']].append(img_data)
        
        # Dividir cada clase proporcionalmente
        splits = {'train': [], 'val': [], 'test': []}
        
        for class_name, class_images in images_by_class.items():
            random.seed(42)
            random.shuffle(class_images)
            
            total = len(class_images)
            train_count = int(total * self.workflow_config['train_ratio'])
            val_count = int(total * self.workflow_config['val_ratio'])
            
            splits['train'].extend(class_images[:train_count])
            splits['val'].extend(class_images[train_count:train_count + val_count])
            splits['test'].extend(class_images[train_count + val_count:])
        
        # Copiar imágenes a sus carpetas correspondientes
        for split_name, images in splits.items():
            log_message(f"📁 Copiando {len(images)} imágenes a {split_name}")
            
            class_counters = defaultdict(int)
            
            for img_data in tqdm(images, desc=f"Copiando {split_name}"):
                try:
                    class_name = img_data['class_name']
                    class_counters[class_name] += 1
                    
                    # Crear nombre único
                    img_ext = img_data['image_path'].suffix
                    new_img_name = f"{img_data['dataset']}_{class_counters[class_name]:06d}{img_ext}"
                    new_img_path = dataset_dir / split_name / class_name / new_img_name
                    
                    if self.safe_copy_file(img_data['image_path'], new_img_path):
                        processing_stats['images_copied'] += 1
                    else:
                        error_msg = f"Error copiando imagen: {img_data['image_path']}"
                        processing_stats['errors'].append(error_msg)
                
                except Exception as e:
                    error_msg = f"Error procesando {img_data['image_path']}: {e}"
                    log_message(f"❌ {error_msg}")
                    processing_stats['errors'].append(error_msg)
        
        # Crear archivo de clases
        classes_file = dataset_dir / 'classes.txt'
        with open(classes_file, 'w', encoding='utf-8') as f:
            for class_name in sorted(unified_classes):
                f.write(f"{class_name}\n")
        
        # Crear metadatos del dataset
        metadata = {
            'dataset_name': output_name,
            'task': 'classification',
            'num_classes': len(unified_classes),
            'classes': sorted(list(unified_classes)),
            'splits': {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test'])
            },
            'source_datasets': datasets_to_merge,
            'created': datetime.now().isoformat()
        }
        
        metadata_file = dataset_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Crear script de entrenamiento
        self._create_classification_training_script(output_name, target_type)
        
        # Guardar estadísticas
        stats_file = log_dir / f'{output_name}_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(processing_stats, f, indent=2, ensure_ascii=False, default=str)
        
        # Resumen final
        log_message("✅ DATASET DE CLASIFICACIÓN CREADO EN DENTAL-AI")
        print(f"\n✅ Dataset de clasificación creado: {dataset_dir}")
        print(f"📊 Estadísticas: {stats_file}")
        print(f"📋 Log completo: {log_file}")
        
        return str(dataset_dir)
    
    def _create_classification_training_script(self, dataset_name: str, target_type: str):
        """Crea script de entrenamiento para clasificación."""
        training_dir = self.output_path / 'training'
        script_file = training_dir / f'train_cls_{dataset_name}.py'
        
        script_content = f'''#!/usr/bin/env python3
# 🦷 Entrenamiento de clasificación para {dataset_name}

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from datetime import datetime

def create_data_loaders(data_dir, batch_size=32):
    """Crea data loaders para entrenamiento."""
    
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
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
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
    output_dir = f"./logs/{dataset_name}_classification"
    os.makedirs(output_dir, exist_ok=True)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {{epoch+1}}/{{num_epochs}}")
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
        
        print(f"Train Loss: {{epoch_loss:.4f}} Acc: {{epoch_acc:.4f}}")
        
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
        
        print(f"Val Loss: {{val_loss:.4f}} Acc: {{val_acc:.4f}}")
        
        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 
                      os.path.join(output_dir, 'best_model.pth'))
        
        scheduler.step()
        print()
    
    print(f"Mejor precisión de validación: {{best_acc:.4f}}")
    return model

def main():
    print("🦷 Iniciando entrenamiento de clasificación para {dataset_name}")
    
    # Configuración
    data_dir = "../datasets/{target_type}/{dataset_name}"
    batch_size = 32
    num_epochs = 50
    
    # Crear data loaders
    train_loader, val_loader, classes = create_data_loaders(data_dir, batch_size)
    print(f"Clases encontradas: {{classes}}")
    print(f"Número de clases: {{len(classes)}}")
    
    # Crear modelo
    model = create_model(len(classes))
    
    # Entrenar
    trained_model = train_model(model, train_loader, val_loader, num_epochs)
    
    print("✅ Entrenamiento completado")
    print(f"📁 Modelo guardado en: ./logs/{dataset_name}_classification/")

if __name__ == "__main__":
    main()
'''
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Hacer ejecutable
        import stat
        script_file.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        
        print(f"📝 Script de entrenamiento de clasificación creado: {script_file}")
    
    def create_api_template(self):
        """Crea plantilla de API para inferencia."""
        try:
            # Verificar que output_path está definido
            if not hasattr(self, 'output_path') or self.output_path is None:
                print("⚠️  output_path no definido. Usando directorio por defecto.")
                self.output_path = Path("dental-ai")
            
            api_dir = self.output_path / 'api'
            api_dir.mkdir(parents=True, exist_ok=True)
            
            # API principal
            main_api_file = api_dir / 'main.py'
        api_content = '''#!/usr/bin/env python3
# 🦷 API de Inferencia Dental AI

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import cv2
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
import uvicorn

app = FastAPI(
    title="🦷 Dental AI API",
    description="API para análisis dental con deep learning",
    version="1.0.0"
)

# Configuración global
MODELS_DIR = Path("../models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache de modelos cargados
loaded_models = {}

@app.get("/")
async def root():
    return {
        "message": "🦷 Dental AI API",
        "version": "1.0.0",
        "status": "active",
        "device": str(DEVICE)
    }

@app.get("/models")
async def list_models():
    """Lista modelos disponibles."""
    models = {
        "detection": list((MODELS_DIR / "yolo_detect").glob("*.pt")) if (MODELS_DIR / "yolo_detect").exists() else [],
        "segmentation": list((MODELS_DIR / "yolo_segment").glob("*.pt")) if (MODELS_DIR / "yolo_segment").exists() else [],
        "classification": list((MODELS_DIR / "cnn_classifier").glob("*.pth")) if (MODELS_DIR / "cnn_classifier").exists() else []
    }
    
    return {
        "available_models": {k: [str(m.name) for m in v] for k, v in models.items()},
        "total_models": sum(len(v) for v in models.values())
    }

@app.post("/predict/detection")
async def predict_detection(
    file: UploadFile = File(...),
    model_name: str = "panoramic_detection_best.pt",
    confidence: float = 0.25
):
    """Detección de estructuras dentales."""
    try:
        # Cargar imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Aquí iría la lógica de inferencia YOLO
        # Por ahora retornamos un ejemplo
        
        return {
            "predictions": [
                {
                    "class": "tooth",
                    "confidence": 0.89,
                    "bbox": [100, 150, 200, 250]
                },
                {
                    "class": "caries",
                    "confidence": 0.67,
                    "bbox": [180, 200, 220, 240]
                }
            ],
            "model_used": model_name,
            "image_size": image.size,
            "processing_time": "0.45s"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/classification")
async def predict_classification(
    file: UploadFile = File(...),
    model_name: str = "dental_classification_best.pth"
):
    """Clasificación de patologías dentales."""
    try:
        # Cargar imagen
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Aquí iría la lógica de inferencia CNN
        # Por ahora retornamos un ejemplo
        
        return {
            "prediction": {
                "class": "caries",
                "confidence": 0.87,
                "probabilities": {
                    "normal": 0.13,
                    "caries": 0.87,
                    "filling": 0.00
                }
            },
            "model_used": model_name,
            "image_size": image.size,
            "processing_time": "0.23s"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Verificación de salud del API."""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": len(loaded_models)
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
'''
        
            try:
                with open(main_api_file, 'w') as f:
                    f.write(api_content)
                
                # Archivo de requirements para la API
                api_requirements = api_dir / 'requirements.txt'
                with open(api_requirements, 'w') as f:
                    f.write('''fastapi>=0.100.0
uvicorn[standard]>=0.22.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
Pillow>=9.5.0
python-multipart>=0.0.6
''')
                
                print(f"✅ API template creada en: {api_dir}")
                print(f"🚀 Para ejecutar: cd {api_dir.name}/api && python main.py")
                
            except PermissionError:
                print(f"❌ Error de permisos al crear archivos en: {api_dir}")
                print("   Verifica que tienes permisos de escritura en el directorio")
            except Exception as e:
                print(f"❌ Error creando API template: {e}")
                
        except Exception as e:
            print(f"❌ Error en create_api_template: {e}")
            print("   Continuando con el resto del proceso...")


    def generate_training_recommendations(self, class_distribution: Dict[str, int], 
                                         fusion_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """Genera recomendaciones de entrenamiento basadas en el análisis de datos."""
        
        total_samples = sum(class_distribution.values())
        avg_samples = total_samples / len(class_distribution) if class_distribution else 0
        
        recommendations = {
            'dataset_preparation': [],
            'training_strategy': [],
            'augmentation_suggestions': [],
            'model_architecture': [],
            'hyperparameters': {}
        }
        
        # Análisis de balance de clases
        imbalanced_classes = []
        for class_name, count in class_distribution.items():
            if count < avg_samples * 0.5:  # Menos del 50% del promedio
                imbalanced_classes.append(class_name)
        
        if imbalanced_classes:
            recommendations['dataset_preparation'].extend([
                f"Clases desbalanceadas detectadas: {', '.join(imbalanced_classes)}",
                "Considerar técnicas de balanceo o augmentación específica",
                "Usar weighted loss functions durante el entrenamiento"
            ])
            
            recommendations['augmentation_suggestions'].extend([
                "Aplicar augmentación agresiva a clases minoritarias",
                "Técnicas recomendadas: rotación, cambio de brillo, noise",
                "Considerar synthetic data generation"
            ])
        
        # Recomendaciones basadas en tamaño del dataset
        if total_samples < 1000:
            recommendations['training_strategy'].extend([
                "Dataset pequeño detectado",
                "Usar transfer learning con modelos pre-entrenados",
                "Aplicar data augmentation extensiva",
                "Usar learning rate bajo para fine-tuning"
            ])
            recommendations['hyperparameters'].update({
                'learning_rate': 0.001,
                'batch_size': 8,
                'epochs': 100
            })
        elif total_samples < 5000:
            recommendations['training_strategy'].extend([
                "Dataset mediano - buen balance entre training from scratch y transfer learning",
                "Considerar mixup o cutmix para regularización"
            ])
            recommendations['hyperparameters'].update({
                'learning_rate': 0.01,
                'batch_size': 16,
                'epochs': 80
            })
        else:
            recommendations['training_strategy'].extend([
                "Dataset grande - training from scratch viable",
                "Considerar ensemble methods"
            ])
            recommendations['hyperparameters'].update({
                'learning_rate': 0.01,
                'batch_size': 32,
                'epochs': 60
            })
        
        # Recomendaciones de arquitectura basadas en el número de clases
        num_classes = len(class_distribution)
        if num_classes <= 5:
            recommendations['model_architecture'].extend([
                "Pocas clases - modelos simples pueden ser efectivos",
                "Recomendados: ResNet50, EfficientNet-B0"
            ])
        elif num_classes <= 20:
            recommendations['model_architecture'].extend([
                "Número moderado de clases - modelos medianos recomendados",
                "Recomendados: ResNet101, EfficientNet-B3"
            ])
        else:
            recommendations['model_architecture'].extend([
                "Muchas clases - modelos más complejos necesarios",
                "Recomendados: ResNet152, EfficientNet-B5+"
            ])
        
        # Recomendaciones basadas en fusión de datasets
        if len(fusion_groups) > 1:
            recommendations['dataset_preparation'].extend([
                "Múltiples grupos de fusión disponibles",
                "Considerar entrenar modelos especializados por grupo",
                "Evaluar domain adaptation techniques"
            ])
        
        return recommendations

    # ...existing code...
    
def main():
    """Función principal para ejecutar el workflow manager."""
    print("🎛️ WORKFLOW MANAGER PARA DATASETS DENTALES (MODO SEGURO)")
    print("="*65)
    print("🛡️ GARANTÍA DE SEGURIDAD: Los datos originales NUNCA se modifican")
    print("📁 Todos los datos procesados se crean en dental-ai/")
    print("="*65)
    
    base_path = "_dataSets"
    # Usar estructura dental-ai directamente
    output_path = "dental-ai"
    
    # Mostrar configuración de seguridad
    print(f"\n📂 Datos originales (SOLO LECTURA): {base_path}")
    print(f"📁 Estructura dental-ai: {output_path}")
    
    manager = DentalDataWorkflowManager(base_path, output_path)
    
    # Verificar que el directorio de datos originales existe
    if not Path(base_path).exists():
        print(f"❌ Error: No se encontró el directorio de datos originales: {base_path}")
        return
    
    # Crear estructura dental-ai si no existe
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ Estructura dental-ai preparada: {output_path}")
    
    while True:
        print(f"\n🎯 ¿Qué operación quieres realizar?")
        print("1. 🔍 Analizar distribución de clases")
        print("2. 🔗 Recomendar fusión de datasets") 
        print("3. 🧱 Crear dataset YOLO (detección) unificado")
        print("4. 🎨 Crear dataset COCO (segmentación) unificado")
        print("5. 📂 Crear dataset de clasificación unificado")
        print("6. ⚖️ Analizar balance de clases")
        print("7. 🎯 Generar estrategia de entrenamiento")
        print("8. 🚀 Ejecutar workflow completo")
        print("9. 📊 Ver configuración de seguridad")
        print("10. 🛠️ Crear estructura dental-ai completa")
        print("11. 🌐 Crear plantilla de API")
        print("12. ❌ Salir")
        
        choice = input(f"\n👉 Selecciona una opción (1-12): ").strip()
        
        if choice == '1':
            data = manager.load_analysis_data()
            if data:
                class_dist, total_samples = manager.analyze_class_distribution(data)
        
        elif choice == '2':
            data = manager.load_analysis_data()
            if data:
                fusion_groups = manager.recommend_dataset_fusion(data)
        
        elif choice == '3':
            data = manager.load_analysis_data()
            if data:
                print(f"\n🛡️ MODO SEGURO ACTIVADO")
                print(f"📂 Los datos originales en {base_path} NO serán modificados")
                print(f"📁 El dataset YOLO se creará en dental-ai/datasets/detection_combined/")
                
                datasets_to_merge = input("\nIngresa nombres de datasets YOLO separados por coma: ").split(',')
                datasets_to_merge = [d.strip() for d in datasets_to_merge if d.strip()]
                
                output_name = input("Nombre del dataset unificado (opcional): ").strip() or "dental_detection"
                
                print(f"\n🔍 Datasets a fusionar: {', '.join(datasets_to_merge)}")
                print(f"📝 Nombre de salida: {output_name}")
                
                confirm = input("\n¿Proceder con la creación segura? (s/N): ").strip().lower()
                if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                    result_path = manager.create_unified_yolo_dataset(data, datasets_to_merge, output_name)
                    print(f"\n✅ Dataset YOLO creado exitosamente en: {result_path}")
                else:
                    print("❌ Operación cancelada")
        
        elif choice == '4':
            data = manager.load_analysis_data()
            if data:
                print(f"\n🛡️ MODO SEGURO ACTIVADO")
                print(f"📂 Los datos originales en {base_path} NO serán modificados")
                print(f"📁 El dataset COCO se creará en dental-ai/datasets/segmentation_coco/")
                
                datasets_to_merge = input("\nIngresa nombres de datasets COCO separados por coma: ").split(',')
                datasets_to_merge = [d.strip() for d in datasets_to_merge if d.strip()]
                
                output_name = input("Nombre del dataset unificado (opcional): ").strip() or "dental_segmentation"
                
                print(f"\n🔍 Datasets a fusionar: {', '.join(datasets_to_merge)}")
                print(f"📝 Nombre de salida: {output_name}")
                
                confirm = input("\n¿Proceder con la creación segura? (s/N): ").strip().lower()
                if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                    result_path = manager.create_unified_coco_dataset(data, datasets_to_merge, output_name)
                    print(f"\n✅ Dataset COCO creado exitosamente en: {result_path}")
                else:
                    print("❌ Operación cancelada")
        
        elif choice == '5':
            data = manager.load_analysis_data()
            if data:
                print(f"\n🛡️ MODO SEGURO ACTIVADO")
                print(f"📂 Los datos originales en {base_path} NO serán modificados")
                print(f"📁 El dataset de clasificación se creará en dental-ai/datasets/classification/")
                
                datasets_to_merge = input("\nIngresa nombres de datasets de clasificación separados por coma: ").split(',')
                datasets_to_merge = [d.strip() for d in datasets_to_merge if d.strip()]
                
                output_name = input("Nombre del dataset unificado (opcional): ").strip() or "dental_classification"
                
                print(f"\n🔍 Datasets a fusionar: {', '.join(datasets_to_merge)}")
                print(f"📝 Nombre de salida: {output_name}")
                
                confirm = input("\n¿Proceder con la creación segura? (s/N): ").strip().lower()
                if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                    result_path = manager.create_unified_classification_dataset(data, datasets_to_merge, output_name)
                    print(f"\n✅ Dataset de clasificación creado exitosamente en: {result_path}")
                else:
                    print("❌ Operación cancelada")
        
        elif choice == '6':
            data = manager.load_analysis_data()
            if data:
                _, total_samples = manager.analyze_class_distribution(data)
                manager.identify_imbalanced_classes(total_samples)
        
        elif choice == '7':
            data = manager.load_analysis_data()
            if data:
                class_dist, _ = manager.analyze_class_distribution(data)
                fusion_groups = manager.recommend_dataset_fusion(data)
                recommendations = manager.generate_training_recommendations(class_dist, fusion_groups)
                print("\n🤖 RECOMENDACIONES DE ENTRENAMIENTO:")
                for key, value in recommendations.items():
                    print(f"\n{key.replace('_', ' ').title()}:")
                    if isinstance(value, list):
                        for item in value:
                            print(f"  • {item}")
                    else:
                        print(f"  {value}")
        
        elif choice == '8':
            print(f"\n🛡️ WORKFLOW COMPLETO EN MODO SEGURO")
            print(f"📂 Datos originales: {base_path} (PROTEGIDOS)")
            print(f"📁 Estructura dental-ai: {output_path} (NUEVA CREACIÓN)")
            
            confirm = input("\n¿Ejecutar workflow completo seguro? (s/N): ").strip().lower()
            if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                manager.run_complete_workflow()
            else:
                print("❌ Workflow cancelado")
        
        elif choice == '9':
            print(f"\n🛡️ CONFIGURACIÓN DE SEGURIDAD ACTIVA")
            print("="*50)
            print(f"📂 Directorio original: {manager.base_path}")
            print(f"📁 Estructura dental-ai: {manager.output_path}")
            print(f"🔒 Modo solo lectura: {manager.safety_config['read_only_source']}")
            print(f"✅ Verificación de copia: {manager.safety_config['verify_copy']}")
            print(f"📋 Backup habilitado: {manager.safety_config['backup_enabled']}")
            print(f"🏗️ Preservar estructura: {manager.safety_config['preserve_original_structure']}")
            print("\n🛡️ GARANTÍAS:")
            print("• Los archivos originales NUNCA se modifican")
            print("• Todas las operaciones son de SOLO LECTURA en origen")
            print("• Los datos procesados se crean en dental-ai/")
            print("• Se mantiene trazabilidad completa")
            print("• Los archivos copiados tienen verificación de integridad")
            print("="*50)
        
        elif choice == '10':
            print(f"\n🏗️ CREANDO ESTRUCTURA COMPLETA DENTAL-AI EN: {output_path}")
            print("Esta operación creará todos los directorios y archivos necesarios")
            confirm = input("¿Proceder? (s/N): ").strip().lower()
            
            if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                manager.create_dental_ai_structure()
                print(f"\n✅ Estructura dental-ai creada completa")
                print(f"📂 Directorios principales:")
                print(f"   • dental-ai/datasets/ (datasets procesados)")
                print(f"   • dental-ai/models/ (modelos entrenados)")
                print(f"   • dental-ai/training/ (scripts y configs)")
                print(f"   • dental-ai/api/ (API de inferencia)")
                print(f"   • dental-ai/docs/ (documentación)")
            else:
                print("❌ Operación cancelada")
        
        elif choice == '11':
            print(f"\n🌐 CREANDO PLANTILLA DE API EN: {output_path}/api/")
            confirm = input("¿Proceder? (s/N): ").strip().lower()
            
            if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                try:
                    manager.create_api_template()
                    print(f"\n✅ Plantilla de API creada")
                    print(f"📝 Archivo principal: dental-ai/api/main.py")
                    print(f"📋 Requirements: dental-ai/api/requirements.txt")
                    print(f"\n🚀 Para usar la API:")
                    print(f"   cd dental-ai/api")
                    print(f"   pip install -r requirements.txt")
                    print(f"   python main.py")
                    print(f"   Navega a: http://localhost:8000/docs")
                except Exception as e:
                    print(f"❌ Error creando plantilla de API: {e}")
                    print("   Verifica permisos y espacio en disco")
            else:
                print("❌ Operación cancelada")
        
        elif choice == '12':
            print("👋 ¡Hasta luego!")
            print("🛡️ Recuerda: Todos tus datos originales están seguros")
            print("🏗️ Tu estructura dental-ai está lista para usar")
            break
        
        else:
            print("❌ Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    main()
