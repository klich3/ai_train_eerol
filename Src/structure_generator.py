"""
🦷 Structure Generator
Generador de estructura de directorios y documentación para dental-ai
"""

import yaml
from pathlib import Path
from datetime import datetime


class StructureGenerator:
    """Generador de estructura de directorios y documentación."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        
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
        
        # Crear archivos principales
        self.create_main_documentation()
        self.create_requirements_file()
        self.create_main_config()
        
        print(f"✅ Estructura dental-ai creada con {len(structure_created)} directorios")
        return structure_created
    
    def create_main_documentation(self):
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
    
    def create_requirements_file(self):
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
    
    def create_main_config(self):
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
