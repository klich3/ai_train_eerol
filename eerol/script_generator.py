"""
🚀 Script Generator Module
==========================

Generates training scripts for different deep learning frameworks
based on dataset format and structure.
"""

import os
import stat
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ScriptGenerator:
    """🚀 Training script generator."""
    
    def __init__(self, dataset_path: Path):
        """Initialize generator with dataset path."""
        self.dataset_path = Path(dataset_path)
        self.framework_templates = {
            'yolo': self._generate_yolo_script,
            'coco': self._generate_coco_script,
            'pytorch': self._generate_pytorch_script,
            'tensorflow': self._generate_tensorflow_script,
            'unet': self._generate_unet_script
        }
    
    def execute_training(self) -> Dict[str, Any]:
        """🚀 Execute training script for the dataset."""
        
        # Check if training script exists
        script_path = self.dataset_path / 'train.py'
        if not script_path.exists():
            # Generate script first
            result = self.generate_training_script()
            if not result['success']:
                return result
        
        print(f"🚀 Ejecutando entrenamiento para: {self.dataset_path.name}")
        print(f"📁 Script: {script_path}")
        
        try:
            # Make script executable
            os.chmod(script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
            
            # Execute script
            result = subprocess.run(
                ['python', str(script_path)],
                cwd=str(self.dataset_path),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'message': 'Entrenamiento iniciado exitosamente',
                    'output': result.stdout
                }
            else:
                return {
                    'success': False,
                    'error': f'Error ejecutando script: {result.stderr}',
                    'output': result.stdout
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Error ejecutando entrenamiento: {str(e)}'
            }
    
    def generate_training_script(self, framework: str = None) -> Dict[str, Any]:
        """📝 Generate training script for the dataset."""
        
        try:
            # Detect dataset format if framework not specified
            if not framework:
                framework = self._detect_dataset_format()
            
            if framework not in self.framework_templates:
                return {
                    'success': False,
                    'error': f'Framework {framework} no soportado. Disponibles: {list(self.framework_templates.keys())}'
                }
            
            # Generate script based on framework
            script_content = self.framework_templates[framework]()
            
            # Save script
            script_path = self.dataset_path / 'train.py'
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
            
            return {
                'success': True,
                'script_path': str(script_path),
                'framework': framework
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error generando script: {str(e)}'
            }
    
    def _detect_dataset_format(self) -> str:
        """🔍 Detect dataset format."""
        
        # Check for YOLO format
        if (self.dataset_path / 'data.yaml').exists():
            return 'yolo'
        
        # Check for COCO format
        json_files = list(self.dataset_path.glob('**/*.json'))
        for json_file in json_files:
            if 'annotation' in json_file.name.lower():
                return 'coco'
        
        # Check directory structure
        subdirs = [d.name.lower() for d in self.dataset_path.iterdir() if d.is_dir()]
        
        if 'train' in subdirs and 'val' in subdirs:
            # Check if it has images and labels structure (YOLO)
            train_dir = self.dataset_path / 'train'
            if (train_dir / 'images').exists() and (train_dir / 'labels').exists():
                return 'yolo'
        
        # Default to PyTorch for generic datasets
        return 'pytorch'
    
    def _generate_yolo_script(self) -> str:
        """🎯 Generate YOLOv8 training script."""
        
        # Load dataset info
        classes = []
        try:
            if YAML_AVAILABLE:
                import yaml
                with open(self.dataset_path / 'data.yaml', 'r') as f:
                    data = yaml.safe_load(f)
                    classes = data.get('names', [])
        except:
            classes = ['class_0']  # Default
        
        script = f'''#!/usr/bin/env python3
"""
🎯 YOLOv8 Training Script
========================

Dataset: {self.dataset_path.name}
Classes: {len(classes)}
Generated: {datetime.now().isoformat()}
"""

import os
import sys
from pathlib import Path

# Add dataset path to sys.path
sys.path.append(str(Path(__file__).parent))

def install_requirements():
    """📦 Install required packages."""
    import subprocess
    
    requirements = [
        'ultralytics',
        'torch',
        'torchvision',
        'opencv-python',
        'pillow',
        'pyyaml'
    ]
    
    for package in requirements:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {{package}} ya está instalado")
        except ImportError:
            print(f"📦 Instalando {{package}}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def train_yolo():
    """🚀 Train YOLOv8 model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics no está instalado. Instalando...")
        install_requirements()
        from ultralytics import YOLO
    
    # Configuration
    dataset_config = Path(__file__).parent / 'data.yaml'
    
    if not dataset_config.exists():
        print(f"❌ No se encontró {{dataset_config}}")
        return False
    
    print("🎯 YOLO TRAINING CONFIGURATION")
    print("=" * 40)
    print(f"📁 Dataset: {{Path(__file__).parent.name}}")
    print(f"⚙️ Config: {{dataset_config}}")
    print(f"🏷️ Classes: {len(classes)}")
    print(f"📝 Class names: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}")
    
    # Initialize model
    model = YOLO('yolov8n.pt')  # Start with nano model
    
    # Training parameters
    results = model.train(
        data=str(dataset_config),
        epochs=100,
        imgsz=640,
        batch=16,
        name='{self.dataset_path.name}_yolo',
        project='runs/train',
        save=True,
        save_period=10,
        cache=True,
        device='auto',  # Use GPU if available
        workers=4,
        patience=50,
        val=True
    )
    
    print("✅ Entrenamiento completado!")
    print(f"📁 Resultados guardados en: runs/train/{{results.save_dir}}")
    
    return True

def main():
    """🚀 Main training function."""
    print("🤖 EEROL YOLOv8 Training Script")
    print("=" * 40)
    
    try:
        success = train_yolo()
        if success:
            print("\\n🎉 ¡Entrenamiento exitoso!")
        else:
            print("\\n❌ Error en el entrenamiento")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\\n⚠️ Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\\n❌ Error inesperado: {{str(e)}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        return script
    
    def _generate_coco_script(self) -> str:
        """🎨 Generate COCO format training script."""
        
        script = f'''#!/usr/bin/env python3
"""
🎨 COCO Format Training Script
=============================

Dataset: {self.dataset_path.name}
Generated: {datetime.now().isoformat()}
"""

import os
import sys
import json
from pathlib import Path

def install_requirements():
    """📦 Install required packages."""
    import subprocess
    
    requirements = [
        'torch',
        'torchvision',
        'opencv-python',
        'pillow',
        'pycocotools',
        'matplotlib'
    ]
    
    for package in requirements:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {{package}} ya está instalado")
        except ImportError:
            print(f"📦 Instalando {{package}}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def load_coco_data():
    """📋 Load COCO annotation data."""
    annotation_file = None
    
    # Find COCO annotation file
    for json_file in Path(__file__).parent.glob('**/*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'images' in data and 'annotations' in data and 'categories' in data:
                    annotation_file = json_file
                    break
        except:
            continue
    
    if not annotation_file:
        print("❌ No se encontró archivo de anotaciones COCO válido")
        return None
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"📁 Archivo COCO: {{annotation_file.name}}")
    print(f"🖼️ Imágenes: {{len(coco_data['images'])}}")
    print(f"📝 Anotaciones: {{len(coco_data['annotations'])}}")
    print(f"🏷️ Categorías: {{len(coco_data['categories'])}}")
    
    return coco_data

def train_coco():
    """🚀 Train model with COCO format."""
    print("🎨 COCO TRAINING CONFIGURATION")
    print("=" * 40)
    
    # Load data
    coco_data = load_coco_data()
    if not coco_data:
        return False
    
    print("\\n⚠️ COCO training requires specific implementation.")
    print("📚 Possible frameworks:")
    print("   - Detectron2 (Facebook)")
    print("   - MMDetection")
    print("   - TensorFlow Object Detection API")
    print("   - PyTorch custom implementation")
    print("\\n💡 Edita este script para agregar tu implementación específica.")
    
    return True

def main():
    """🚀 Main training function."""
    print("🤖 EEROL COCO Training Script")
    print("=" * 40)
    
    try:
        install_requirements()
        success = train_coco()
        if success:
            print("\\n✅ Setup completado")
        else:
            print("\\n❌ Error en setup")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\\n⚠️ Proceso interrumpido")
    except Exception as e:
        print(f"\\n❌ Error: {{str(e)}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        return script
    
    def _generate_pytorch_script(self) -> str:
        """🔥 Generate PyTorch training script."""
        
        script = f'''#!/usr/bin/env python3
"""
🔥 PyTorch Training Script
=========================

Dataset: {self.dataset_path.name}
Generated: {datetime.now().isoformat()}
"""

import os
import sys
from pathlib import Path

def install_requirements():
    """📦 Install required packages."""
    import subprocess
    
    requirements = [
        'torch',
        'torchvision',
        'opencv-python',
        'pillow',
        'matplotlib',
        'numpy',
        'tqdm'
    ]
    
    for package in requirements:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {{package}} ya está instalado")
        except ImportError:
            print(f"📦 Instalando {{package}}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def train_pytorch():
    """🔥 Train PyTorch model."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        import torchvision.transforms as transforms
    except ImportError:
        print("❌ PyTorch no está instalado. Instalando...")
        install_requirements()
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        import torchvision.transforms as transforms
    
    print("🔥 PYTORCH TRAINING CONFIGURATION")
    print("=" * 40)
    print(f"📁 Dataset: {{Path(__file__).parent.name}}")
    print(f"🎯 Device: {{'CUDA' if torch.cuda.is_available() else 'CPU'}}")
    
    # Basic configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    
    print(f"⚙️ Batch size: {{batch_size}}")
    print(f"📈 Learning rate: {{learning_rate}}")
    print(f"🔄 Epochs: {{num_epochs}}")
    
    print("\\n⚠️ PyTorch training requires custom implementation.")
    print("📚 Necesitas implementar:")
    print("   - Dataset personalizado")
    print("   - Modelo de red neuronal")
    print("   - Loop de entrenamiento")
    print("   - Métricas de evaluación")
    print("\\n💡 Edita este script para agregar tu implementación específica.")
    
    return True

def main():
    """🚀 Main training function."""
    print("🤖 EEROL PyTorch Training Script")
    print("=" * 40)
    
    try:
        install_requirements()
        success = train_pytorch()
        if success:
            print("\\n✅ Setup completado")
        else:
            print("\\n❌ Error en setup")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\\n⚠️ Entrenamiento interrumpido")
    except Exception as e:
        print(f"\\n❌ Error: {{str(e)}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        return script
    
    def _generate_tensorflow_script(self) -> str:
        """🧠 Generate TensorFlow training script."""
        
        script = f'''#!/usr/bin/env python3
"""
🧠 TensorFlow Training Script
============================

Dataset: {self.dataset_path.name}
Generated: {datetime.now().isoformat()}
"""

import os
import sys
from pathlib import Path

def install_requirements():
    """📦 Install required packages."""
    import subprocess
    
    requirements = [
        'tensorflow',
        'opencv-python',
        'pillow',
        'matplotlib',
        'numpy'
    ]
    
    for package in requirements:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {{package}} ya está instalado")
        except ImportError:
            print(f"📦 Instalando {{package}}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def train_tensorflow():
    """🧠 Train TensorFlow model."""
    try:
        import tensorflow as tf
    except ImportError:
        print("❌ TensorFlow no está instalado. Instalando...")
        install_requirements()
        import tensorflow as tf
    
    print("🧠 TENSORFLOW TRAINING CONFIGURATION")
    print("=" * 40)
    print(f"📁 Dataset: {{Path(__file__).parent.name}}")
    print(f"🎯 TensorFlow version: {{tf.__version__}}")
    print(f"🎮 GPU available: {{tf.config.list_physical_devices('GPU')}}")
    
    print("\\n⚠️ TensorFlow training requires custom implementation.")
    print("📚 Considera usar:")
    print("   - TensorFlow Object Detection API")
    print("   - Keras functional API")
    print("   - tf.data para carga de datos")
    print("\\n💡 Edita este script para agregar tu implementación específica.")
    
    return True

def main():
    """🚀 Main training function."""
    print("🤖 EEROL TensorFlow Training Script")
    print("=" * 40)
    
    try:
        install_requirements()
        success = train_tensorflow()
        if success:
            print("\\n✅ Setup completado")
        else:
            print("\\n❌ Error en setup")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\\n⚠️ Entrenamiento interrumpido")
    except Exception as e:
        print(f"\\n❌ Error: {{str(e)}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        return script
    
    def _generate_unet_script(self) -> str:
        """🎭 Generate U-Net training script."""
        
        script = f'''#!/usr/bin/env python3
"""
🎭 U-Net Segmentation Training Script
====================================

Dataset: {self.dataset_path.name}
Generated: {datetime.now().isoformat()}
"""

import os
import sys
from pathlib import Path

def install_requirements():
    """📦 Install required packages."""
    import subprocess
    
    requirements = [
        'torch',
        'torchvision',
        'opencv-python',
        'pillow',
        'matplotlib',
        'numpy',
        'tqdm',
        'albumentations'
    ]
    
    for package in requirements:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {{package}} ya está instalado")
        except ImportError:
            print(f"📦 Instalando {{package}}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def train_unet():
    """🎭 Train U-Net model."""
    print("🎭 U-NET SEGMENTATION TRAINING")
    print("=" * 40)
    print(f"📁 Dataset: {{Path(__file__).parent.name}}")
    
    print("\\n⚠️ U-Net training requires specific implementation.")
    print("📚 Necesitas implementar:")
    print("   - U-Net architecture")
    print("   - Segmentation dataset loader")
    print("   - Loss functions (Dice, IoU, etc.)")
    print("   - Metrics de segmentación")
    print("\\n💡 Edita este script para agregar tu implementación específica.")
    
    return True

def main():
    """🚀 Main training function."""
    print("🤖 EEROL U-Net Training Script")
    print("=" * 40)
    
    try:
        install_requirements()
        success = train_unet()
        if success:
            print("\\n✅ Setup completado")
        else:
            print("\\n❌ Error en setup")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\\n⚠️ Entrenamiento interrumpido")
    except Exception as e:
        print(f"\\n❌ Error: {{str(e)}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        return script
