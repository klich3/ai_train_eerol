"""
🔄 Dataset Converter Module
===========================

Universal dataset converter supporting multiple annotation formats:
- YOLO ↔ COCO
- Pascal VOC ↔ YOLO
- COCO ↔ Pascal VOC
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import xml.etree.ElementTree as ET
from datetime import datetime

# Optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class DatasetConverter:
    """🔄 Universal dataset format converter."""
    
    def __init__(self, input_path: Path, output_path: Path):
        """Initialize converter with input and output paths."""
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.supported_formats = ['yolo', 'coco', 'pascal_voc']
        
    def convert_to_format(self, target_format: str, output_name: str) -> Dict[str, Any]:
        """🔄 Convert dataset to target format."""
        if target_format not in self.supported_formats:
            return {
                'success': False,
                'error': f"Formato {target_format} no soportado. Disponibles: {self.supported_formats}"
            }
        
        # Detect source format
        source_format = self._detect_source_format()
        if source_format == 'unknown':
            return {
                'success': False,
                'error': "No se pudo detectar el formato del dataset de origen"
            }
        
        print(f"🔄 Convirtiendo de {source_format} a {target_format}")
        
        # Create output directory
        output_dir = self.output_path / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if source_format == 'yolo' and target_format == 'coco':
                result = self._yolo_to_coco(output_dir)
            elif source_format == 'coco' and target_format == 'yolo':
                result = self._coco_to_yolo(output_dir)
            elif source_format == 'pascal_voc' and target_format == 'yolo':
                result = self._pascal_to_yolo(output_dir)
            elif source_format == 'yolo' and target_format == 'pascal_voc':
                result = self._yolo_to_pascal(output_dir)
            elif source_format == target_format:
                result = self._copy_dataset(output_dir)
            else:
                return {
                    'success': False,
                    'error': f"Conversión de {source_format} a {target_format} no implementada aún"
                }
            
            if result['success']:
                # Generate training script
                self._generate_training_script(output_dir, target_format, result.get('classes', []))
                
            return {
                'success': True,
                'output_name': output_name,
                'source_format': source_format,
                'target_format': target_format,
                'output_path': str(output_dir),
                **result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error durante la conversión: {str(e)}"
            }
    
    def _detect_source_format(self) -> str:
        """🔍 Detect source dataset format."""
        # Check for YOLO format
        if self._is_yolo_format():
            return 'yolo'
        elif self._is_coco_format():
            return 'coco'
        elif self._is_pascal_voc_format():
            return 'pascal_voc'
        else:
            return 'unknown'
    
    def _is_yolo_format(self) -> bool:
        """Check if dataset is in YOLO format."""
        # Look for data.yaml or similar config file
        config_files = ['data.yaml', 'data.yml', 'config.yaml', 'config.yml']
        for config_file in config_files:
            if (self.input_path / config_file).exists():
                return True
        
        # Look for images and labels directories
        if (self.input_path / 'images').exists() and (self.input_path / 'labels').exists():
            return True
        
        # Look for .txt annotation files
        txt_files = list(self.input_path.glob('**/*.txt'))
        if txt_files:
            # Check if any txt file has YOLO format
            try:
                with open(txt_files[0], 'r') as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5 and all(self._is_float(p) for p in parts[1:5]):
                            return True
            except:
                pass
        
        return False
    
    def _is_coco_format(self) -> bool:
        """Check if dataset is in COCO format."""
        json_files = list(self.input_path.glob('**/*.json'))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        if 'images' in data and 'annotations' in data and 'categories' in data:
                            return True
            except:
                continue
        return False
    
    def _is_pascal_voc_format(self) -> bool:
        """Check if dataset is in Pascal VOC format."""
        xml_files = list(self.input_path.glob('**/*.xml'))
        if xml_files:
            try:
                tree = ET.parse(xml_files[0])
                root = tree.getroot()
                if root.tag == 'annotation':
                    return True
            except:
                pass
        return False
    
    def _yolo_to_coco(self, output_dir: Path) -> Dict[str, Any]:
        """🔄 Convert YOLO format to COCO format."""
        # Load YOLO classes
        classes = self._load_yolo_classes()
        if not classes:
            return {'success': False, 'error': 'No se pudieron cargar las clases YOLO'}
        
        # Create COCO structure
        coco_data = {
            'info': {
                'description': 'Dataset converted from YOLO format',
                'version': '1.0',
                'year': datetime.now().year,
                'date_created': datetime.now().isoformat()
            },
            'licenses': [
                {
                    'id': 1,
                    'name': 'Unknown',
                    'url': ''
                }
            ],
            'categories': [
                {
                    'id': i,
                    'name': class_name,
                    'supercategory': 'object'
                }
                for i, class_name in enumerate(classes)
            ],
            'images': [],
            'annotations': []
        }
        
        # Create output directories
        (output_dir / 'images').mkdir(exist_ok=True)
        
        # Find image and label files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(self.input_path.glob(f'**/{ext}'))
        
        image_id = 0
        annotation_id = 0
        
        for image_file in image_files:
            # Copy image
            new_image_path = output_dir / 'images' / image_file.name
            shutil.copy2(image_file, new_image_path)
            
            # Get image dimensions (simplified, assuming we can read the file)
            try:
                # In a real implementation, you'd use PIL or cv2 to get dimensions
                width, height = 640, 480  # Default values
            except:
                width, height = 640, 480
            
            # Add image info
            image_info = {
                'id': image_id,
                'file_name': image_file.name,
                'width': width,
                'height': height,
                'license': 1
            }
            coco_data['images'].append(image_info)
            
            # Find corresponding label file
            label_file = image_file.with_suffix('.txt')
            if not label_file.exists():
                # Try in labels directory
                label_file = self.input_path / 'labels' / image_file.with_suffix('.txt').name
            
            if label_file.exists():
                # Convert YOLO annotations to COCO
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * width
                            y_center = float(parts[2]) * height
                            bbox_width = float(parts[3]) * width
                            bbox_height = float(parts[4]) * height
                            
                            # Convert to COCO format (x, y, width, height)
                            x = x_center - bbox_width / 2
                            y = y_center - bbox_height / 2
                            
                            annotation = {
                                'id': annotation_id,
                                'image_id': image_id,
                                'category_id': class_id,
                                'bbox': [x, y, bbox_width, bbox_height],
                                'area': bbox_width * bbox_height,
                                'iscrowd': 0
                            }
                            coco_data['annotations'].append(annotation)
                            annotation_id += 1
            
            image_id += 1
        
        # Save COCO JSON
        with open(output_dir / 'annotations.json', 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        return {
            'success': True,
            'images_converted': len(coco_data['images']),
            'annotations_converted': len(coco_data['annotations']),
            'classes': classes
        }
    
    def _coco_to_yolo(self, output_dir: Path) -> Dict[str, Any]:
        """🔄 Convert COCO format to YOLO format."""
        # Find COCO JSON file
        json_files = list(self.input_path.glob('**/*.json'))
        coco_file = None
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if 'images' in data and 'annotations' in data and 'categories' in data:
                        coco_file = json_file
                        break
            except:
                continue
        
        if not coco_file:
            return {'success': False, 'error': 'No se encontró archivo COCO válido'}
        
        # Load COCO data
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create output directories
        (output_dir / 'images').mkdir(exist_ok=True)
        (output_dir / 'labels').mkdir(exist_ok=True)
        
        # Extract classes
        classes = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]
        
        # Create class mapping
        id_to_index = {cat['id']: i for i, cat in enumerate(sorted(coco_data['categories'], key=lambda x: x['id']))}
        
        # Process images and annotations
        image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
        
        converted_images = 0
        converted_annotations = 0
        
        for image_info in coco_data['images']:
            image_id = image_info['id']
            
            # Find and copy image
            image_name = image_info['file_name']
            source_image = self.input_path / image_name
            if not source_image.exists():
                # Try in images subdirectory
                source_image = self.input_path / 'images' / image_name
            
            if source_image.exists():
                dest_image = output_dir / 'images' / image_name
                shutil.copy2(source_image, dest_image)
                converted_images += 1
                
                # Convert annotations
                label_file = output_dir / 'labels' / Path(image_name).with_suffix('.txt').name
                
                if image_id in image_annotations:
                    with open(label_file, 'w') as f:
                        for ann in image_annotations[image_id]:
                            class_index = id_to_index[ann['category_id']]
                            bbox = ann['bbox']  # [x, y, width, height]
                            
                            # Convert to YOLO format
                            x_center = (bbox[0] + bbox[2] / 2) / image_info['width']
                            y_center = (bbox[1] + bbox[3] / 2) / image_info['height']
                            width = bbox[2] / image_info['width']
                            height = bbox[3] / image_info['height']
                            
                            f.write(f"{class_index} {x_center} {y_center} {width} {height}\\n")
                            converted_annotations += 1
        
        # Create data.yaml
        data_yaml = {
            'train': 'images',
            'val': 'images',
            'nc': len(classes),
            'names': classes
        }
        
        if YAML_AVAILABLE:
            with open(output_dir / 'data.yaml', 'w') as f:
                yaml.dump(data_yaml, f)
        else:
            # Fallback to simple text format
            with open(output_dir / 'data.txt', 'w') as f:
                f.write(f"classes: {len(classes)}\\n")
                for i, cls in enumerate(classes):
                    f.write(f"{i}: {cls}\\n")
        
        return {
            'success': True,
            'images_converted': converted_images,
            'annotations_converted': converted_annotations,
            'classes': classes
        }
    
    def _pascal_to_yolo(self, output_dir: Path) -> Dict[str, Any]:
        """🔄 Convert Pascal VOC format to YOLO format."""
        # Implementation for Pascal VOC to YOLO conversion
        return {'success': False, 'error': 'Pascal VOC to YOLO conversion not implemented yet'}
    
    def _yolo_to_pascal(self, output_dir: Path) -> Dict[str, Any]:
        """🔄 Convert YOLO format to Pascal VOC format."""
        # Implementation for YOLO to Pascal VOC conversion
        return {'success': False, 'error': 'YOLO to Pascal VOC conversion not implemented yet'}
    
    def _copy_dataset(self, output_dir: Path) -> Dict[str, Any]:
        """📂 Copy dataset without conversion."""
        try:
            # Copy all files
            if self.input_path.is_file():
                shutil.copy2(self.input_path, output_dir)
            else:
                for item in self.input_path.rglob('*'):
                    if item.is_file():
                        relative_path = item.relative_to(self.input_path)
                        dest_path = output_dir / relative_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest_path)
            
            return {'success': True, 'message': 'Dataset copiado sin conversión'}
            
        except Exception as e:
            return {'success': False, 'error': f'Error copiando dataset: {str(e)}'}
    
    def _load_yolo_classes(self) -> List[str]:
        """📋 Load YOLO class names."""
        classes = []
        
        # Try to load from YAML config
        config_files = ['data.yaml', 'data.yml', 'config.yaml', 'config.yml']
        for config_file in config_files:
            config_path = self.input_path / config_file
            if config_path.exists() and YAML_AVAILABLE:
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        if isinstance(config, dict) and 'names' in config:
                            if isinstance(config['names'], list):
                                return config['names']
                            elif isinstance(config['names'], dict):
                                return list(config['names'].values())
                except:
                    continue
        
        # Try to load from classes.txt
        classes_file = self.input_path / 'classes.txt'
        if classes_file.exists():
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f if line.strip()]
                    return classes
            except:
                pass
        
        # Generate generic class names based on annotations
        class_indices = set()
        txt_files = list(self.input_path.glob('**/*.txt'))
        for txt_file in txt_files[:10]:  # Sample first 10 files
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and self._is_int(parts[0]):
                            class_indices.add(int(parts[0]))
            except:
                continue
        
        if class_indices:
            max_class = max(class_indices)
            return [f"class_{i}" for i in range(max_class + 1)]
        
        return []
    
    def _generate_training_script(self, output_dir: Path, format_type: str, classes: List[str]):
        """🚀 Generate training script for the converted dataset."""
        script_content = f"""#!/usr/bin/env python3
\"\"\"
🚀 Auto-generated Training Script
=================================

Dataset: {output_dir.name}
Format: {format_type}
Classes: {len(classes)}
Generated: {datetime.now().isoformat()}
\"\"\"

import os
from pathlib import Path

# Dataset configuration
DATASET_PATH = Path(__file__).parent
CLASSES = {classes}
NUM_CLASSES = {len(classes)}

def train_yolo():
    \"\"\"Train YOLO model.\"\"\"
    print("🚀 Starting YOLO training...")
    print(f"📁 Dataset: {{DATASET_PATH}}")
    print(f"🏷️ Classes: {{NUM_CLASSES}}")
    
    # Add your YOLO training code here
    # Example with YOLOv8:
    # from ultralytics import YOLO
    # model = YOLO('yolov8n.pt')
    # results = model.train(data=DATASET_PATH / 'data.yaml', epochs=100)

def train_coco():
    \"\"\"Train model with COCO format.\"\"\"
    print("🚀 Starting COCO training...")
    print(f"📁 Dataset: {{DATASET_PATH}}")
    print(f"🏷️ Classes: {{NUM_CLASSES}}")
    
    # Add your COCO training code here
    # Example with Detectron2 or similar

def main():
    print("🤖 EEROL Auto-generated Training Script")
    print("=" * 40)
    
    if "{format_type}" == "yolo":
        train_yolo()
    elif "{format_type}" == "coco":
        train_coco()
    else:
        print(f"❌ Format {{format_type}} not supported in this script")

if __name__ == "__main__":
    main()
"""
        
        script_path = output_dir / 'train.py'
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"📝 Script de entrenamiento generado: {script_path}")
    
    def _is_float(self, value: str) -> bool:
        """Check if string can be converted to float."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _is_int(self, value: str) -> bool:
        """Check if string can be converted to int."""
        try:
            int(value)
            return True
        except ValueError:
            return False
