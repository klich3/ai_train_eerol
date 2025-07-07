"""
👁️ Dataset Preview Module
=========================

Universal dataset preview tool for visualizing annotations
on images across different formats.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import PIL.Image as Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class DatasetPreview:
    """👁️ Universal dataset preview tool."""
    
    def __init__(self):
        """Initialize preview tool."""
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
            '#FFA500', '#800080', '#FFC0CB', '#A52A2A', '#808080', '#000080',
            '#008000', '#800000', '#808000', '#C0C0C0', '#FF6347', '#4682B4',
            '#D2691E', '#9ACD32', '#20B2AA', '#87CEEB', '#6495ED', '#DC143C'
        ]
    
    def show_preview(self, image_path: Path, annotation_path: Path, format_type: str = 'yolo'):
        """👁️ Show preview of image with annotations."""
        if not MATPLOTLIB_AVAILABLE:
            print("❌ matplotlib no está disponible. Instalalo con: pip install matplotlib")
            return
        
        if not PIL_AVAILABLE:
            print("❌ PIL no está disponible. Instalalo con: pip install Pillow")
            return
        
        try:
            # Load image
            image = Image.open(image_path)
            width, height = image.size
            
            # Load annotations
            annotations = self._load_annotations(annotation_path, format_type, width, height)
            
            if not annotations:
                print("⚠️ No se encontraron anotaciones válidas")
                return
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            ax.set_title(f"Preview: {image_path.name} ({format_type} format)")
            
            # Draw annotations
            for i, ann in enumerate(annotations):
                color = self.colors[i % len(self.colors)]
                
                if 'bbox' in ann:
                    # Bounding box annotation
                    x, y, w, h = ann['bbox']
                    rect = patches.Rectangle(
                        (x, y), w, h,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    label = ann.get('label', f"class_{ann.get('class_id', 'unknown')}")
                    ax.text(
                        x, y - 5,
                        label,
                        color=color,
                        fontsize=10,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
                    )
            
            ax.axis('off')
            plt.tight_layout()
            plt.show()
            
            # Print annotation details
            print(f"\\n📊 DETALLES DE ANOTACIONES:")
            print(f"🖼️ Imagen: {image_path.name} ({width}x{height})")
            print(f"📝 Formato: {format_type}")
            print(f"🏷️ Anotaciones encontradas: {len(annotations)}")
            
            for i, ann in enumerate(annotations):
                label = ann.get('label', f"class_{ann.get('class_id', 'unknown')}")
                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']
                    print(f"  {i+1}. {label}: bbox=({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f})")
        
        except Exception as e:
            print(f"❌ Error mostrando preview: {str(e)}")
    
    def _load_annotations(self, annotation_path: Path, format_type: str, 
                         image_width: int, image_height: int) -> List[Dict[str, Any]]:
        """📋 Load annotations from file."""
        annotations = []
        
        try:
            if format_type.lower() == 'yolo':
                annotations = self._load_yolo_annotations(annotation_path, image_width, image_height)
            elif format_type.lower() == 'coco':
                annotations = self._load_coco_annotations(annotation_path)
            elif format_type.lower() in ['pascal_voc', 'xml']:
                annotations = self._load_pascal_annotations(annotation_path)
            else:
                print(f"⚠️ Formato {format_type} no soportado")
        
        except Exception as e:
            print(f"❌ Error cargando anotaciones: {str(e)}")
        
        return annotations
    
    def _load_yolo_annotations(self, annotation_path: Path, width: int, height: int) -> List[Dict[str, Any]]:
        """📋 Load YOLO format annotations."""
        annotations = []
        
        if not annotation_path.exists():
            return annotations
        
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        bbox_width = float(parts[3]) * width
                        bbox_height = float(parts[4]) * height
                        
                        # Convert to absolute coordinates (x, y, width, height)
                        x = x_center - bbox_width / 2
                        y = y_center - bbox_height / 2
                        
                        annotations.append({
                            'class_id': class_id,
                            'bbox': [x, y, bbox_width, bbox_height],
                            'label': f"class_{class_id}"
                        })
                    except ValueError:
                        continue
        
        return annotations
    
    def _load_coco_annotations(self, annotation_path: Path) -> List[Dict[str, Any]]:
        """📋 Load COCO format annotations."""
        annotations = []
        
        if not annotation_path.exists():
            return annotations
        
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            return annotations
        
        # Create category mapping
        categories = {}
        if 'categories' in data:
            for cat in data['categories']:
                categories[cat['id']] = cat['name']
        
        # Load annotations
        if 'annotations' in data:
            for ann in data['annotations']:
                if 'bbox' in ann and 'category_id' in ann:
                    bbox = ann['bbox']  # [x, y, width, height]
                    category_id = ann['category_id']
                    label = categories.get(category_id, f"class_{category_id}")
                    
                    annotations.append({
                        'class_id': category_id,
                        'bbox': bbox,
                        'label': label
                    })
        
        return annotations
    
    def _load_pascal_annotations(self, annotation_path: Path) -> List[Dict[str, Any]]:
        """📋 Load Pascal VOC format annotations."""
        annotations = []
        
        if not annotation_path.exists():
            return annotations
        
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                bbox_elem = obj.find('bndbox')
                
                if name_elem is not None and bbox_elem is not None:
                    label = name_elem.text
                    
                    xmin = float(bbox_elem.find('xmin').text)
                    ymin = float(bbox_elem.find('ymin').text)
                    xmax = float(bbox_elem.find('xmax').text)
                    ymax = float(bbox_elem.find('ymax').text)
                    
                    # Convert to (x, y, width, height) format
                    x = xmin
                    y = ymin
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    annotations.append({
                        'bbox': [x, y, width, height],
                        'label': label
                    })
        
        except ET.ParseError as e:
            print(f"❌ Error parsing XML: {str(e)}")
        
        return annotations
    
    def preview_dataset_sample(self, dataset_path: Path, num_samples: int = 5):
        """📊 Preview multiple samples from a dataset."""
        if not MATPLOTLIB_AVAILABLE:
            print("❌ matplotlib no está disponible para preview múltiple")
            return
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f'**/*{ext}'))
        
        if not image_files:
            print("❌ No se encontraron imágenes en el dataset")
            return
        
        # Limit to requested number of samples
        sample_files = image_files[:num_samples]
        
        print(f"📊 Mostrando {len(sample_files)} muestras del dataset:")
        
        for i, image_file in enumerate(sample_files):
            print(f"\\n🖼️ Muestra {i+1}: {image_file.name}")
            
            # Try to find corresponding annotation file
            annotation_file = None
            
            # Try different annotation file patterns
            base_name = image_file.stem
            possible_annotations = [
                image_file.with_suffix('.txt'),  # YOLO format
                image_file.with_suffix('.xml'),  # Pascal VOC format
                dataset_path / 'labels' / f'{base_name}.txt',  # YOLO in labels dir
                dataset_path / 'annotations' / f'{base_name}.xml',  # Pascal in annotations dir
            ]
            
            for possible_ann in possible_annotations:
                if possible_ann.exists():
                    annotation_file = possible_ann
                    break
            
            if annotation_file:
                # Detect format based on extension
                if annotation_file.suffix == '.txt':
                    format_type = 'yolo'
                elif annotation_file.suffix == '.xml':
                    format_type = 'pascal_voc'
                else:
                    format_type = 'unknown'
                
                self.show_preview(image_file, annotation_file, format_type)
            else:
                print(f"⚠️ No se encontró archivo de anotación para {image_file.name}")
                
                # Show image without annotations
                try:
                    image = Image.open(image_file)
                    plt.figure(figsize=(8, 6))
                    plt.imshow(image)
                    plt.title(f"Image: {image_file.name} (No annotations)")
                    plt.axis('off')
                    plt.show()
                except Exception as e:
                    print(f"❌ Error mostrando imagen: {str(e)}")
    
    def generate_preview_report(self, dataset_path: Path, output_path: Path = None):
        """📊 Generate a preview report of the dataset."""
        if output_path is None:
            output_path = dataset_path / 'preview_report.txt'
        
        print(f"📊 Generando reporte de preview en: {output_path}")
        
        # Scan dataset
        report_lines = [
            "📊 DATASET PREVIEW REPORT",
            "=" * 50,
            f"📁 Dataset: {dataset_path.name}",
            f"📍 Path: {dataset_path}",
            f"📅 Generated: {datetime.now().isoformat()}",
            ""
        ]
        
        # Count files by type
        image_count = 0
        annotation_count = 0
        formats_found = set()
        
        for file_path in dataset_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    image_count += 1
                elif ext in ['.txt', '.xml', '.json']:
                    annotation_count += 1
                    if ext == '.txt':
                        formats_found.add('YOLO')
                    elif ext == '.xml':
                        formats_found.add('Pascal VOC')
                    elif ext == '.json':
                        formats_found.add('COCO')
        
        report_lines.extend([
            f"🖼️ Total images: {image_count}",
            f"📝 Total annotations: {annotation_count}",
            f"📋 Formats detected: {', '.join(formats_found) if formats_found else 'None'}",
            ""
        ])
        
        # Directory structure
        report_lines.append("📁 DIRECTORY STRUCTURE:")
        for item in sorted(dataset_path.iterdir()):
            if item.is_dir():
                item_count = len(list(item.rglob('*')))
                report_lines.append(f"  📁 {item.name}/ ({item_count} items)")
            else:
                report_lines.append(f"  📄 {item.name}")
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(report_lines))
        
        print(f"✅ Reporte generado exitosamente")
        
        # Also print to console
        for line in report_lines:
            print(line)
