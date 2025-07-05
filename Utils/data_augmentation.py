"""
🔧 Data Augmentation Utilities
Utilidades para augmentación de datos dentales
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import List, Tuple, Dict, Any
import random


class DentalDataAugmenter:
    """Augmentador de datos específico para imágenes dentales."""
    
    def __init__(self):
        # Transformaciones específicas para radiografías dentales
        self.dental_transforms = A.Compose([
            # Transformaciones geométricas suaves
            A.RandomRotate90(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.HorizontalFlip(p=0.5),
            
            # Ajustes de contraste y brillo (importante para radiografías)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            
            # Gamma correction (útil para radiografías)
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Blur suave para simular diferentes calidades de imagen
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.2),
            
            # Ruido para simular diferentes equipos
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            ], p=0.2),
            
            # Distorsiones ópticas suaves
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.2),
            
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def augment_image_with_bboxes(self, image: np.ndarray, bboxes: List[List], 
                                 class_labels: List[int]) -> Tuple[np.ndarray, List[List], List[int]]:
        """Aplica augmentación a imagen con bounding boxes."""
        try:
            transformed = self.dental_transforms(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            return (
                transformed['image'],
                transformed['bboxes'],
                transformed['class_labels']
            )
        except Exception as e:
            print(f"⚠️ Error en augmentación: {e}")
            return image, bboxes, class_labels
    
    def augment_classification_image(self, image: np.ndarray) -> np.ndarray:
        """Aplica augmentación a imagen de clasificación."""
        # Transformaciones sin bounding boxes
        simple_transforms = A.Compose([
            A.RandomRotate90(p=0.2),
            A.Rotate(limit=15, p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        ])
        
        try:
            return simple_transforms(image=image)['image']
        except Exception as e:
            print(f"⚠️ Error en augmentación de clasificación: {e}")
            return image


class DataBalancer:
    """Balanceador de datasets."""
    
    def __init__(self, target_samples_per_class: int = 1000):
        self.target_samples = target_samples_per_class
        self.augmenter = DentalDataAugmenter()
    
    def balance_yolo_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Balancea un dataset YOLO usando augmentación."""
        print(f"⚖️ Balanceando dataset YOLO: {dataset_path}")
        
        # Analizar distribución actual de clases
        class_distribution = self._analyze_yolo_distribution(dataset_path)
        
        # Determinar clases que necesitan augmentación
        stats = {
            'original_distribution': class_distribution,
            'augmented_samples': {},
            'total_augmented': 0
        }
        
        max_samples = max(class_distribution.values()) if class_distribution else self.target_samples
        target = min(max_samples * 2, self.target_samples)  # No exceder 2x del máximo actual
        
        for class_id, count in class_distribution.items():
            if count < target:
                needed = target - count
                augmented = self._augment_yolo_class(dataset_path, class_id, needed)
                stats['augmented_samples'][class_id] = augmented
                stats['total_augmented'] += augmented
        
        return stats
    
    def _analyze_yolo_distribution(self, dataset_path: Path) -> Dict[str, int]:
        """Analiza la distribución de clases en dataset YOLO."""
        distribution = {}
        
        # Buscar archivos de anotación
        for split in ['train', 'val']:
            labels_dir = dataset_path / split / 'labels'
            if labels_dir.exists():
                for ann_file in labels_dir.glob('*.txt'):
                    with open(ann_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = parts[0]
                                distribution[class_id] = distribution.get(class_id, 0) + 1
        
        return distribution
    
    def _augment_yolo_class(self, dataset_path: Path, class_id: str, needed_samples: int) -> int:
        """Augmenta muestras de una clase específica en YOLO."""
        augmented_count = 0
        
        # Encontrar imágenes que contienen esta clase
        class_images = self._find_images_with_class(dataset_path, class_id)
        
        if not class_images:
            return 0
        
        # Generar muestras augmentadas
        while augmented_count < needed_samples and augmented_count < 500:  # Límite de seguridad
            # Seleccionar imagen aleatoria
            img_path, ann_path = random.choice(class_images)
            
            try:
                # Cargar imagen y anotaciones
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                bboxes, class_labels = self._load_yolo_annotations(ann_path)
                
                # Aplicar augmentación
                aug_image, aug_bboxes, aug_labels = self.augmenter.augment_image_with_bboxes(
                    image, bboxes, class_labels
                )
                
                # Guardar imagen y anotación augmentada
                self._save_augmented_yolo_sample(
                    dataset_path, aug_image, aug_bboxes, aug_labels, 
                    f"aug_{class_id}_{augmented_count}", img_path.parent.parent.name
                )
                
                augmented_count += 1
                
            except Exception as e:
                print(f"⚠️ Error augmentando muestra: {e}")
                continue
        
        return augmented_count
    
    def _find_images_with_class(self, dataset_path: Path, class_id: str) -> List[Tuple[Path, Path]]:
        """Encuentra imágenes que contienen una clase específica."""
        images_with_class = []
        
        for split in ['train', 'val']:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                for ann_file in labels_dir.glob('*.txt'):
                    # Verificar si contiene la clase
                    with open(ann_file, 'r') as f:
                        if any(line.strip().startswith(class_id + ' ') for line in f):
                            img_file = images_dir / f"{ann_file.stem}.jpg"
                            if not img_file.exists():
                                img_file = images_dir / f"{ann_file.stem}.png"
                            
                            if img_file.exists():
                                images_with_class.append((img_file, ann_file))
        
        return images_with_class
    
    def _load_yolo_annotations(self, ann_path: Path) -> Tuple[List[List], List[int]]:
        """Carga anotaciones YOLO."""
        bboxes = []
        class_labels = []
        
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    class_labels.append(class_id)
                    bboxes.append(bbox)
        
        return bboxes, class_labels
    
    def _save_augmented_yolo_sample(self, dataset_path: Path, image: np.ndarray, 
                                   bboxes: List[List], class_labels: List[int], 
                                   base_name: str, split: str):
        """Guarda una muestra augmentada en formato YOLO."""
        # Guardar imagen
        img_path = dataset_path / split / 'images' / f"{base_name}.jpg"
        cv2.imwrite(str(img_path), image)
        
        # Guardar anotaciones
        ann_path = dataset_path / split / 'labels' / f"{base_name}.txt"
        with open(ann_path, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                f.write(f"{class_id} {' '.join(map(str, bbox))}\n")


class QualityChecker:
    """Verificador de calidad de datasets."""
    
    def check_dataset_quality(self, dataset_path: Path, format_type: str) -> Dict[str, Any]:
        """Verifica la calidad de un dataset."""
        quality_report = {
            'dataset_path': str(dataset_path),
            'format': format_type,
            'issues': [],
            'quality_score': 0.0,
            'recommendations': []
        }
        
        if format_type == 'YOLO':
            self._check_yolo_quality(dataset_path, quality_report)
        elif format_type == 'COCO':
            self._check_coco_quality(dataset_path, quality_report)
        elif format_type == 'Classification':
            self._check_classification_quality(dataset_path, quality_report)
        
        return quality_report
    
    def _check_yolo_quality(self, dataset_path: Path, report: Dict):
        """Verifica calidad de dataset YOLO."""
        score = 100.0
        
        # Verificar estructura de directorios
        required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
        for dir_path in required_dirs:
            if not (dataset_path / dir_path).exists():
                report['issues'].append(f"Directorio faltante: {dir_path}")
                score -= 10
        
        # Verificar archivo data.yaml
        if not (dataset_path / 'data.yaml').exists():
            report['issues'].append("Falta archivo data.yaml")
            report['recommendations'].append("Crear archivo data.yaml con configuración")
            score -= 5
        
        # Verificar correspondencia imagen-anotación
        for split in ['train', 'val']:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'
            
            if images_dir.exists() and labels_dir.exists():
                img_files = set(f.stem for f in images_dir.glob('*.[jp][pn]g'))
                ann_files = set(f.stem for f in labels_dir.glob('*.txt'))
                
                missing_annotations = img_files - ann_files
                orphan_annotations = ann_files - img_files
                
                if missing_annotations:
                    report['issues'].append(f"Imágenes sin anotación en {split}: {len(missing_annotations)}")
                    score -= len(missing_annotations) * 0.1
                
                if orphan_annotations:
                    report['issues'].append(f"Anotaciones huérfanas en {split}: {len(orphan_annotations)}")
                    score -= len(orphan_annotations) * 0.05
        
        report['quality_score'] = max(0.0, score)
    
    def _check_coco_quality(self, dataset_path: Path, report: Dict):
        """Verifica calidad de dataset COCO."""
        score = 100.0
        
        # Buscar archivo JSON principal
        json_files = list(dataset_path.glob('*.json')) + list(dataset_path.glob('annotations/*.json'))
        
        if not json_files:
            report['issues'].append("No se encontró archivo JSON de anotaciones")
            score -= 50
        else:
            # Verificar estructura JSON
            try:
                with open(json_files[0], 'r') as f:
                    coco_data = json.load(f)
                
                required_keys = ['images', 'annotations', 'categories']
                for key in required_keys:
                    if key not in coco_data:
                        report['issues'].append(f"Clave faltante en JSON: {key}")
                        score -= 15
                
            except Exception as e:
                report['issues'].append(f"Error leyendo JSON: {e}")
                score -= 30
        
        report['quality_score'] = max(0.0, score)
    
    def _check_classification_quality(self, dataset_path: Path, report: Dict):
        """Verifica calidad de dataset de clasificación."""
        score = 100.0
        
        # Verificar estructura de carpetas
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if len(class_dirs) < 2:
            report['issues'].append("Dataset necesita al menos 2 clases")
            score -= 30
        
        # Verificar distribución de clases
        class_counts = {}
        for class_dir in class_dirs:
            img_count = len(list(class_dir.glob('*.[jp][pn]g')))
            class_counts[class_dir.name] = img_count
            
            if img_count < 10:
                report['issues'].append(f"Clase '{class_dir.name}' tiene muy pocas muestras: {img_count}")
                score -= 5
        
        # Verificar balance entre clases
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            
            if max_count > min_count * 10:  # Desbalance mayor a 10:1
                report['issues'].append("Dataset muy desbalanceado")
                report['recommendations'].append("Considerar augmentación de clases minoritarias")
                score -= 10
        
        report['quality_score'] = max(0.0, score)
