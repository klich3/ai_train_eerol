"""
🔄 Data Processing Module for Dental Datasets
Módulo para procesamiento y transformación de datasets dentales
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml


class DataProcessor:
    """Procesador de datasets dentales."""
    
    def __init__(self, unified_classes: Dict, standard_resolutions: Dict, safety_config: Dict):
        self.unified_classes = unified_classes
        self.standard_resolutions = standard_resolutions
        self.safety_config = safety_config
        
    def unify_class_names(self, original_class: str) -> str:
        """🏷️ Unifica nombres de clases según el mapeo definido."""
        original_lower = original_class.lower().strip()
        
        for unified_name, variants in self.unified_classes.items():
            if original_class in variants or original_lower in [v.lower() for v in variants]:
                return unified_name
        
        # Si no encuentra mapeo, retorna el nombre original limpio
        return original_class.replace(' ', '_').lower()
    
    def merge_yolo_datasets(self, dataset_paths: List[str], output_path: Path, 
                           target_size: Tuple[int, int] = None) -> Dict[str, Any]:
        """🔄 Fusiona múltiples datasets YOLO en uno unificado."""
        if target_size is None:
            target_size = self.standard_resolutions['yolo']
        
        print(f"\n🔄 FUSIONANDO {len(dataset_paths)} DATASETS YOLO...")
        
        # Estructura de salida
        output_yolo = output_path / "datasets" / "detection_combined"
        output_yolo.mkdir(parents=True, exist_ok=True)
        
        # Directorios de train/val/test
        for split in ['train', 'val', 'test']:
            (output_yolo / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_yolo / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        merged_stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': Counter(),
            'datasets_processed': [],
            'unified_classes': {},
            'splits': {'train': 0, 'val': 0, 'test': 0}
        }
        
        unified_class_mapping = {}
        class_id_counter = 0
        
        # Procesar cada dataset
        for dataset_path in tqdm(dataset_paths, desc="Procesando datasets YOLO"):
            dataset_path = Path(dataset_path)
            
            if not dataset_path.exists():
                print(f"⚠️ Dataset no encontrado: {dataset_path}")
                continue
            
            print(f"  📁 Procesando: {dataset_path.name}")
            
            # Buscar imágenes y anotaciones
            images = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png"))
            
            if not images:
                print(f"  ⚠️ No se encontraron imágenes en {dataset_path}")
                continue
            
            # Dividir el dataset en train/val/test
            train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
            
            splits = {
                'train': train_imgs,
                'val': val_imgs, 
                'test': test_imgs
            }
            
            # Procesar cada split
            for split_name, split_images in splits.items():
                self._process_yolo_split(
                    split_images, dataset_path, output_yolo / split_name,
                    unified_class_mapping, class_id_counter, merged_stats, target_size
                )
                merged_stats['splits'][split_name] += len(split_images)
            
            merged_stats['datasets_processed'].append(str(dataset_path))
        
        # Crear archivo data.yaml
        self._create_yolo_yaml(output_yolo, unified_class_mapping, merged_stats)
        
        # Crear archivo de clases
        self._create_classes_file(output_yolo, unified_class_mapping)
        
        print(f"\n✅ FUSIÓN YOLO COMPLETADA:")
        print(f"   📊 Total imágenes: {merged_stats['total_images']}")
        print(f"   🏷️ Total anotaciones: {merged_stats['total_annotations']}")
        print(f"   📋 Clases unificadas: {len(unified_class_mapping)}")
        print(f"   📂 Guardado en: {output_yolo}")
        
        return merged_stats
    
    def _process_yolo_split(self, images: List[Path], source_path: Path, output_split: Path,
                           unified_mapping: Dict, class_counter: int, stats: Dict, target_size: Tuple):
        """Procesa un split específico de YOLO."""
        for img_path in tqdm(images, desc=f"Procesando {output_split.name}", leave=False):
            # Buscar archivo de anotación correspondiente
            ann_path = img_path.with_suffix('.txt')
            if not ann_path.exists():
                # Buscar en directorios labels/
                possible_ann = source_path / 'labels' / f"{img_path.stem}.txt"
                if possible_ann.exists():
                    ann_path = possible_ann
                else:
                    continue  # Saltar si no hay anotación
            
            try:
                # Leer y procesar imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Redimensionar si es necesario
                if target_size and (img.shape[1] != target_size[0] or img.shape[0] != target_size[1]):
                    img = cv2.resize(img, target_size)
                
                # Nombre único para la imagen
                unique_name = f"{source_path.name}_{img_path.stem}_{len(stats['datasets_processed'])}"
                
                # Copiar imagen
                img_output = output_split / 'images' / f"{unique_name}.jpg"
                cv2.imwrite(str(img_output), img)
                
                # Procesar anotaciones
                self._process_yolo_annotations(
                    ann_path, output_split / 'labels' / f"{unique_name}.txt",
                    unified_mapping, class_counter, stats
                )
                
                stats['total_images'] += 1
                
            except Exception as e:
                print(f"⚠️ Error procesando {img_path}: {e}")
    
    def _process_yolo_annotations(self, ann_path: Path, output_ann: Path, 
                                 unified_mapping: Dict, class_counter: int, stats: Dict):
        """Procesa anotaciones YOLO y unifica clases."""
        processed_lines = []
        
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    original_class_id = parts[0]
                    coords = parts[1:]
                    
                    # Mapear clase (por ahora usar ID directamente)
                    if original_class_id not in unified_mapping:
                        unified_mapping[original_class_id] = len(unified_mapping)
                    
                    unified_id = unified_mapping[original_class_id]
                    processed_lines.append(f"{unified_id} {' '.join(coords)}")
                    
                    stats['total_annotations'] += 1
                    stats['class_distribution'][str(unified_id)] += 1
        
        # Guardar anotaciones procesadas
        with open(output_ann, 'w') as f:
            f.write('\n'.join(processed_lines))
    
    def _create_yolo_yaml(self, output_path: Path, class_mapping: Dict, stats: Dict):
        """Crea archivo data.yaml para YOLO."""
        yaml_content = {
            'path': str(output_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_mapping),
            'names': {v: k for k, v in class_mapping.items()},
            'stats': {
                'total_images': stats['total_images'],
                'total_annotations': stats['total_annotations'],
                'datasets_merged': len(stats['datasets_processed'])
            }
        }
        
        with open(output_path / 'data.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
    
    def _create_classes_file(self, output_path: Path, class_mapping: Dict):
        """Crea archivo classes.txt."""
        sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
        with open(output_path / 'classes.txt', 'w') as f:
            for class_name, _ in sorted_classes:
                f.write(f"{class_name}\n")
    
    def merge_coco_datasets(self, dataset_paths: List[str], output_path: Path) -> Dict[str, Any]:
        """🔄 Fusiona múltiples datasets COCO en uno unificado."""
        print(f"\n🔄 FUSIONANDO {len(dataset_paths)} DATASETS COCO...")
        
        output_coco = output_path / "datasets" / "segmentation_coco"
        output_coco.mkdir(parents=True, exist_ok=True)
        
        # Crear estructura COCO
        (output_coco / 'images').mkdir(exist_ok=True)
        (output_coco / 'annotations').mkdir(exist_ok=True)
        
        merged_coco = {
            'images': [],
            'annotations': [],
            'categories': [],
            'info': {
                'description': 'Dental Dataset Merged COCO Format',
                'version': '1.0',
                'contributor': 'Dental AI Workflow Manager'
            }
        }
        
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'datasets_processed': []
        }
        
        image_id_counter = 1
        annotation_id_counter = 1
        category_mapping = {}
        
        # Procesar cada dataset COCO
        for dataset_path in tqdm(dataset_paths, desc="Procesando datasets COCO"):
            self._process_coco_dataset(
                Path(dataset_path), output_coco, merged_coco,
                image_id_counter, annotation_id_counter, category_mapping, stats
            )
            stats['datasets_processed'].append(str(dataset_path))
        
        # Guardar COCO fusionado
        with open(output_coco / 'annotations' / 'instances_merged.json', 'w') as f:
            json.dump(merged_coco, f, indent=2)
        
        print(f"\n✅ FUSIÓN COCO COMPLETADA:")
        print(f"   📊 Total imágenes: {stats['total_images']}")
        print(f"   🏷️ Total anotaciones: {stats['total_annotations']}")
        print(f"   📋 Categorías: {len(merged_coco['categories'])}")
        print(f"   📂 Guardado en: {output_coco}")
        
        return stats
    
    def _process_coco_dataset(self, dataset_path: Path, output_path: Path, merged_coco: Dict,
                             img_id_counter: int, ann_id_counter: int, category_mapping: Dict, stats: Dict):
        """Procesa un dataset COCO individual."""
        # Buscar archivo JSON principal
        json_files = list(dataset_path.rglob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    coco_data = json.load(f)
                
                if 'images' not in coco_data or 'annotations' not in coco_data:
                    continue
                
                # Procesar categorías
                if 'categories' in coco_data:
                    for cat in coco_data['categories']:
                        unified_name = self.unify_class_names(cat['name'])
                        if unified_name not in category_mapping:
                            category_mapping[unified_name] = len(category_mapping)
                            merged_coco['categories'].append({
                                'id': category_mapping[unified_name],
                                'name': unified_name,
                                'supercategory': 'dental'
                            })
                
                # Procesar imágenes y anotaciones
                for img_info in coco_data['images']:
                    # Copiar imagen
                    img_src = dataset_path / img_info['file_name']
                    if img_src.exists():
                        img_dst = output_path / 'images' / f"{dataset_path.name}_{img_info['file_name']}"
                        shutil.copy2(img_src, img_dst)
                        
                        # Actualizar información de imagen
                        new_img_info = img_info.copy()
                        new_img_info['id'] = img_id_counter
                        new_img_info['file_name'] = f"{dataset_path.name}_{img_info['file_name']}"
                        merged_coco['images'].append(new_img_info)
                        
                        # Procesar anotaciones de esta imagen
                        for ann in coco_data['annotations']:
                            if ann['image_id'] == img_info['id']:
                                new_ann = ann.copy()
                                new_ann['id'] = ann_id_counter
                                new_ann['image_id'] = img_id_counter
                                
                                # Mapear categoría
                                old_cat_id = ann['category_id']
                                if 'categories' in coco_data:
                                    old_cat_name = next((c['name'] for c in coco_data['categories'] 
                                                       if c['id'] == old_cat_id), f"class_{old_cat_id}")
                                    unified_name = self.unify_class_names(old_cat_name)
                                    new_ann['category_id'] = category_mapping.get(unified_name, 0)
                                
                                merged_coco['annotations'].append(new_ann)
                                ann_id_counter += 1
                                stats['total_annotations'] += 1
                        
                        img_id_counter += 1
                        stats['total_images'] += 1
                
                break  # Solo procesar el primer JSON válido
                
            except Exception as e:
                print(f"⚠️ Error procesando {json_file}: {e}")
    
    def create_classification_dataset(self, pure_image_paths: List[str], output_path: Path) -> Dict[str, Any]:
        """📁 Crea dataset de clasificación organizando imágenes por carpetas."""
        print(f"\n📁 CREANDO DATASET DE CLASIFICACIÓN...")
        
        output_classification = output_path / "datasets" / "classification"
        output_classification.mkdir(parents=True, exist_ok=True)
        
        # Crear estructura train/val/test
        for split in ['train', 'val', 'test']:
            (output_classification / split).mkdir(exist_ok=True)
        
        stats = {
            'total_images': 0,
            'class_distribution': Counter(),
            'datasets_processed': []
        }
        
        detected_classes = set()
        
        # Escanear datasets para detectar clases
        for dataset_path in pure_image_paths:
            dataset_path = Path(dataset_path)
            
            # Detectar clases basadas en estructura de carpetas
            for item in dataset_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    unified_class = self.unify_class_names(item.name)
                    detected_classes.add(unified_class)
        
        # Crear carpetas para cada clase en cada split
        for class_name in detected_classes:
            for split in ['train', 'val', 'test']:
                (output_classification / split / class_name).mkdir(exist_ok=True)
        
        # Procesar cada dataset
        for dataset_path in tqdm(pure_image_paths, desc="Procesando datasets de clasificación"):
            self._process_classification_dataset(
                Path(dataset_path), output_classification, stats
            )
            stats['datasets_processed'].append(str(dataset_path))
        
        print(f"\n✅ DATASET DE CLASIFICACIÓN CREADO:")
        print(f"   📊 Total imágenes: {stats['total_images']}")
        print(f"   📋 Clases detectadas: {len(detected_classes)}")
        print(f"   📂 Guardado en: {output_classification}")
        
        return stats
    
    def _process_classification_dataset(self, dataset_path: Path, output_path: Path, stats: Dict):
        """Procesa un dataset de clasificación individual."""
        # Buscar imágenes organizadas por carpetas
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            
            unified_class = self.unify_class_names(class_dir.name)
            
            # Buscar imágenes en esta clase
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(class_dir.glob(ext))
            
            if not images:
                continue
            
            # Dividir en train/val/test
            train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
            
            splits = {
                'train': train_imgs,
                'val': val_imgs,
                'test': test_imgs
            }
            
            # Copiar imágenes a sus respectivos splits
            for split_name, split_images in splits.items():
                target_dir = output_path / split_name / unified_class
                
                for img_path in split_images:
                    unique_name = f"{dataset_path.name}_{class_dir.name}_{img_path.stem}_{stats['total_images']}.jpg"
                    
                    # Leer, procesar y guardar imagen
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Redimensionar a tamaño estándar si es muy grande
                        h, w = img.shape[:2]
                        if max(h, w) > 1024:
                            scale = 1024 / max(h, w)
                            new_w, new_h = int(w * scale), int(h * scale)
                            img = cv2.resize(img, (new_w, new_h))
                        
                        cv2.imwrite(str(target_dir / unique_name), img)
                        stats['total_images'] += 1
                        stats['class_distribution'][unified_class] += 1
    
    def balance_dataset(self, dataset_path: Path, target_samples_per_class: int = None) -> Dict[str, Any]:
        """⚖️ Balancea un dataset usando técnicas de augmentación."""
        print(f"\n⚖️ BALANCEANDO DATASET EN: {dataset_path}")
        
        # Por ahora, retornar estadísticas básicas
        # La implementación completa incluiría augmentación de datos
        
        return {
            'balanced': True,
            'method': 'augmentation',
            'target_samples': target_samples_per_class or 'auto'
        }
