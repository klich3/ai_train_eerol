"""
🔍 Dataset Scanner Module
========================

Universal dataset scanner that detects and analyzes datasets
in any directory structure.
"""

import os
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import xml.etree.ElementTree as ET

# Optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class DatasetScanner:
    """🔍 Universal dataset scanner."""
    
    def __init__(self, base_path: Path):
        """Initialize scanner with base path."""
        self.base_path = Path(base_path)
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.supported_annotation_formats = {'.txt', '.json', '.xml', '.csv'}
        
    def scan_datasets(self) -> Dict[str, Any]:
        """🔍 Scan all datasets in the base path."""
        print(f"\n🔍 ESCANEANDO DATASETS EN: {self.base_path}")
        print("=" * 50)
        
        results = {
            'datasets': [],
            'total_datasets': 0,
            'total_images': 0,
            'formats_found': Counter(),
            'categories_found': set(),
            'scan_path': str(self.base_path)
        }
        
        # Scan directories recursively but not too deep
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                dataset_info = self._analyze_directory(item)
                if dataset_info['is_dataset']:
                    results['datasets'].append(dataset_info)
                    results['total_images'] += dataset_info['total_images']
                    results['formats_found'][dataset_info['format']] += 1
                    results['categories_found'].update(dataset_info['categories'])
        
        results['total_datasets'] = len(results['datasets'])
        results['categories_found'] = list(results['categories_found'])
        
        return results
    
    def _analyze_directory(self, dir_path: Path) -> Dict[str, Any]:
        """🔬 Analyze a single directory to determine if it's a dataset."""
        dataset_info = {
            'name': dir_path.name,
            'path': str(dir_path),
            'is_dataset': False,
            'format': 'unknown',
            'total_images': 0,
            'total_annotations': 0,
            'categories': set(),
            'splits': {},
            'config_files': [],
            'structure': {}
        }
        
        # Get directory contents
        contents = list(dir_path.iterdir())
        image_files = []
        annotation_files = []
        config_files = []
        subdirs = []
        
        for item in contents:
            if item.is_file():
                if item.suffix.lower() in self.supported_image_formats:
                    image_files.append(item)
                elif item.suffix.lower() in self.supported_annotation_formats:
                    annotation_files.append(item)
                elif item.name.lower() in ['config.yaml', 'config.yml', 'data.yaml', 
                                         'data.yml', 'classes.txt', 'labels.txt']:
                    config_files.append(item)
            elif item.is_dir() and not item.name.startswith('.'):
                subdirs.append(item)
        
        # Check if this looks like a dataset
        has_images = len(image_files) > 0
        has_annotations = len(annotation_files) > 0
        has_typical_structure = any(subdir.name.lower() in ['train', 'test', 'val', 'valid', 
                                                          'images', 'labels', 'annotations'] 
                                  for subdir in subdirs)
        
        if has_images or has_typical_structure:
            dataset_info['is_dataset'] = True
            dataset_info['total_images'] = len(image_files)
            dataset_info['total_annotations'] = len(annotation_files)
            dataset_info['config_files'] = [str(f) for f in config_files]
            
            # Determine format
            dataset_info['format'] = self._detect_format(dir_path, annotation_files, config_files)
            
            # Analyze subdirectories
            for subdir in subdirs:
                split_info = self._analyze_split_directory(subdir)
                if split_info['images'] > 0:
                    dataset_info['splits'][subdir.name] = split_info
                    dataset_info['total_images'] += split_info['images']
                    dataset_info['total_annotations'] += split_info['annotations']
            
            # Extract categories
            dataset_info['categories'] = self._extract_categories(dir_path, config_files, annotation_files)
            
            # Build structure summary
            dataset_info['structure'] = self._build_structure_summary(dir_path)
        
        return dataset_info
    
    def _analyze_split_directory(self, split_path: Path) -> Dict[str, int]:
        """📊 Analyze a split directory (train/test/val)."""
        split_info = {
            'images': 0,
            'annotations': 0
        }
        
        if not split_path.is_dir():
            return split_info
        
        # Check direct files
        for item in split_path.iterdir():
            if item.is_file():
                if item.suffix.lower() in self.supported_image_formats:
                    split_info['images'] += 1
                elif item.suffix.lower() in self.supported_annotation_formats:
                    split_info['annotations'] += 1
            elif item.is_dir():
                # Check subdirectories (like images/ and labels/)
                for subitem in item.iterdir():
                    if subitem.is_file():
                        if subitem.suffix.lower() in self.supported_image_formats:
                            split_info['images'] += 1
                        elif subitem.suffix.lower() in self.supported_annotation_formats:
                            split_info['annotations'] += 1
        
        return split_info
    
    def _detect_format(self, dir_path: Path, annotation_files: List[Path], 
                      config_files: List[Path]) -> str:
        """🔍 Detect dataset format."""
        # Check config files first
        for config_file in config_files:
            if config_file.name.lower() in ['data.yaml', 'data.yml', 'config.yaml', 'config.yml']:
                try:
                    if YAML_AVAILABLE:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f)
                            if isinstance(config, dict):
                                if 'names' in config or 'nc' in config:
                                    return 'yolo'
                except:
                    pass
        
        # Check annotation files
        if annotation_files:
            sample_file = annotation_files[0]
            if sample_file.suffix.lower() == '.txt':
                # Check if it's YOLO format
                try:
                    with open(sample_file, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5 and all(self._is_float(p) for p in parts[1:5]):
                                return 'yolo'
                except:
                    pass
            elif sample_file.suffix.lower() == '.json':
                try:
                    with open(sample_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            if 'images' in data and 'annotations' in data and 'categories' in data:
                                return 'coco'
                except:
                    pass
            elif sample_file.suffix.lower() == '.xml':
                try:
                    tree = ET.parse(sample_file)
                    root = tree.getroot()
                    if root.tag == 'annotation':
                        return 'pascal_voc'
                except:
                    pass
        
        # Check directory structure
        subdirs = [d.name.lower() for d in dir_path.iterdir() if d.is_dir()]
        if 'images' in subdirs and 'labels' in subdirs:
            return 'yolo'
        elif 'annotations' in subdirs:
            return 'coco'
        
        return 'unknown'
    
    def _extract_categories(self, dir_path: Path, config_files: List[Path], 
                          annotation_files: List[Path]) -> set:
        """🏷️ Extract categories from dataset."""
        categories = set()
        
        # Try config files first
        for config_file in config_files:
            try:
                if config_file.suffix.lower() in ['.yaml', '.yml'] and YAML_AVAILABLE:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        if isinstance(config, dict) and 'names' in config:
                            if isinstance(config['names'], list):
                                categories.update(config['names'])
                            elif isinstance(config['names'], dict):
                                categories.update(config['names'].values())
                elif config_file.name.lower() in ['classes.txt', 'labels.txt']:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        categories.update(line.strip() for line in f if line.strip())
            except:
                continue
        
        # If no categories found, try to extract from annotations
        if not categories and annotation_files:
            categories = self._extract_categories_from_annotations(annotation_files[:10])  # Sample first 10
        
        return categories
    
    def _extract_categories_from_annotations(self, annotation_files: List[Path]) -> set:
        """🔍 Extract categories from annotation files."""
        categories = set()
        class_indices = set()
        
        for ann_file in annotation_files:
            try:
                if ann_file.suffix.lower() == '.txt':
                    # YOLO format
                    with open(ann_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and self._is_int(parts[0]):
                                class_indices.add(int(parts[0]))
                elif ann_file.suffix.lower() == '.json':
                    # COCO format
                    with open(ann_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'categories' in data:
                            for cat in data['categories']:
                                if 'name' in cat:
                                    categories.add(cat['name'])
                elif ann_file.suffix.lower() == '.xml':
                    # Pascal VOC format
                    tree = ET.parse(ann_file)
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        name_elem = obj.find('name')
                        if name_elem is not None:
                            categories.add(name_elem.text)
            except:
                continue
        
        # If we only have class indices, generate generic names
        if class_indices and not categories:
            categories = {f"class_{i}" for i in class_indices}
        
        return categories
    
    def _build_structure_summary(self, dir_path: Path) -> Dict[str, Any]:
        """🏗️ Build structure summary."""
        structure = {
            'type': 'flat',
            'has_splits': False,
            'directories': [],
            'files': []
        }
        
        for item in dir_path.iterdir():
            if item.is_dir():
                structure['directories'].append(item.name)
                if item.name.lower() in ['train', 'test', 'val', 'valid']:
                    structure['has_splits'] = True
                    structure['type'] = 'split'
            else:
                structure['files'].append(item.name)
        
        return structure
    
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
    
    def display_results(self, results: Dict[str, Any]):
        """📊 Display scan results."""
        print(f"\n📊 RESULTADOS DEL ESCANEO")
        print("=" * 50)
        print(f"📁 Directorio escaneado: {results['scan_path']}")
        print(f"📊 Datasets encontrados: {results['total_datasets']}")
        print(f"🖼️ Imágenes totales: {results['total_images']}")
        
        if results['formats_found']:
            print(f"\n📋 FORMATOS DETECTADOS:")
            for format_type, count in results['formats_found'].items():
                print(f"  {format_type}: {count} datasets")
        
        if results['categories_found']:
            print(f"\n🏷️ CATEGORÍAS ENCONTRADAS: {len(results['categories_found'])}")
            categories_preview = list(results['categories_found'])[:10]
            print(f"  {', '.join(categories_preview)}")
            if len(results['categories_found']) > 10:
                print(f"  ... y {len(results['categories_found']) - 10} más")
        
        if results['datasets']:
            print(f"\n📁 DETALLES DE DATASETS:")
            print("-" * 50)
            for dataset in results['datasets']:
                print(f"\n📂 {dataset['name']}")
                print(f"   📍 Ruta: {dataset['path']}")
                print(f"   📊 Formato: {dataset['format']}")
                print(f"   🖼️ Imágenes: {dataset['total_images']}")
                print(f"   📝 Anotaciones: {dataset['total_annotations']}")
                
                if dataset['splits']:
                    print(f"   📁 Divisiones:")
                    for split_name, split_info in dataset['splits'].items():
                        print(f"      {split_name}: {split_info['images']} imágenes, {split_info['annotations']} anotaciones")
                
                if dataset['categories']:
                    categories_preview = list(dataset['categories'])[:5]
                    print(f"   🏷️ Categorías ({len(dataset['categories'])}): {', '.join(categories_preview)}")
                    if len(dataset['categories']) > 5:
                        print(f"      ... y {len(dataset['categories']) - 5} más")
                
                if dataset['config_files']:
                    print(f"   ⚙️ Archivos de configuración: {len(dataset['config_files'])}")
        else:
            print("\n❌ No se encontraron datasets en el directorio especificado.")
            print("💡 Asegúrate de que el directorio contenga:")
            print("   - Imágenes con extensiones: .jpg, .jpeg, .png, .bmp, .tiff")
            print("   - Anotaciones con extensiones: .txt, .json, .xml")
            print("   - O carpetas con nombres como: train, test, val, images, labels")
        
        print(f"\n✅ Escaneo completado.")
