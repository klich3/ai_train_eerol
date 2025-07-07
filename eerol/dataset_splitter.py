"""
✂️ Dataset Splitter Module
==========================

Universal dataset splitter for creating train/validation/test splits
with custom proportions.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class DatasetSplitter:
    """✂️ Universal dataset splitter."""
    
    def __init__(self, input_path: Path, output_path: Path):
        """Initialize splitter with input and output paths."""
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
    def split_dataset(self, output_name: str, train_ratio: float = 0.7, 
                     val_ratio: float = 0.3, test_ratio: float = 0.0) -> Dict[str, Any]:
        """✂️ Split dataset with specified proportions."""
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            return {
                'success': False,
                'error': f"Las proporciones deben sumar 1.0, actual: {total_ratio}"
            }
        
        print(f"✂️ Dividiendo dataset con proporciones:")
        print(f"   🏃 Train: {train_ratio:.1%}")
        print(f"   ✅ Validation: {val_ratio:.1%}")
        if test_ratio > 0:
            print(f"   🧪 Test: {test_ratio:.1%}")
        
        # Create output directory
        output_dir = self.output_path / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Detect dataset format and structure
            dataset_info = self._analyze_dataset_structure()
            
            print(f"🔍 Estructura detectada: {dataset_info['structure']}")
            print(f"📝 Formato detectado: {dataset_info['format']}")
            print(f"🖼️ Imágenes encontradas: {len(dataset_info['image_files'])}")
            print(f"📄 Archivos de anotación: {len(dataset_info['annotation_files'])}")
            if dataset_info['special_files']:
                print(f"✨ Archivos especiales: {list(dataset_info['special_files'].keys())}")
            
            if dataset_info['format'] == 'unknown':
                return {
                    'success': False,
                    'error': 'No se pudo determinar el formato del dataset'
                }
            
            # Perform split based on format and structure
            if dataset_info.get('yaml_config'):
                # Special handling for YAML-configured datasets
                result = self._split_yaml_dataset(output_dir, dataset_info, train_ratio, val_ratio, test_ratio)
            elif dataset_info['structure'] == 'flat':
                result = self._split_flat_dataset(output_dir, dataset_info, train_ratio, val_ratio, test_ratio)
            elif dataset_info['structure'] == 'split':
                result = self._merge_and_split_dataset(output_dir, dataset_info, train_ratio, val_ratio, test_ratio)
            else:
                result = self._split_nested_dataset(output_dir, dataset_info, train_ratio, val_ratio, test_ratio)
            
            if result['success']:
                # Generate configuration file
                self._generate_config_file(output_dir, dataset_info, result)
                
                return {
                    'success': True,
                    'output_path': str(output_dir),
                    'train_count': result['train_count'],
                    'val_count': result['val_count'],
                    'test_count': result['test_count'],
                    'format': dataset_info['format']
                }
            else:
                return result
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Error durante la división: {str(e)}"
            }
    
    def _analyze_dataset_structure(self) -> Dict[str, Any]:
        """🔍 Analyze dataset structure and format."""
        info = {
            'format': 'unknown',
            'structure': 'unknown',
            'image_files': [],
            'annotation_files': [],
            'config_files': [],
            'subdirs': [],
            'classes': [],
            'special_files': {},
            'yaml_config': None
        }
        
        # Get directory contents
        contents = list(self.input_path.iterdir())
        
        for item in contents:
            if item.is_file():
                ext = item.suffix.lower()
                filename = item.name.lower()
                
                if ext in self.supported_image_formats:
                    info['image_files'].append(item)
                elif ext in ['.txt', '.json', '.xml', '.csv']:
                    info['annotation_files'].append(item)
                    # Check for specific annotation file patterns
                    if filename == '_annotations.csv':
                        info['format'] = 'tensorflow'
                        info['special_files']['annotations_csv'] = item
                    elif filename == '_annotations.coco.json':
                        info['format'] = 'coco'
                        info['special_files']['annotations_coco'] = item
                elif filename in ['data.yaml', 'data.yml', 'config.yaml', 'config.yml', 'classes.txt']:
                    info['config_files'].append(item)
                    # Load YAML config if it's a data.yaml
                    if filename in ['data.yaml', 'data.yml'] and YAML_AVAILABLE:
                        try:
                            with open(item, 'r', encoding='utf-8') as f:
                                yaml_data = yaml.safe_load(f)
                                if isinstance(yaml_data, dict):
                                    info['yaml_config'] = yaml_data
                                    info['format'] = 'yolo'  # YAML config usually means YOLO
                        except:
                            pass
            elif item.is_dir() and not item.name.startswith('.'):
                info['subdirs'].append(item)
        
        # If we have YAML config, analyze the paths specified in it
        if info['yaml_config']:
            yaml_structure = self._analyze_yaml_structure(info['yaml_config'], info)
            if yaml_structure:
                info.update(yaml_structure)
                return info
        
        # Determine structure first (fallback method)
        subdir_names = [d.name.lower() for d in info['subdirs']]
        
        if any(name in ['train', 'test', 'val', 'valid'] for name in subdir_names):
            info['structure'] = 'split'
        elif 'images' in subdir_names and 'labels' in subdir_names:
            info['structure'] = 'nested'
        elif info['image_files'] or info['annotation_files']:
            info['structure'] = 'flat'
        
        # Determine format based on structure and files
        info['format'] = self._detect_format(info)
        
        # Load classes
        info['classes'] = self._load_classes(info)
        
        return info
    
    def _detect_format(self, dataset_info: Dict[str, Any]) -> str:
        """🔍 Detect dataset format."""
        
        # Priority 1: Check for specific special files
        if 'annotations_csv' in dataset_info['special_files']:
            return 'tensorflow'
        if 'annotations_coco' in dataset_info['special_files']:
            return 'coco'
        
        # Priority 2: Check config files for YOLO
        for config_file in dataset_info['config_files']:
            if config_file.name.lower() in ['data.yaml', 'data.yml']:
                return 'yolo'
        
        # Priority 3: Check directory structure for YOLO (images/labels)
        subdir_names = [d.name.lower() for d in dataset_info['subdirs']]
        if 'images' in subdir_names and ('labels' in subdir_names or 'annotations' in subdir_names):
            return 'yolo'
        
        # Priority 4: Check annotation files content
        if dataset_info['annotation_files']:
            sample_file = dataset_info['annotation_files'][0]
            filename = sample_file.name.lower()
            
            # Check specific naming patterns
            if filename.endswith('_annotations.csv'):
                return 'tensorflow'
            elif filename.endswith('_annotations.coco.json') or filename == 'annotations.json':
                return 'coco'
            elif sample_file.suffix.lower() == '.json':
                # Try to check JSON structure for COCO
                try:
                    import json
                    with open(sample_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and all(key in data for key in ['images', 'annotations', 'categories']):
                            return 'coco'
                except:
                    pass
                return 'coco'  # Default for JSON
            elif sample_file.suffix.lower() == '.xml':
                return 'pascal_voc'
            elif sample_file.suffix.lower() == '.txt':
                # Check YOLO format
                try:
                    with open(sample_file, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5 and all(self._is_float(p) for p in parts[1:5]):
                                return 'yolo'
                except:
                    pass
            elif sample_file.suffix.lower() == '.csv':
                return 'tensorflow'
        
        # Priority 5: Check if it's a split structure with nested formats
        if dataset_info['structure'] == 'split':
            # Check subdirectories for format indicators
            for subdir in dataset_info['subdirs']:
                if subdir.name.lower() in ['train', 'test', 'val', 'valid']:
                    # Check what's inside this split
                    subdir_contents = list(subdir.iterdir())
                    subdir_subdirs = [item.name.lower() for item in subdir_contents if item.is_dir()]
                    
                    # If split contains images/labels structure -> YOLO
                    if 'images' in subdir_subdirs and ('labels' in subdir_subdirs or 'annotations' in subdir_subdirs):
                        return 'yolo'
                    
                    # Check for annotation files in split
                    for item in subdir_contents:
                        if item.is_file():
                            if item.name.lower() == '_annotations.csv':
                                return 'tensorflow'
                            elif item.name.lower() == '_annotations.coco.json':
                                return 'coco'
        
        # If we have images but no clear format indicators, default to generic
        if dataset_info['image_files']:
            return 'generic'
        
        return 'unknown'
    
    def _split_flat_dataset(self, output_dir: Path, dataset_info: Dict[str, Any], 
                           train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, Any]:
        """✂️ Split flat dataset structure."""
        
        # Create output structure
        self._create_split_structure(output_dir, test_ratio > 0)
        
        # Get image-annotation pairs
        pairs = self._get_image_annotation_pairs(dataset_info)
        
        if not pairs:
            return {'success': False, 'error': 'No se encontraron pares imagen-anotación'}
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        # Calculate split indices
        total_count = len(pairs)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        test_count = total_count - train_count - val_count
        
        # Split pairs
        train_pairs = pairs[:train_count]
        val_pairs = pairs[train_count:train_count + val_count]
        test_pairs = pairs[train_count + val_count:] if test_count > 0 else []
        
        # Copy files
        self._copy_pairs_to_split(train_pairs, output_dir / 'train', dataset_info['format'])
        self._copy_pairs_to_split(val_pairs, output_dir / 'val', dataset_info['format'])
        if test_pairs:
            self._copy_pairs_to_split(test_pairs, output_dir / 'test', dataset_info['format'])
        
        return {
            'success': True,
            'train_count': len(train_pairs),
            'val_count': len(val_pairs),
            'test_count': len(test_pairs)
        }
    
    def _merge_and_split_dataset(self, output_dir: Path, dataset_info: Dict[str, Any],
                                train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, Any]:
        """✂️ Merge existing splits and re-split with new proportions."""
        
        # Collect all pairs from existing splits
        all_pairs = []
        special_files = {}
        
        for subdir in dataset_info['subdirs']:
            if subdir.name.lower() in ['train', 'test', 'val', 'valid']:
                subdir_info = {
                    'image_files': [],
                    'annotation_files': [],
                    'subdirs': [],
                    'special_files': {}
                }
                
                # Scan subdirectory - check for nested structure
                subdir_contents = list(subdir.iterdir())
                subdir_subdirs = [item.name.lower() for item in subdir_contents if item.is_dir()]
                
                if 'images' in subdir_subdirs:
                    # Nested structure: split/images/ and split/labels/
                    images_dir = subdir / 'images'
                    labels_dir = None
                    for potential_name in ['labels', 'annotations']:
                        potential_dir = subdir / potential_name
                        if potential_dir.exists():
                            labels_dir = potential_dir
                            break
                    
                    # Get images from images subdirectory
                    for item in images_dir.iterdir():
                        if item.is_file() and item.suffix.lower() in self.supported_image_formats:
                            subdir_info['image_files'].append(item)
                    
                    # Get annotations from labels subdirectory
                    if labels_dir:
                        for item in labels_dir.iterdir():
                            if item.is_file() and item.suffix.lower() in ['.txt', '.json', '.xml', '.csv']:
                                subdir_info['annotation_files'].append(item)
                                # Check for special files
                                if item.name.lower() == '_annotations.csv':
                                    subdir_info['special_files']['annotations_csv'] = item
                                    special_files['annotations_csv'] = item
                                elif item.name.lower() == '_annotations.coco.json':
                                    subdir_info['special_files']['annotations_coco'] = item
                                    special_files['annotations_coco'] = item
                else:
                    # Flat structure within split
                    for item in subdir.iterdir():
                        if item.is_file():
                            ext = item.suffix.lower()
                            if ext in self.supported_image_formats:
                                subdir_info['image_files'].append(item)
                            elif ext in ['.txt', '.json', '.xml', '.csv']:
                                subdir_info['annotation_files'].append(item)
                                # Check for special files
                                if item.name.lower() == '_annotations.csv':
                                    subdir_info['special_files']['annotations_csv'] = item
                                    special_files['annotations_csv'] = item
                                elif item.name.lower() == '_annotations.coco.json':
                                    subdir_info['special_files']['annotations_coco'] = item
                                    special_files['annotations_coco'] = item
                        elif item.is_dir():
                            subdir_info['subdirs'].append(item)
                
                pairs = self._get_image_annotation_pairs(subdir_info)
                all_pairs.extend(pairs)
        
        if not all_pairs:
            return {'success': False, 'error': 'No se encontraron pares imagen-anotación en los splits existentes'}
        
        # Update dataset_info with merged data
        dataset_info['image_files'] = [pair[0] for pair in all_pairs]
        dataset_info['annotation_files'] = [pair[1] for pair in all_pairs if pair[1]]
        dataset_info['special_files'].update(special_files)
        
        return self._split_flat_dataset(output_dir, dataset_info, train_ratio, val_ratio, test_ratio)
    
    def _split_nested_dataset(self, output_dir: Path, dataset_info: Dict[str, Any],
                             train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, Any]:
        """✂️ Split nested dataset structure (e.g., images/ and labels/ dirs)."""
        
        # Find images and labels directories
        images_dir = None
        labels_dir = None
        
        for subdir in dataset_info['subdirs']:
            if subdir.name.lower() == 'images':
                images_dir = subdir
            elif subdir.name.lower() in ['labels', 'annotations']:
                labels_dir = subdir
        
        if not images_dir:
            return {'success': False, 'error': 'No se encontró directorio de imágenes'}
        
        # Get image files from images directory
        image_files = []
        for ext in self.supported_image_formats:
            image_files.extend(images_dir.glob(f'*{ext}'))
            image_files.extend(images_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            return {'success': False, 'error': 'No se encontraron imágenes en el directorio images/'}
        
        # Create pairs
        pairs = []
        special_files = {}
        
        # Check for special annotation files in labels directory
        if labels_dir:
            for item in labels_dir.iterdir():
                if item.is_file():
                    if item.name.lower() == '_annotations.csv':
                        special_files['annotations_csv'] = item
                        dataset_info['format'] = 'tensorflow'
                    elif item.name.lower() == '_annotations.coco.json':
                        special_files['annotations_coco'] = item
                        dataset_info['format'] = 'coco'
        
        # Update dataset_info with special files
        dataset_info['special_files'].update(special_files)
        
        # If we have special annotation files, use them
        if special_files:
            if 'annotations_csv' in special_files:
                for image_file in image_files:
                    pairs.append((image_file, special_files['annotations_csv']))
            elif 'annotations_coco' in special_files:
                for image_file in image_files:
                    pairs.append((image_file, special_files['annotations_coco']))
        else:
            # Standard pairing
            for image_file in image_files:
                annotation_file = None
                if labels_dir:
                    # Try different annotation extensions
                    for ext in ['.txt', '.json', '.xml']:
                        potential_ann = labels_dir / image_file.with_suffix(ext).name
                        if potential_ann.exists():
                            annotation_file = potential_ann
                            break
                
                pairs.append((image_file, annotation_file))
        
        # Update dataset_info for splitting
        dataset_info['image_files'] = [pair[0] for pair in pairs]
        dataset_info['annotation_files'] = [pair[1] for pair in pairs if pair[1]]
        
        return self._split_flat_dataset(output_dir, dataset_info, train_ratio, val_ratio, test_ratio)
    
    def _get_image_annotation_pairs(self, dataset_info: Dict[str, Any]) -> List[Tuple[Path, Optional[Path]]]:
        """🔗 Get image-annotation pairs."""
        pairs = []
        
        # Handle special annotation formats
        if dataset_info['format'] == 'tensorflow' and 'annotations_csv' in dataset_info['special_files']:
            # For TensorFlow format with _annotations.csv
            csv_file = dataset_info['special_files']['annotations_csv']
            pairs = self._get_pairs_from_csv(csv_file, dataset_info['image_files'])
            
        elif dataset_info['format'] == 'coco' and 'annotations_coco' in dataset_info['special_files']:
            # For COCO format with _annotations.coco.json
            json_file = dataset_info['special_files']['annotations_coco']
            pairs = self._get_pairs_from_coco_json(json_file, dataset_info['image_files'])
            
        else:
            # Standard pairing approach
            for image_file in dataset_info['image_files']:
                annotation_file = None
                
                # Try to find corresponding annotation file
                base_name = image_file.stem
                
                # Look for annotation with same base name
                for ann_file in dataset_info['annotation_files']:
                    if ann_file.stem == base_name:
                        annotation_file = ann_file
                        break
                
                pairs.append((image_file, annotation_file))
        
        return pairs
    
    def _get_pairs_from_csv(self, csv_file: Path, image_files: List[Path]) -> List[Tuple[Path, Optional[Path]]]:
        """🔗 Get pairs from TensorFlow _annotations.csv format."""
        pairs = []
        
        # For TensorFlow format, all images share the same annotation file
        for image_file in image_files:
            pairs.append((image_file, csv_file))
        
        return pairs
    
    def _get_pairs_from_coco_json(self, json_file: Path, image_files: List[Path]) -> List[Tuple[Path, Optional[Path]]]:
        """🔗 Get pairs from COCO _annotations.coco.json format."""
        pairs = []
        
        # For COCO format, all images share the same annotation file
        for image_file in image_files:
            pairs.append((image_file, json_file))
        
        return pairs
    
    def _create_split_structure(self, output_dir: Path, include_test: bool = False):
        """🏗️ Create split directory structure."""
        splits = ['train', 'val']
        if include_test:
            splits.append('test')
        
        for split in splits:
            split_dir = output_dir / split
            (split_dir / 'images').mkdir(parents=True, exist_ok=True)
            # Create both labels and annotations directories for flexibility
            (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
            (split_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    
    def _copy_pairs_to_split(self, pairs: List[Tuple[Path, Optional[Path]]], 
                            split_dir: Path, format_type: str):
        """📂 Copy image-annotation pairs to split directory."""
        
        # Ensure directories exist
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        
        # Create appropriate annotation directory based on format
        if format_type in ['yolo', 'generic']:
            (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
            ann_subdir = 'labels'
        else:
            (split_dir / 'annotations').mkdir(parents=True, exist_ok=True)
            ann_subdir = 'annotations'
        
        # Track which annotation files we've already copied (for shared annotations)
        copied_annotations = set()
        
        for image_file, annotation_file in pairs:
            try:
                # Copy image
                dest_image = split_dir / 'images' / image_file.name
                shutil.copy2(image_file, dest_image)
                
                # Copy annotation if exists
                if annotation_file:
                    dest_annotation = split_dir / ann_subdir / annotation_file.name
                    
                    # For shared annotation files (like _annotations.csv), only copy once
                    annotation_key = str(annotation_file)
                    if annotation_key not in copied_annotations:
                        shutil.copy2(annotation_file, dest_annotation)
                        copied_annotations.add(annotation_key)
                    
            except FileNotFoundError as e:
                print(f"⚠️ Error copiando archivo: {e}")
                print(f"   Origen imagen: {image_file}")
                if annotation_file:
                    print(f"   Origen anotación: {annotation_file}")
                print(f"   Destino: {split_dir}")
                continue
            except Exception as e:
                print(f"⚠️ Error inesperado: {e}")
                continue
    
    def _load_classes(self, dataset_info: Dict[str, Any]) -> List[str]:
        """📋 Load class names from dataset."""
        classes = []
        
        # Try config files first
        for config_file in dataset_info['config_files']:
            try:
                if config_file.suffix.lower() in ['.yaml', '.yml'] and YAML_AVAILABLE:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        if isinstance(config, dict) and 'names' in config:
                            if isinstance(config['names'], list):
                                return config['names']
                            elif isinstance(config['names'], dict):
                                return list(config['names'].values())
                elif config_file.name.lower() == 'classes.txt':
                    with open(config_file, 'r', encoding='utf-8') as f:
                        classes = [line.strip() for line in f if line.strip()]
                        return classes
            except:
                continue
        
        # If no classes found, generate generic ones
        if not classes and dataset_info['annotation_files']:
            class_indices = set()
            for ann_file in dataset_info['annotation_files'][:10]:  # Sample first 10
                try:
                    if ann_file.suffix.lower() == '.txt':
                        with open(ann_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts and self._is_int(parts[0]):
                                    class_indices.add(int(parts[0]))
                except:
                    continue
            
            if class_indices:
                max_class = max(class_indices)
                classes = [f"class_{i}" for i in range(max_class + 1)]
        
        return classes
    
    def _generate_config_file(self, output_dir: Path, dataset_info: Dict[str, Any], 
                             split_result: Dict[str, Any]):
        """⚙️ Generate configuration file for the split dataset."""
        
        config = {
            'path': str(output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(dataset_info['classes']),
            'names': dataset_info['classes']
        }
        
        if split_result['test_count'] > 0:
            config['test'] = 'test/images'
        
        # Save as YAML or fallback format
        if YAML_AVAILABLE:
            with open(output_dir / 'data.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Also save split info
            split_info = {
                'split_date': datetime.now().isoformat(),
                'original_dataset': str(self.input_path),
                'total_images': split_result['train_count'] + split_result['val_count'] + split_result['test_count'],
                'splits': {
                    'train': split_result['train_count'],
                    'val': split_result['val_count'],
                    'test': split_result['test_count']
                },
                'format': dataset_info['format']
            }
            
            with open(output_dir / 'split_info.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(split_info, f, default_flow_style=False)
            
            print(f"⚙️ Archivos de configuración generados:")
            print(f"   📄 data.yaml")
            print(f"   📄 split_info.yaml")
        else:
            # Fallback to simple text format
            with open(output_dir / 'data.txt', 'w', encoding='utf-8') as f:
                f.write(f"path: {config['path']}\\n")
                f.write(f"train: {config['train']}\\n")
                f.write(f"val: {config['val']}\\n")
                if 'test' in config:
                    f.write(f"test: {config['test']}\\n")
                f.write(f"nc: {config['nc']}\\n")
                f.write("names:\\n")
                for i, name in enumerate(config['names']):
                    f.write(f"  {i}: {name}\\n")
            
            print(f"⚙️ Archivo de configuración generado:")
            print(f"   📄 data.txt (YAML no disponible)")
    
    def _analyze_yaml_structure(self, yaml_config: Dict[str, Any], dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """🔍 Analyze dataset structure based on YAML configuration."""
        yaml_info = {
            'format': 'yolo',
            'structure': 'unknown',
            'image_files': [],
            'annotation_files': [],
            'classes': []
        }
        
        # Extract classes from YAML
        if 'names' in yaml_config:
            if isinstance(yaml_config['names'], list):
                yaml_info['classes'] = yaml_config['names']
            elif isinstance(yaml_config['names'], dict):
                yaml_info['classes'] = list(yaml_config['names'].values())
        
        # Analyze paths specified in YAML
        yaml_paths = {}
        for key in ['train', 'val', 'test']:
            if key in yaml_config:
                path_str = yaml_config[key]
                yaml_path = None
                
                # Try multiple resolution strategies
                if not os.path.isabs(path_str):
                    # Find the YAML file location
                    yaml_file = None
                    for config_file in dataset_info.get('config_files', []):
                        if config_file.name.lower() in ['data.yaml', 'data.yml']:
                            yaml_file = config_file
                            break
                    
                    if yaml_file:
                        # Strategy 1: Resolve relative to YAML file location
                        candidate1 = (yaml_file.parent / path_str).resolve()
                        
                        # Strategy 2: If YAML path uses ../, try without ../
                        if path_str.startswith('../'):
                            path_without_parent = path_str[3:]  # Remove '../'
                            candidate2 = (yaml_file.parent / path_without_parent).resolve()
                        else:
                            candidate2 = None
                        
                        # Strategy 3: Try just the last part of the path (e.g., train/images)
                        path_parts = Path(path_str).parts
                        if len(path_parts) >= 2:
                            candidate3 = (yaml_file.parent / path_parts[-2] / path_parts[-1]).resolve()
                        else:
                            candidate3 = None
                        
                        # Choose the first existing path
                        for candidate in [candidate1, candidate2, candidate3]:
                            if candidate and candidate.exists():
                                yaml_path = candidate
                                break
                        
                        # If none exist, use the original resolution
                        if not yaml_path:
                            yaml_path = candidate1
                    else:
                        yaml_path = (self.input_path / path_str).resolve()
                else:
                    yaml_path = Path(path_str)
                
                yaml_paths[key] = yaml_path
                print(f"🔍 Resolviendo path {key}: '{path_str}' -> {yaml_path}")
                print(f"   📂 Existe: {yaml_path.exists()}")
        
        if not yaml_paths:
            print("❌ No se encontraron paths válidos en el YAML")
            return None  # No valid paths found
        
        # Determine structure based on YAML paths
        all_images = []
        all_annotations = []
        
        for split_name, split_path in yaml_paths.items():
            if split_path.exists():
                if split_path.is_dir():
                    # If path points to a directory, scan for images
                    for ext in self.supported_image_formats:
                        all_images.extend(split_path.glob(f'*{ext}'))
                        all_images.extend(split_path.glob(f'*{ext.upper()}'))
                    
                    # Look for corresponding labels directory
                    parent_dir = split_path.parent
                    labels_dir = None
                    
                    # Try different label directory patterns
                    for labels_name in ['labels', 'annotations']:
                        potential_labels = parent_dir / labels_name
                        if potential_labels.exists():
                            labels_dir = potential_labels
                            break
                        
                        # Also try as sibling of images directory
                        if split_path.name == 'images':
                            potential_labels = parent_dir / labels_name
                            if potential_labels.exists():
                                labels_dir = potential_labels
                                break
                    
                    if labels_dir:
                        for ext in ['.txt', '.json', '.xml']:
                            all_annotations.extend(labels_dir.glob(f'*{ext}'))
        
        yaml_info['image_files'] = all_images
        yaml_info['annotation_files'] = all_annotations
        
        # Determine structure
        if len(yaml_paths) > 1:
            yaml_info['structure'] = 'split'
        elif all_images:
            yaml_info['structure'] = 'nested'
        
        print(f"📄 Configuración YAML detectada:")
        print(f"   🏷️ Clases: {len(yaml_info['classes'])}")
        print(f"   📁 Splits: {list(yaml_paths.keys())}")
        print(f"   🖼️ Imágenes encontradas: {len(all_images)}")
        
        return yaml_info
    
    def _split_yaml_dataset(self, output_dir: Path, dataset_info: Dict[str, Any],
                           train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, Any]:
        """✂️ Split YAML-configured dataset by merging existing splits."""
        
        yaml_config = dataset_info['yaml_config']
        all_pairs = []
        
        # Collect all images from YAML-specified paths
        for split_name in ['train', 'val', 'test']:
            if split_name in yaml_config:
                path_str = yaml_config[split_name]
                split_path = None
                
                # Use the same intelligent path resolution as in _analyze_yaml_structure
                if not os.path.isabs(path_str):
                    # Find the YAML file location
                    yaml_file = None
                    for config_file in dataset_info.get('config_files', []):
                        if config_file.name.lower() in ['data.yaml', 'data.yml']:
                            yaml_file = config_file
                            break
                    
                    if yaml_file:
                        # Strategy 1: Resolve relative to YAML file location
                        candidate1 = (yaml_file.parent / path_str).resolve()
                        
                        # Strategy 2: If YAML path uses ../, try without ../
                        if path_str.startswith('../'):
                            path_without_parent = path_str[3:]  # Remove '../'
                            candidate2 = (yaml_file.parent / path_without_parent).resolve()
                        else:
                            candidate2 = None
                        
                        # Strategy 3: Try just the last part of the path (e.g., train/images)
                        path_parts = Path(path_str).parts
                        if len(path_parts) >= 2:
                            candidate3 = (yaml_file.parent / path_parts[-2] / path_parts[-1]).resolve()
                        else:
                            candidate3 = None
                        
                        # Choose the first existing path
                        for candidate in [candidate1, candidate2, candidate3]:
                            if candidate and candidate.exists():
                                split_path = candidate
                                break
                        
                        # If none exist, use the original resolution
                        if not split_path:
                            split_path = candidate1
                    else:
                        split_path = (self.input_path / path_str).resolve()
                else:
                    split_path = Path(path_str)
                
                print(f"🔍 Procesando split {split_name}: {split_path} (existe: {split_path.exists()})")
                
                if split_path.exists() and split_path.is_dir():
                    # Get images from this split
                    images = []
                    for ext in self.supported_image_formats:
                        images.extend(split_path.glob(f'*{ext}'))
                        images.extend(split_path.glob(f'*{ext.upper()}'))
                    
                    # Find corresponding labels directory
                    labels_dir = None
                    parent_dir = split_path.parent
                    
                    # Try different patterns for labels directory
                    for labels_name in ['labels', 'annotations']:
                        # Same level as images directory
                        potential_labels = parent_dir / labels_name
                        if potential_labels.exists():
                            labels_dir = potential_labels
                            break
                        
                        # If images is a subdirectory, try sibling labels
                        if split_path.name == 'images':
                            potential_labels = parent_dir / labels_name
                            if potential_labels.exists():
                                labels_dir = potential_labels
                                break
                    
                    # Create pairs for this split
                    for image in images:
                        annotation = None
                        if labels_dir:
                            # Try different annotation extensions
                            for ext in ['.txt', '.json', '.xml']:
                                potential_ann = labels_dir / image.with_suffix(ext).name
                                if potential_ann.exists():
                                    annotation = potential_ann
                                    break
                        
                        all_pairs.append((image, annotation))
        
        if not all_pairs:
            return {'success': False, 'error': 'No se encontraron pares imagen-anotación en los paths del YAML'}
        
        print(f"📄 Mergando {len(all_pairs)} pares desde configuración YAML")
        
        # Now treat as flat dataset and split with new proportions
        dataset_info['image_files'] = [pair[0] for pair in all_pairs]
        dataset_info['annotation_files'] = [pair[1] for pair in all_pairs if pair[1]]
        
        return self._split_flat_dataset(output_dir, dataset_info, train_ratio, val_ratio, test_ratio)
    
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
