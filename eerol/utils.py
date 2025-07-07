"""
🛠️ EEROL Utils Module
=====================

Utility functions for EEROL dataset management tool.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Optional dependencies
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class EerolUtils:
    """🛠️ EEROL utility functions."""
    
    @staticmethod
    def get_dataset_info(dataset_path: Path) -> Dict[str, Any]:
        """📊 Get basic information about a dataset."""
        dataset_path = Path(dataset_path)
        
        info = {
            'name': dataset_path.name,
            'path': str(dataset_path),
            'total_images': 0,
            'format': 'unknown',
            'has_script': False,
            'size_mb': 0
        }
        
        if not dataset_path.exists():
            return info
        
        # Count images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for ext in image_extensions:
            info['total_images'] += len(list(dataset_path.glob(f'**/*{ext}')))
        
        # Detect format
        if (dataset_path / 'data.yaml').exists():
            info['format'] = 'yolo'
        elif list(dataset_path.glob('**/*.json')):
            # Check if it's COCO format
            for json_file in dataset_path.glob('**/*.json'):
                try:
                    import json
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'images' in data and 'annotations' in data:
                            info['format'] = 'coco'
                            break
                except:
                    continue
        elif list(dataset_path.glob('**/*.xml')):
            info['format'] = 'pascal_voc'
        
        # Check for training script
        info['has_script'] = (dataset_path / 'train.py').exists()
        
        # Calculate size
        total_size = 0
        for file_path in dataset_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        info['size_mb'] = round(total_size / (1024 * 1024), 2)
        
        return info
    
    @staticmethod
    def clean_unused_files() -> Dict[str, Any]:
        """🧹 Clean unused files from the current directory."""
        current_dir = Path.cwd()
        
        # Files to clean
        patterns_to_clean = [
            '**/__pycache__',
            '**/*.pyc',
            '**/*.pyo',
            '**/.DS_Store',
            '**/Thumbs.db',
            '**/*.tmp',
            '**/*.temp',
            '**/.pytest_cache',
            '**/node_modules'  # If any JS files
        ]
        
        # Files that contain old/unused patterns
        obsolete_files = [
            'main.py',  # Old main file
            'dataset_preview_tool.py',
            'demo_*.py',
            'test_*.py',
            'diagnostico_*.py',
            'ejemplo_*.py',
            'regenerar_*.py',
            'resumen_*.py',
            'scan_*.py',
            'setup_*.py',
            'fix_*.py',
            'limpiar_*.py',
            'generar_*.py',
            'generate_*.py',
            'install_*.py'
        ]
        
        deleted_count = 0
        space_freed = 0
        errors = []
        
        try:
            # Clean cache and temporary files
            for pattern in patterns_to_clean:
                for path in current_dir.glob(pattern):
                    if path.exists():
                        try:
                            if path.is_dir():
                                space_freed += EerolUtils._get_dir_size(path)
                                shutil.rmtree(path)
                            else:
                                space_freed += path.stat().st_size
                                path.unlink()
                            deleted_count += 1
                        except Exception as e:
                            errors.append(f"Error eliminando {path}: {str(e)}")
            
            # Check for obsolete files (but ask user confirmation in real implementation)
            obsolete_found = []
            for pattern in obsolete_files:
                for path in current_dir.glob(pattern):
                    if path.is_file() and path.name != 'eerol.py':  # Don't delete our main script
                        obsolete_found.append(path)
            
            if obsolete_found:
                print(f"\\n🗑️ ARCHIVOS OBSOLETOS ENCONTRADOS:")
                for path in obsolete_found:
                    print(f"   📄 {path.name}")
                
                # In a real implementation, you'd ask for user confirmation here
                # For now, we'll just report them
                print(f"\\n💡 Para eliminar archivos obsoletos, hazlo manualmente o confirma la eliminación.")
            
            return {
                'success': True,
                'deleted_count': deleted_count,
                'space_freed': EerolUtils._format_size(space_freed),
                'obsolete_found': len(obsolete_found),
                'errors': errors
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error durante la limpieza: {str(e)}',
                'deleted_count': deleted_count,
                'space_freed': EerolUtils._format_size(space_freed)
            }
    
    @staticmethod
    def _get_dir_size(path: Path) -> int:
        """Calculate directory size in bytes."""
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    @staticmethod
    def validate_dataset_structure(dataset_path: Path) -> Dict[str, Any]:
        """✅ Validate dataset structure."""
        dataset_path = Path(dataset_path)
        
        validation = {
            'is_valid': False,
            'format': 'unknown',
            'issues': [],
            'recommendations': []
        }
        
        if not dataset_path.exists():
            validation['issues'].append(f"Dataset path does not exist: {dataset_path}")
            return validation
        
        # Check for images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_files.extend(list(dataset_path.glob(f'**/*{ext}')))
        
        if not image_files:
            validation['issues'].append("No image files found")
            return validation
        
        # Detect format and validate accordingly
        if (dataset_path / 'data.yaml').exists():
            validation['format'] = 'yolo'
            validation.update(EerolUtils._validate_yolo_structure(dataset_path))
        elif list(dataset_path.glob('**/*.json')):
            validation['format'] = 'coco'
            validation.update(EerolUtils._validate_coco_structure(dataset_path))
        elif list(dataset_path.glob('**/*.xml')):
            validation['format'] = 'pascal_voc'
            validation.update(EerolUtils._validate_pascal_structure(dataset_path))
        else:
            validation['issues'].append("Unknown annotation format")
            validation['recommendations'].append("Add proper annotation files or configuration")
        
        validation['is_valid'] = len(validation['issues']) == 0
        
        return validation
    
    @staticmethod
    def _validate_yolo_structure(dataset_path: Path) -> Dict[str, List[str]]:
        """Validate YOLO dataset structure."""
        issues = []
        recommendations = []
        
        # Check for data.yaml
        data_yaml = dataset_path / 'data.yaml'
        if not data_yaml.exists():
            issues.append("Missing data.yaml configuration file")
        else:
            try:
                if YAML_AVAILABLE:
                    import yaml
                    with open(data_yaml, 'r') as f:
                        config = yaml.safe_load(f)
                        if 'names' not in config:
                            issues.append("data.yaml missing 'names' field")
                        if 'nc' not in config:
                            issues.append("data.yaml missing 'nc' field")
                else:
                    issues.append("YAML library not available to validate data.yaml")
            except:
                issues.append("Invalid data.yaml format")
        
        # Check for proper directory structure
        expected_dirs = ['train', 'val']
        for dir_name in expected_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                issues.append(f"Missing {dir_name} directory")
            else:
                if not (dir_path / 'images').exists():
                    issues.append(f"Missing {dir_name}/images directory")
                if not (dir_path / 'labels').exists():
                    issues.append(f"Missing {dir_name}/labels directory")
        
        return {'issues': issues, 'recommendations': recommendations}
    
    @staticmethod
    def _validate_coco_structure(dataset_path: Path) -> Dict[str, List[str]]:
        """Validate COCO dataset structure."""
        issues = []
        recommendations = []
        
        # Check for annotation file
        json_files = list(dataset_path.glob('**/*.json'))
        valid_coco = False
        
        for json_file in json_files:
            try:
                import json
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        required_keys = ['images', 'annotations', 'categories']
                        if all(key in data for key in required_keys):
                            valid_coco = True
                            break
            except:
                continue
        
        if not valid_coco:
            issues.append("No valid COCO annotation file found")
        
        return {'issues': issues, 'recommendations': recommendations}
    
    @staticmethod
    def _validate_pascal_structure(dataset_path: Path) -> Dict[str, List[str]]:
        """Validate Pascal VOC dataset structure."""
        issues = []
        recommendations = []
        
        xml_files = list(dataset_path.glob('**/*.xml'))
        if not xml_files:
            issues.append("No XML annotation files found")
        
        return {'issues': issues, 'recommendations': recommendations}
    
    @staticmethod
    def create_backup(source_path: Path, backup_dir: Path = None) -> Dict[str, Any]:
        """💾 Create backup of a dataset."""
        source_path = Path(source_path)
        
        if backup_dir is None:
            backup_dir = source_path.parent / 'backups'
        
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{source_path.name}_backup_{timestamp}"
        backup_path = backup_dir / backup_name
        
        try:
            if source_path.is_file():
                shutil.copy2(source_path, backup_path)
            else:
                shutil.copytree(source_path, backup_path)
            
            return {
                'success': True,
                'backup_path': str(backup_path),
                'size': EerolUtils._format_size(EerolUtils._get_dir_size(backup_path))
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error creating backup: {str(e)}'
            }
    
    @staticmethod
    def setup_environment() -> Dict[str, Any]:
        """🔧 Setup EEROL environment."""
        print("🔧 Configurando entorno EEROL...")
        
        # Create necessary directories
        directories = ['Train', 'Results', 'Backups']
        created_dirs = []
        
        for dir_name in directories:
            dir_path = Path.cwd() / dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
                created_dirs.append(dir_name)
        
        # Create .eerol config directory
        config_dir = Path.home() / '.eerol'
        config_dir.mkdir(exist_ok=True)
        
        # Create basic config file
        config_file = config_dir / 'config.yaml'
        if not config_file.exists():
            config_content = f'''# EEROL Configuration
version: "3.0"
created: "{datetime.now().isoformat()}"

# Default settings
settings:
  default_train_dir: "Train"
  default_results_dir: "Results"
  backup_enabled: true
  auto_cleanup: false

# Supported formats
formats:
  - yolo
  - coco
  - pascal_voc

# Default training parameters
training:
  yolo:
    epochs: 100
    batch_size: 16
    image_size: 640
  
  pytorch:
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
'''
            
            with open(config_file, 'w') as f:
                f.write(config_content)
        
        return {
            'success': True,
            'created_directories': created_dirs,
            'config_path': str(config_file)
        }
