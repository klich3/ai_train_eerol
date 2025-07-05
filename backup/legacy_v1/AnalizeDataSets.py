"""
█▀ █▄█ █▀▀ █░█ █▀▀ █░█
▄█ ░█░ █▄▄ █▀█ ██▄ ▀▄▀

Author: <Anton Sychev> (anton at sychev dot xyz)
script.py (c) 2025
Created:  2025-07-05 06:25:08 
Desc: Dataset Analyzer for Dental X-Ray AI Training
Docs: Analyzes YOLO, COCO, UNET and other datasets for dental radiography
"""

import os
import json
import yaml
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any
import pandas as pd
from datetime import datetime

class DentalDatasetAnalyzer:
    """
    Analizador de datasets dentales para diferentes arquitecturas de redes neuronales.
    Soporta YOLO, COCO, UNET y otros formatos comunes.
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.results = {}
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Mapeo de tipos de dataset basado en estructura de carpetas
        self.dataset_types = {
            '_YOLO': 'YOLO Object Detection',
            '_COCO': 'COCO Format (Detection/Segmentation)', 
            '_UNET': 'U-Net Semantic Segmentation',
        }
        
        # Categorías dentales comunes encontradas en datasets
        self.common_dental_categories = {
            'caries', 'cavity', 'tooth', 'teeth', 'molar', 'premolar', 'canine', 'incisor',
            'root', 'crown', 'decay', 'filling', 'implant', 'bridge', 'orthodontic',
            'panoramic', 'periapical', 'bitewing', 'dental', 'radiograph', 'xray',
            'mandible', 'maxilla', 'jaw', 'bone', 'nerve', 'periodontitis', 'gingivitis'
        }
    
    def analyze_all_datasets(self) -> Dict[str, Any]:
        """Analiza todos los datasets en el directorio base."""
        print("🦷 Iniciando análisis de datasets dentales...")
        
        # Analizar cada tipo de dataset
        for dataset_type_folder in self.base_path.iterdir():
            if dataset_type_folder.is_dir() and dataset_type_folder.name.startswith('_'):
                print(f"\n📁 Analizando {dataset_type_folder.name}...")
                self.results[dataset_type_folder.name] = self._analyze_dataset_type(dataset_type_folder)
        
        # Generar resumen general
        self.results['summary'] = self._generate_summary()
        
        return self.results
    
    def _analyze_dataset_type(self, dataset_folder: Path) -> Dict[str, Any]:
        """Analiza un tipo específico de dataset."""
        analysis = {
            'type': self.dataset_types.get(dataset_folder.name, 'Unknown Format'),
            'datasets': {},
            'total_datasets': 0,
            'total_images': 0,
            'categories_found': set(),
            'folder_path': str(dataset_folder)
        }
        
        # Analizar cada subdirectorio
        for subdataset in dataset_folder.iterdir():
            if subdataset.is_dir():
                print(f"  📊 Procesando: {subdataset.name}")
                dataset_analysis = self._analyze_individual_dataset(subdataset, dataset_folder.name)
                analysis['datasets'][subdataset.name] = dataset_analysis
                analysis['total_datasets'] += 1
                analysis['total_images'] += dataset_analysis['image_count']
                analysis['categories_found'].update(dataset_analysis['categories'])
        
        # Convertir sets a listas para JSON serialization
        analysis['categories_found'] = list(analysis['categories_found'])
        
        return analysis
    
    def _analyze_individual_dataset(self, dataset_path: Path, dataset_type: str) -> Dict[str, Any]:
        """Analiza un dataset individual."""
        analysis = {
            'name': dataset_path.name,
            'path': str(dataset_path),
            'format_type': dataset_type,
            'image_count': 0,
            'annotation_count': 0,
            'categories': [],
            'class_distribution': {},
            'image_formats': set(),
            'structure': {},
            'metadata': {},
            'recommended_use': '',
            'dataset_quality': 'Unknown'
        }
        
        # Analizar estructura de archivos
        analysis['structure'] = self._analyze_folder_structure(dataset_path)
        
        # Contar imágenes
        analysis['image_count'] = self._count_images(dataset_path)
        
        # Análisis específico por tipo
        if dataset_type == '_YOLO':
            analysis.update(self._analyze_yolo_dataset(dataset_path))
        elif dataset_type == '_COCO':
            analysis.update(self._analyze_coco_dataset(dataset_path))
        elif dataset_type == '_UNET':
            analysis.update(self._analyze_unet_dataset(dataset_path))
        else:
            analysis.update(self._analyze_generic_dataset(dataset_path))
        
        # Determinar uso recomendado
        analysis['recommended_use'] = self._determine_recommended_use(analysis)
        
        # Evaluar calidad del dataset
        analysis['dataset_quality'] = self._evaluate_dataset_quality(analysis)
        
        # Convertir sets a listas
        analysis['image_formats'] = list(analysis['image_formats'])
        
        return analysis
    
    def _analyze_yolo_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Analiza un dataset en formato YOLO."""
        yolo_analysis = {
            'annotation_format': 'YOLO (.txt)',
            'has_classes_file': False,
            'has_data_yaml': False,
            'splits': [],
            'classes': []
        }
        
        # Buscar archivo de clases
        classes_file = None
        for file_pattern in ['classes.txt', 'obj.names', 'data.names']:
            potential_file = dataset_path / file_pattern
            if potential_file.exists():
                yolo_analysis['has_classes_file'] = True
                classes_file = potential_file
                break
        
        # Buscar data.yaml
        data_yaml = dataset_path / 'data.yaml'
        if data_yaml.exists():
            yolo_analysis['has_data_yaml'] = True
            try:
                with open(data_yaml, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if 'names' in yaml_data:
                        yolo_analysis['classes'] = yaml_data['names']
                        if isinstance(yaml_data['names'], dict):
                            yolo_analysis['classes'] = list(yaml_data['names'].values())
            except:
                pass
        
        # Si no hay yaml, leer archivo de clases
        if not yolo_analysis['classes'] and classes_file:
            try:
                with open(classes_file, 'r') as f:
                    yolo_analysis['classes'] = [line.strip() for line in f if line.strip()]
            except:
                pass
        
        # Buscar splits (train/val/test)
        for split in ['train', 'val', 'test', 'valid']:
            if (dataset_path / split).exists():
                yolo_analysis['splits'].append(split)
        
        # Contar anotaciones
        txt_files = list(dataset_path.rglob('*.txt'))
        yolo_analysis['annotation_count'] = len([f for f in txt_files if f.name != 'classes.txt'])
        
        # Analizar distribución de clases
        if yolo_analysis['classes']:
            class_counts = defaultdict(int)
            for txt_file in txt_files:
                if txt_file.name in ['classes.txt', 'data.yaml']:
                    continue
                try:
                    with open(txt_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and parts[0].isdigit():
                                class_id = int(parts[0])
                                if class_id < len(yolo_analysis['classes']):
                                    class_counts[yolo_analysis['classes'][class_id]] += 1
                except:
                    continue
            yolo_analysis['class_distribution'] = dict(class_counts)
        
        yolo_analysis['categories'] = yolo_analysis['classes']
        return yolo_analysis
    
    def _analyze_coco_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Analiza un dataset en formato COCO."""
        coco_analysis = {
            'annotation_format': 'COCO JSON',
            'annotation_files': [],
            'classes': [],
            'has_images_folder': False,
            'task_type': 'detection'  # detection, segmentation, keypoints
        }
        
        # Buscar archivos JSON de anotaciones
        json_files = list(dataset_path.rglob('*.json'))
        coco_analysis['annotation_files'] = [str(f.relative_to(dataset_path)) for f in json_files]
        coco_analysis['annotation_count'] = len(json_files)
        
        # Analizar el primer archivo JSON válido
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    coco_data = json.load(f)
                
                if 'categories' in coco_data:
                    coco_analysis['classes'] = [cat['name'] for cat in coco_data['categories']]
                
                if 'annotations' in coco_data:
                    annotations = coco_data['annotations']
                    if annotations and 'segmentation' in annotations[0]:
                        coco_analysis['task_type'] = 'segmentation'
                    elif annotations and 'keypoints' in annotations[0]:
                        coco_analysis['task_type'] = 'keypoints'
                
                # Distribución de clases
                if 'annotations' in coco_data and 'categories' in coco_data:
                    class_counts = defaultdict(int)
                    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
                    for ann in coco_data['annotations']:
                        if 'category_id' in ann:
                            class_name = cat_id_to_name.get(ann['category_id'], 'unknown')
                            class_counts[class_name] += 1
                    coco_analysis['class_distribution'] = dict(class_counts)
                
                break  # Usar el primer archivo válido
            except:
                continue
        
        # Verificar carpeta de imágenes
        for img_folder in ['images', 'imgs', 'img']:
            if (dataset_path / img_folder).exists():
                coco_analysis['has_images_folder'] = True
                break
        
        coco_analysis['categories'] = coco_analysis['classes']
        return coco_analysis
    
    def _analyze_unet_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Analiza un dataset para U-Net."""
        unet_analysis = {
            'annotation_format': 'Masks/Segmentation Maps',
            'has_masks': False,
            'mask_formats': set(),
            'classes': [],
            'image_mask_pairs': 0
        }
        
        # Buscar carpetas de máscaras
        mask_folders = []
        for folder_name in ['masks', 'mask', 'labels', 'annotations', 'ground_truth', 'gt']:
            potential_folder = dataset_path / folder_name
            if potential_folder.exists():
                mask_folders.append(potential_folder)
                unet_analysis['has_masks'] = True
        
        # Analizar máscaras
        if mask_folders:
            mask_files = []
            for mask_folder in mask_folders:
                mask_files.extend(mask_folder.rglob('*'))
            
            mask_files = [f for f in mask_files if f.suffix.lower() in self.supported_image_formats]
            unet_analysis['annotation_count'] = len(mask_files)
            
            # Formatos de máscaras
            unet_analysis['mask_formats'] = {f.suffix.lower() for f in mask_files}
            
            # Intentar determinar clases analizando valores únicos en las máscaras
            if mask_files:
                try:
                    sample_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
                    if sample_mask is not None:
                        unique_values = np.unique(sample_mask)
                        unet_analysis['classes'] = [f'class_{i}' for i in range(len(unique_values))]
                        unet_analysis['class_distribution'] = {f'class_{i}': int(np.sum(sample_mask == val)) 
                                                             for i, val in enumerate(unique_values)}
                except:
                    pass
        
        # Contar pares imagen-máscara
        image_files = self._get_image_files(dataset_path)
        if mask_folders and image_files:
            # Estimación simple de pares
            unet_analysis['image_mask_pairs'] = min(len(image_files), unet_analysis['annotation_count'])
        
        unet_analysis['mask_formats'] = list(unet_analysis['mask_formats'])
        unet_analysis['categories'] = unet_analysis['classes']
        return unet_analysis
    
    def _analyze_generic_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Analiza un dataset genérico."""
        generic_analysis = {
            'annotation_format': 'Mixed/Unknown',
            'potential_formats': [],
            'classes': [],
            'structure_type': 'unknown'
        }
        
        # Detectar posibles formatos basándose en archivos
        if list(dataset_path.rglob('*.xml')):
            generic_analysis['potential_formats'].append('Pascal VOC (XML)')
        if list(dataset_path.rglob('*.txt')):
            generic_analysis['potential_formats'].append('YOLO (TXT)')
        if list(dataset_path.rglob('*.json')):
            generic_analysis['potential_formats'].append('COCO (JSON)')
        
        # Analizar estructura
        subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        if any(name in [d.name.lower() for d in subdirs] for name in ['train', 'val', 'test']):
            generic_analysis['structure_type'] = 'split_folders'
        elif any(name in [d.name.lower() for d in subdirs] for name in ['images', 'masks', 'labels']):
            generic_analysis['structure_type'] = 'separated_data'
        
        # Intentar extraer categorías de nombres de archivos/carpetas
        categories = set()
        for item in dataset_path.rglob('*'):
            item_name = item.name.lower()
            for dental_term in self.common_dental_categories:
                if dental_term in item_name:
                    categories.add(dental_term)
        
        generic_analysis['categories'] = list(categories)
        generic_analysis['annotation_count'] = len(list(dataset_path.rglob('*.xml'))) + len(list(dataset_path.rglob('*.json')))
        
        return generic_analysis
    
    def _analyze_folder_structure(self, path: Path) -> Dict[str, Any]:
        """Analiza la estructura de carpetas."""
        structure = {
            'total_files': 0,
            'total_folders': 0,
            'depth': 0,
            'main_folders': []
        }
        
        try:
            for root, dirs, files in os.walk(path):
                structure['total_files'] += len(files)
                structure['total_folders'] += len(dirs)
                
                # Calcular profundidad
                depth = len(Path(root).relative_to(path).parts)
                structure['depth'] = max(structure['depth'], depth)
                
                # Carpetas principales (primer nivel)
                if Path(root) == path:
                    structure['main_folders'] = dirs
        except:
            pass
        
        return structure
    
    def _count_images(self, path: Path) -> int:
        """Cuenta el número total de imágenes en un directorio."""
        return len(self._get_image_files(path))
    
    def _get_image_files(self, path: Path) -> List[Path]:
        """Obtiene lista de archivos de imagen."""
        image_files = []
        for fmt in self.supported_image_formats:
            image_files.extend(path.rglob(f'*{fmt}'))
        return image_files
    
    def _determine_recommended_use(self, analysis: Dict[str, Any]) -> str:
        """Determina el uso recomendado basándose en el análisis."""
        format_type = analysis.get('format_type', '')
        categories = analysis.get('categories', [])
        image_count = analysis.get('image_count', 0)
        
        uses = []
        
        if format_type == '_YOLO':
            uses.append("Detección de objetos con YOLO")
        elif format_type == '_COCO':
            task_type = analysis.get('task_type', 'detection')
            if task_type == 'segmentation':
                uses.append("Segmentación de instancias con Mask R-CNN")
            else:
                uses.append("Detección de objetos con modelos COCO")
        elif format_type == '_UNET':
            uses.append("Segmentación semántica con U-Net")
        
        # Basarse en categorías encontradas
        dental_terms = [cat for cat in categories if any(term in cat.lower() for term in self.common_dental_categories)]
        if dental_terms:
            if any(term in dental_terms for term in ['caries', 'cavity', 'decay']):
                uses.append("Detección de caries dentales")
            if any(term in dental_terms for term in ['tooth', 'teeth', 'molar']):
                uses.append("Segmentación/clasificación de dientes")
            if any(term in dental_terms for term in ['panoramic', 'periapical']):
                uses.append("Análisis de radiografías dentales")
        
        # Basarse en tamaño del dataset
        if image_count > 1000:
            uses.append("Entrenamiento de modelos robustos")
        elif image_count > 100:
            uses.append("Entrenamiento con transfer learning")
        else:
            uses.append("Validación o testing")
        
        return " | ".join(uses) if uses else "Uso general en visión por computadora dental"
    
    def _evaluate_dataset_quality(self, analysis: Dict[str, Any]) -> str:
        """Evalúa la calidad del dataset."""
        score = 0
        factors = []
        
        # Factor 1: Cantidad de imágenes
        image_count = analysis.get('image_count', 0)
        if image_count > 1000:
            score += 3
            factors.append("Gran cantidad de imágenes")
        elif image_count > 500:
            score += 2
            factors.append("Cantidad moderada de imágenes")
        elif image_count > 100:
            score += 1
            factors.append("Cantidad básica de imágenes")
        
        # Factor 2: Presencia de anotaciones
        annotation_count = analysis.get('annotation_count', 0)
        if annotation_count > 0:
            score += 2
            factors.append("Tiene anotaciones")
        
        # Factor 3: Categorías bien definidas
        categories = analysis.get('categories', [])
        if len(categories) > 0:
            score += 1
            factors.append("Categorías definidas")
        
        # Factor 4: Estructura organizada
        structure = analysis.get('structure', {})
        if structure.get('main_folders'):
            score += 1
            factors.append("Estructura organizada")
        
        # Factor 5: Distribución de clases
        class_dist = analysis.get('class_distribution', {})
        if class_dist:
            score += 1
            factors.append("Distribución de clases disponible")
        
        # Evaluar calidad final
        if score >= 7:
            quality = "Excelente"
        elif score >= 5:
            quality = "Buena"
        elif score >= 3:
            quality = "Regular"
        else:
            quality = "Básica"
        
        return f"{quality} (Score: {score}/8) - {', '.join(factors)}"
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Genera un resumen general de todos los datasets."""
        summary = {
            'total_dataset_types': len([k for k in self.results.keys() if k.startswith('_')]),
            'total_individual_datasets': 0,
            'total_images': 0,
            'all_categories': set(),
            'format_distribution': defaultdict(int),
            'recommended_architectures': [],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        for dataset_type, data in self.results.items():
            if dataset_type.startswith('_'):
                summary['total_individual_datasets'] += data['total_datasets']
                summary['total_images'] += data['total_images']
                summary['all_categories'].update(data['categories_found'])
                summary['format_distribution'][data['type']] += data['total_datasets']
        
        # Convertir sets a listas
        summary['all_categories'] = list(summary['all_categories'])
        summary['format_distribution'] = dict(summary['format_distribution'])
        
        # Recomendar arquitecturas basándose en los formatos encontrados
        if summary['format_distribution'].get('YOLO Object Detection', 0) > 0:
            summary['recommended_architectures'].append('YOLOv8/YOLOv9 para detección')
        if summary['format_distribution'].get('COCO Format (Detection/Segmentation)', 0) > 0:
            summary['recommended_architectures'].append('Mask R-CNN para segmentación')
        if summary['format_distribution'].get('U-Net Semantic Segmentation', 0) > 0:
            summary['recommended_architectures'].append('U-Net para segmentación semántica')
        
        return summary
    
    def save_results(self, output_file: str = 'dental_dataset_analysis.json'):
        """Guarda los resultados del análisis."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        print(f"📊 Resultados guardados en: {output_file}")
    
    def generate_report(self, output_file: str = 'dental_dataset_report.md'):
        """Genera un reporte en Markdown."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 🦷 Análisis de Datasets Dentales para IA\n\n")
            f.write(f"**Fecha de análisis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Resumen general
            if 'summary' in self.results:
                summary = self.results['summary']
                f.write("## 📈 Resumen General\n\n")
                f.write(f"- **Total de tipos de datasets:** {summary['total_dataset_types']}\n")
                f.write(f"- **Total de datasets individuales:** {summary['total_individual_datasets']}\n")
                f.write(f"- **Total de imágenes:** {summary['total_images']:,}\n")
                f.write(f"- **Categorías únicas encontradas:** {len(summary['all_categories'])}\n\n")
                
                f.write("### 🏗️ Distribución por Formato\n\n")
                for format_type, count in summary['format_distribution'].items():
                    f.write(f"- **{format_type}:** {count} datasets\n")
                f.write("\n")
                
                f.write("### 🤖 Arquitecturas Recomendadas\n\n")
                for arch in summary['recommended_architectures']:
                    f.write(f"- {arch}\n")
                f.write("\n")
            
            # Detalles por tipo de dataset
            for dataset_type, data in self.results.items():
                if dataset_type.startswith('_'):
                    f.write(f"## 📁 {dataset_type} - {data['type']}\n\n")
                    f.write(f"- **Datasets:** {data['total_datasets']}\n")
                    f.write(f"- **Imágenes totales:** {data['total_images']:,}\n")
                    f.write(f"- **Categorías:** {', '.join(data['categories_found'][:10])}{'...' if len(data['categories_found']) > 10 else ''}\n\n")
                    
                    # Top 5 datasets por número de imágenes
                    datasets_by_images = sorted(data['datasets'].items(), 
                                              key=lambda x: x[1]['image_count'], reverse=True)[:5]
                    
                    f.write("### 🔝 Top Datasets por Imágenes\n\n")
                    for name, dataset_data in datasets_by_images:
                        f.write(f"1. **{name}**\n")
                        f.write(f"   - Imágenes: {dataset_data['image_count']:,}\n")
                        f.write(f"   - Calidad: {dataset_data['dataset_quality']}\n")
                        f.write(f"   - Uso recomendado: {dataset_data['recommended_use']}\n\n")
        
        print(f"📄 Reporte generado en: {output_file}")


def main():
    """Función principal para ejecutar el análisis."""
    # Configurar la ruta base
    base_path = "_dataSets"
    
    print("🚀 Iniciando análisis completo de datasets dentales...")
    print(f"📂 Directorio base: {base_path}")
    
    # Crear analizador
    analyzer = DentalDatasetAnalyzer(base_path)
    
    # Ejecutar análisis
    results = analyzer.analyze_all_datasets()
    
    # Guardar resultados
    analyzer.save_results('dental_dataset_analysis.json')
    analyzer.generate_report('dental_dataset_report.md')
    
    # Mostrar resumen en consola
    if 'summary' in results:
        summary = results['summary']
        print("\n" + "="*60)
        print("🦷 RESUMEN DEL ANÁLISIS")
        print("="*60)
        print(f"📊 Total de datasets: {summary['total_individual_datasets']}")
        print(f"🖼️  Total de imágenes: {summary['total_images']:,}")
        print(f"🏷️  Categorías encontradas: {len(summary['all_categories'])}")
        print(f"📁 Formatos disponibles: {', '.join(summary['format_distribution'].keys())}")
        print("\n🤖 Arquitecturas recomendadas:")
        for arch in summary['recommended_architectures']:
            print(f"   • {arch}")
        print("="*60)
    
    print("\n✅ Análisis completado exitosamente!")
    print("📊 Revisa 'dental_dataset_analysis.json' para datos detallados")
    print("📄 Revisa 'dental_dataset_report.md' para el reporte completo")


if __name__ == "__main__":
    main()

