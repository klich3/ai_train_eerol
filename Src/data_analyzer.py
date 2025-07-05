"""
📊 Data Analysis Module for Dental Datasets
Módulo para análisis de datos de datasets dentales
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class DataAnalyzer:
    """Analizador de datasets dentales."""
    
    def __init__(self, base_path: Path, unified_classes: Dict):
        self.base_path = Path(base_path)
        self.unified_classes = unified_classes
        
    def scan_datasets(self) -> Dict[str, Any]:
        """🔍 Escanea todos los datasets disponibles."""
        print("\n🔍 ESCANEANDO DATASETS DISPONIBLES...")
        
        analysis = {
            'yolo_datasets': [],
            'coco_datasets': [],
            'pure_image_datasets': [],
            'unet_datasets': [],
            'total_datasets': 0,
            'total_images': 0,
            'dataset_details': {},
            'class_distribution': Counter(),
            'format_distribution': Counter(),
            'quality_metrics': {}
        }
        
        # Escanear directorios principales
        for main_dir in ['_YOLO', '_COCO', '_pure images and masks', '_UNET']:
            dir_path = self.base_path / main_dir
            if dir_path.exists():
                print(f"  📁 Escaneando: {main_dir}")
                self._scan_directory(dir_path, analysis, main_dir)
        
        analysis['total_datasets'] = len(analysis['dataset_details'])
        
        print(f"\n📊 RESUMEN DEL ESCANEO:")
        print(f"   📊 Total datasets: {analysis['total_datasets']}")
        print(f"   🖼️ Total imágenes estimadas: {analysis['total_images']}")
        print(f"   📋 Formatos detectados: {dict(analysis['format_distribution'])}")
        
        return analysis
    
    def _scan_directory(self, dir_path: Path, analysis: Dict, format_type: str):
        """Escanea un directorio específico."""
        for dataset_dir in dir_path.iterdir():
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                dataset_info = self._analyze_dataset_structure(dataset_dir, format_type)
                if dataset_info['valid']:
                    analysis['dataset_details'][str(dataset_dir)] = dataset_info
                    analysis['total_images'] += dataset_info['image_count']
                    
                    # Actualizar distribuciones
                    analysis['format_distribution'][dataset_info['format']] += 1
                    
                    # Agregar a la lista correspondiente
                    if format_type == '_YOLO':
                        analysis['yolo_datasets'].append(str(dataset_dir))
                    elif format_type == '_COCO':
                        analysis['coco_datasets'].append(str(dataset_dir))
                    elif format_type == '_pure images and masks':
                        analysis['pure_image_datasets'].append(str(dataset_dir))
                    elif format_type == '_UNET':
                        analysis['unet_datasets'].append(str(dataset_dir))
    
    def _analyze_dataset_structure(self, dataset_path: Path, format_type: str) -> Dict[str, Any]:
        """Analiza la estructura de un dataset específico."""
        info = {
            'path': str(dataset_path),
            'name': dataset_path.name,
            'format': 'unknown',
            'valid': False,
            'image_count': 0,
            'annotation_count': 0,
            'classes': [],
            'structure': {},
            'quality_score': 0.0
        }
        
        try:
            # Contar archivos por tipo
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            annotation_extensions = {'.txt', '.json', '.xml'}
            
            image_files = []
            annotation_files = []
            
            for file_path in dataset_path.rglob('*'):
                if file_path.is_file():
                    if file_path.suffix.lower() in image_extensions:
                        image_files.append(file_path)
                    elif file_path.suffix.lower() in annotation_extensions:
                        annotation_files.append(file_path)
            
            info['image_count'] = len(image_files)
            info['annotation_count'] = len(annotation_files)
            
            # Determinar formato basado en estructura
            info['format'] = self._detect_format(dataset_path, format_type, image_files, annotation_files)
            
            # Marcar como válido si tiene imágenes
            if info['image_count'] > 0:
                info['valid'] = True
                
                # Calcular puntuación de calidad
                info['quality_score'] = self._calculate_quality_score(info, image_files[:5])  # Muestra pequeña
                
                # Detectar clases si es posible
                info['classes'] = self._detect_classes(dataset_path, info['format'], annotation_files[:10])
        
        except Exception as e:
            print(f"⚠️ Error analizando {dataset_path}: {e}")
        
        return info
    
    def _detect_format(self, dataset_path: Path, format_type: str, image_files: List, annotation_files: List) -> str:
        """Detecta el formato del dataset."""
        # Basado en el directorio padre
        if format_type == '_YOLO':
            return 'YOLO'
        elif format_type == '_COCO':
            return 'COCO'
        elif format_type == '_UNET':
            return 'U-Net'
        elif format_type == '_pure images and masks':
            # Determinar si es clasificación o segmentación
            if any('mask' in str(f).lower() for f in image_files):
                return 'Segmentation_Images'
            else:
                return 'Classification'
        
        # Detección automática basada en archivos
        if any(f.suffix == '.json' for f in annotation_files):
            return 'COCO'
        elif any(f.suffix == '.txt' for f in annotation_files):
            return 'YOLO'
        elif len(annotation_files) == 0:
            return 'Pure_Images'
        
        return 'Unknown'
    
    def _calculate_quality_score(self, info: Dict, sample_images: List) -> float:
        """Calcula una puntuación de calidad del dataset."""
        score = 0.0
        
        # Factor 1: Relación imagen/anotación
        if info['image_count'] > 0:
            annotation_ratio = info['annotation_count'] / info['image_count']
            score += min(annotation_ratio, 1.0) * 30  # Máximo 30 puntos
        
        # Factor 2: Cantidad de imágenes
        if info['image_count'] >= 1000:
            score += 25
        elif info['image_count'] >= 500:
            score += 20
        elif info['image_count'] >= 100:
            score += 15
        elif info['image_count'] >= 50:
            score += 10
        
        # Factor 3: Calidad de imágenes (muestra)
        try:
            if sample_images:
                resolutions = []
                for img_path in sample_images[:3]:  # Solo 3 muestras para velocidad
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            h, w = img.shape[:2]
                            resolutions.append(w * h)
                    except:
                        continue
                
                if resolutions:
                    avg_resolution = np.mean(resolutions)
                    if avg_resolution >= 1024*1024:  # >= 1MP
                        score += 25
                    elif avg_resolution >= 512*512:
                        score += 20
                    elif avg_resolution >= 256*256:
                        score += 15
                    else:
                        score += 10
        except:
            score += 10  # Puntuación por defecto
        
        # Factor 4: Nombre del dataset (indicativo de calidad)
        name_lower = info['name'].lower()
        quality_keywords = ['high', 'quality', 'annotated', 'curated', 'professional']
        if any(keyword in name_lower for keyword in quality_keywords):
            score += 10
        
        # Factor 5: Estructura organizada
        if 'train' in str(info['path']).lower() or 'val' in str(info['path']).lower():
            score += 10
        
        return min(score, 100.0)  # Máximo 100
    
    def _detect_classes(self, dataset_path: Path, format_type: str, sample_annotations: List) -> List[str]:
        """Detecta las clases presentes en el dataset."""
        classes = set()
        
        try:
            if format_type == 'YOLO':
                # Buscar archivo classes.txt o data.yaml
                classes_file = dataset_path / 'classes.txt'
                data_yaml = dataset_path / 'data.yaml'
                
                if classes_file.exists():
                    with open(classes_file, 'r') as f:
                        classes.update(line.strip() for line in f if line.strip())
                elif data_yaml.exists():
                    import yaml
                    with open(data_yaml, 'r') as f:
                        data = yaml.safe_load(f)
                        if 'names' in data:
                            classes.update(data['names'])
                else:
                    # Analizar archivos .txt de muestra
                    for ann_file in sample_annotations:
                        if ann_file.suffix == '.txt':
                            with open(ann_file, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts:
                                        classes.add(f"class_{parts[0]}")
            
            elif format_type == 'COCO':
                # Analizar archivos JSON
                for ann_file in sample_annotations:
                    if ann_file.suffix == '.json':
                        with open(ann_file, 'r') as f:
                            data = json.load(f)
                            if 'categories' in data:
                                for cat in data['categories']:
                                    classes.add(cat.get('name', f"id_{cat.get('id')}"))
                        break  # Solo el primer archivo JSON válido
            
            elif format_type == 'Classification':
                # Las clases son los nombres de las carpetas
                for item in dataset_path.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        classes.add(item.name)
        
        except Exception as e:
            print(f"⚠️ Error detectando clases en {dataset_path}: {e}")
        
        return sorted(list(classes))
    
    def generate_analysis_report(self, analysis: Dict, output_path: Path) -> Path:
        """📋 Genera un reporte completo del análisis."""
        report_path = output_path / "dental_dataset_analysis.json"
        
        # Agregar metadatos del análisis
        analysis['metadata'] = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'analyzer_version': '1.0',
            'base_path': str(self.base_path)
        }
        
        # Guardar análisis completo
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📋 Reporte guardado en: {report_path}")
        return report_path
    
    def create_summary_table(self, analysis: Dict, output_path: Path) -> Path:
        """📊 Crea una tabla resumen de los datasets."""
        table_path = output_path / "datasets_summary_table.csv"
        
        # Crear DataFrame con información de datasets
        rows = []
        for dataset_path, info in analysis['dataset_details'].items():
            rows.append({
                'Dataset': info['name'],
                'Format': info['format'],
                'Images': info['image_count'],
                'Annotations': info['annotation_count'],
                'Classes': len(info['classes']),
                'Quality_Score': round(info['quality_score'], 1),
                'Valid': info['valid'],
                'Path': dataset_path
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values(['Quality_Score', 'Images'], ascending=[False, False])
        df.to_csv(table_path, index=False)
        
        print(f"📊 Tabla resumen guardada en: {table_path}")
        return table_path
    
    def create_visualizations(self, analysis: Dict, output_path: Path):
        """📈 Crea visualizaciones del análisis."""
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Distribución de formatos
        plt.figure(figsize=(10, 6))
        formats = list(analysis['format_distribution'].keys())
        counts = list(analysis['format_distribution'].values())
        
        plt.subplot(2, 2, 1)
        plt.pie(counts, labels=formats, autopct='%1.1f%%', startangle=90)
        plt.title('Distribución de Formatos de Dataset')
        
        # 2. Distribución de imágenes por dataset
        plt.subplot(2, 2, 2)
        image_counts = [info['image_count'] for info in analysis['dataset_details'].values()]
        plt.hist(image_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Número de Imágenes')
        plt.ylabel('Número de Datasets')
        plt.title('Distribución de Imágenes por Dataset')
        
        # 3. Puntuaciones de calidad
        plt.subplot(2, 2, 3)
        quality_scores = [info['quality_score'] for info in analysis['dataset_details'].values()]
        plt.hist(quality_scores, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Puntuación de Calidad')
        plt.ylabel('Número de Datasets')
        plt.title('Distribución de Calidad de Datasets')
        
        # 4. Top datasets por calidad
        plt.subplot(2, 2, 4)
        sorted_datasets = sorted(analysis['dataset_details'].items(), 
                               key=lambda x: x[1]['quality_score'], reverse=True)[:10]
        names = [Path(item[0]).name[:15] + '...' if len(Path(item[0]).name) > 15 
                else Path(item[0]).name for item, _ in sorted_datasets]
        scores = [info['quality_score'] for _, info in sorted_datasets]
        
        plt.barh(names, scores, color='coral')
        plt.xlabel('Puntuación de Calidad')
        plt.title('Top 10 Datasets por Calidad')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path / 'dataset_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualizaciones guardadas en: {output_path}")
