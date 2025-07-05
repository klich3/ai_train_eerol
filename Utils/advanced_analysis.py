"""
📊 Advanced Dataset Analysis Module
Módulo de análisis avanzado para datasets dentales
Integra funcionalidades de AnalizeDataSets.py y ReadStatistics.py
"""

import os
import json
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")


class AdvancedDatasetAnalyzer:
    """
    Analizador avanzado de datasets dentales integrado.
    Combina análisis profundo con visualización interactiva.
    """
    
    def __init__(self, base_path: str, output_path: str = "StatisticsResults"):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        self.results = {}
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Tipos de dataset soportados
        self.dataset_types = {
            '_YOLO': 'YOLO Object Detection',
            '_COCO': 'COCO Format (Detection/Segmentation)', 
            '_UNET': 'U-Net Semantic Segmentation',
            '_pure images and masks': 'Pure Images/Classification'
        }
        
        # Categorías dentales comunes
        self.common_dental_categories = {
            'caries', 'cavity', 'tooth', 'teeth', 'molar', 'premolar', 'canine', 'incisor',
            'root', 'crown', 'decay', 'filling', 'implant', 'bridge', 'orthodontic',
            'panoramic', 'periapical', 'bitewing', 'dental', 'radiograph', 'xray',
            'mandible', 'maxilla', 'jaw', 'bone', 'nerve', 'periodontitis', 'gingivitis'
        }
        
        # Colores para visualizaciones
        self.colors = {
            'YOLO': '#FF6B6B',
            'COCO': '#4ECDC4', 
            'UNET': '#45B7D1',
            'Classification': '#96CEB4',
            'Unknown': '#95A5A6'
        }
    
    def analyze_all_datasets(self) -> Dict[str, Any]:
        """🔍 Análisis completo de todos los datasets disponibles."""
        print("🔍 INICIANDO ANÁLISIS COMPLETO DE DATASETS...")
        
        analysis_results = {
            'analysis_date': datetime.now().isoformat(),
            'base_path': str(self.base_path),
            'total_datasets': 0,
            'total_images': 0,
            'format_distribution': Counter(),
            'quality_metrics': {},
            'dataset_details': {}
        }
        
        # Analizar cada tipo de dataset
        for folder_type, description in self.dataset_types.items():
            folder_path = self.base_path / folder_type
            
            if folder_path.exists():
                print(f"  📁 Analizando {folder_type} ({description})")
                
                folder_results = self._analyze_dataset_folder(folder_path, folder_type)
                analysis_results[folder_type] = folder_results
                
                # Actualizar estadísticas globales
                analysis_results['total_datasets'] += folder_results.get('dataset_count', 0)
                analysis_results['total_images'] += folder_results.get('total_images', 0)
                
                # Actualizar distribución de formatos
                format_type = self._get_format_from_folder_type(folder_type)
                analysis_results['format_distribution'][format_type] += folder_results.get('dataset_count', 0)
        
        # Calcular métricas de calidad globales
        analysis_results['quality_metrics'] = self._calculate_global_quality_metrics(analysis_results)
        
        # Guardar resultados
        output_file = self.output_path / 'dental_dataset_analysis_advanced.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ Análisis completo guardado en: {output_file}")
        return analysis_results
    
    def _analyze_dataset_folder(self, folder_path: Path, folder_type: str) -> Dict[str, Any]:
        """Analiza una carpeta específica de datasets."""
        folder_results = {
            'type': self.dataset_types[folder_type],
            'path': str(folder_path),
            'dataset_count': 0,
            'total_images': 0,
            'datasets': {}
        }
        
        # Buscar subdirectorios (cada uno es un dataset)
        for dataset_dir in folder_path.iterdir():
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                dataset_info = self._analyze_individual_dataset(dataset_dir, folder_type)
                
                if dataset_info['valid']:
                    folder_results['datasets'][dataset_dir.name] = dataset_info
                    folder_results['dataset_count'] += 1
                    folder_results['total_images'] += dataset_info['image_count']
        
        return folder_results
    
    def _analyze_individual_dataset(self, dataset_path: Path, folder_type: str) -> Dict[str, Any]:
        """Analiza un dataset individual en profundidad."""
        dataset_info = {
            'path': str(dataset_path),
            'name': dataset_path.name,
            'valid': False,
            'image_count': 0,
            'annotation_count': 0,
            'classes': [],
            'file_structure': {},
            'quality_score': 0.0,
            'image_resolutions': [],
            'annotation_quality': {},
            'format_compliance': {}
        }
        
        try:
            # Análisis de archivos
            self._analyze_files(dataset_path, dataset_info)
            
            # Análisis específico por formato
            if folder_type == '_YOLO':
                self._analyze_yolo_format(dataset_path, dataset_info)
            elif folder_type == '_COCO':
                self._analyze_coco_format(dataset_path, dataset_info)
            elif folder_type == '_UNET':
                self._analyze_unet_format(dataset_path, dataset_info)
            else:
                self._analyze_classification_format(dataset_path, dataset_info)
            
            # Calcular puntuación de calidad
            dataset_info['quality_score'] = self._calculate_dataset_quality(dataset_info)
            
            # Marcar como válido si tiene contenido
            if dataset_info['image_count'] > 0:
                dataset_info['valid'] = True
        
        except Exception as e:
            print(f"⚠️ Error analizando {dataset_path}: {e}")
        
        return dataset_info
    
    def _analyze_files(self, dataset_path: Path, dataset_info: Dict):
        """Analiza la estructura de archivos del dataset."""
        image_files = []
        annotation_files = []
        other_files = []
        
        for file_path in dataset_path.rglob('*'):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                
                if suffix in self.supported_image_formats:
                    image_files.append(file_path)
                elif suffix in {'.txt', '.json', '.xml', '.yaml', '.yml'}:
                    annotation_files.append(file_path)
                else:
                    other_files.append(file_path)
        
        dataset_info['image_count'] = len(image_files)
        dataset_info['annotation_count'] = len(annotation_files)
        
        dataset_info['file_structure'] = {
            'images': len(image_files),
            'annotations': len(annotation_files),
            'other_files': len(other_files),
            'total_files': len(image_files) + len(annotation_files) + len(other_files)
        }
        
        # Analizar algunas imágenes para obtener resoluciones
        if image_files:
            sample_images = image_files[:min(10, len(image_files))]
            resolutions = []
            
            for img_path in sample_images:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        resolutions.append((w, h))
                except:
                    continue
            
            dataset_info['image_resolutions'] = resolutions
    
    def _analyze_yolo_format(self, dataset_path: Path, dataset_info: Dict):
        """Análisis específico para formato YOLO."""
        # Buscar archivo data.yaml o classes.txt
        data_yaml = dataset_path / 'data.yaml'
        classes_txt = dataset_path / 'classes.txt'
        
        classes = []
        
        if data_yaml.exists():
            try:
                with open(data_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        classes = data['names']
                        if isinstance(classes, dict):
                            classes = list(classes.values())
            except:
                pass
        
        if not classes and classes_txt.exists():
            try:
                with open(classes_txt, 'r') as f:
                    classes = [line.strip() for line in f if line.strip()]
            except:
                pass
        
        dataset_info['classes'] = classes
        
        # Verificar estructura de directorios YOLO
        yolo_structure = {
            'has_train_images': (dataset_path / 'train' / 'images').exists(),
            'has_train_labels': (dataset_path / 'train' / 'labels').exists(),
            'has_val_images': (dataset_path / 'val' / 'images').exists(),
            'has_val_labels': (dataset_path / 'val' / 'labels').exists(),
            'has_data_yaml': data_yaml.exists(),
            'has_classes_txt': classes_txt.exists()
        }
        
        dataset_info['format_compliance'] = yolo_structure
        
        # Analizar calidad de anotaciones
        self._analyze_yolo_annotations_quality(dataset_path, dataset_info)
    
    def _analyze_coco_format(self, dataset_path: Path, dataset_info: Dict):
        """Análisis específico para formato COCO."""
        json_files = list(dataset_path.rglob('*.json'))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    coco_data = json.load(f)
                
                if 'categories' in coco_data:
                    classes = [cat['name'] for cat in coco_data['categories']]
                    dataset_info['classes'] = classes
                
                # Verificar estructura COCO
                coco_structure = {
                    'has_images': 'images' in coco_data,
                    'has_annotations': 'annotations' in coco_data,
                    'has_categories': 'categories' in coco_data,
                    'image_count': len(coco_data.get('images', [])),
                    'annotation_count': len(coco_data.get('annotations', [])),
                    'category_count': len(coco_data.get('categories', []))
                }
                
                dataset_info['format_compliance'] = coco_structure
                break
                
            except:
                continue
    
    def _analyze_unet_format(self, dataset_path: Path, dataset_info: Dict):
        """Análisis específico para formato U-Net."""
        # Buscar máscaras y imágenes
        image_dirs = []
        mask_dirs = []
        
        for item in dataset_path.rglob('*'):
            if item.is_dir():
                dir_name = item.name.lower()
                if 'image' in dir_name or 'img' in dir_name:
                    image_dirs.append(item)
                elif 'mask' in dir_name or 'label' in dir_name or 'gt' in dir_name:
                    mask_dirs.append(item)
        
        unet_structure = {
            'image_directories': len(image_dirs),
            'mask_directories': len(mask_dirs),
            'has_paired_structure': len(image_dirs) > 0 and len(mask_dirs) > 0
        }
        
        dataset_info['format_compliance'] = unet_structure
    
    def _analyze_classification_format(self, dataset_path: Path, dataset_info: Dict):
        """Análisis específico para formato de clasificación."""
        class_dirs = []
        
        for item in dataset_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                class_dirs.append(item.name)
        
        dataset_info['classes'] = class_dirs
        
        classification_structure = {
            'class_count': len(class_dirs),
            'classes': class_dirs,
            'has_class_structure': len(class_dirs) > 1
        }
        
        dataset_info['format_compliance'] = classification_structure
    
    def _analyze_yolo_annotations_quality(self, dataset_path: Path, dataset_info: Dict):
        """Analiza la calidad de las anotaciones YOLO."""
        quality_metrics = {
            'valid_annotations': 0,
            'invalid_annotations': 0,
            'bbox_issues': 0,
            'class_issues': 0
        }
        
        # Buscar archivos de anotación
        for txt_file in dataset_path.rglob('*.txt'):
            if txt_file.parent.name == 'labels':
                try:
                    with open(txt_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split()
                            if len(parts) != 5:
                                quality_metrics['invalid_annotations'] += 1
                                continue
                            
                            try:
                                class_id = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])
                                
                                # Verificar rangos válidos
                                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                    quality_metrics['bbox_issues'] += 1
                                else:
                                    quality_metrics['valid_annotations'] += 1
                                
                            except ValueError:
                                quality_metrics['class_issues'] += 1
                                
                except:
                    continue
        
        dataset_info['annotation_quality'] = quality_metrics
    
    def _calculate_dataset_quality(self, dataset_info: Dict) -> float:
        """Calcula una puntuación de calidad para el dataset."""
        score = 0.0
        
        # Factor 1: Cantidad de imágenes (max 25 puntos)
        image_count = dataset_info['image_count']
        if image_count >= 1000:
            score += 25
        elif image_count >= 500:
            score += 20
        elif image_count >= 100:
            score += 15
        elif image_count >= 50:
            score += 10
        elif image_count >= 10:
            score += 5
        
        # Factor 2: Presencia de anotaciones (max 25 puntos)
        annotation_count = dataset_info['annotation_count']
        if annotation_count > 0:
            ratio = annotation_count / max(image_count, 1)
            score += min(ratio * 25, 25)
        
        # Factor 3: Cumplimiento de formato (max 25 puntos)
        format_compliance = dataset_info.get('format_compliance', {})
        if format_compliance:
            compliance_score = sum(1 for v in format_compliance.values() if v) / len(format_compliance)
            score += compliance_score * 25
        
        # Factor 4: Calidad de anotaciones (max 25 puntos)
        annotation_quality = dataset_info.get('annotation_quality', {})
        if annotation_quality:
            valid = annotation_quality.get('valid_annotations', 0)
            invalid = annotation_quality.get('invalid_annotations', 0)
            total = valid + invalid
            
            if total > 0:
                quality_ratio = valid / total
                score += quality_ratio * 25
            else:
                score += 10  # Puntuación por defecto si no hay anotaciones que evaluar
        else:
            score += 15  # Puntuación por defecto para datasets sin anotaciones evaluables
        
        return min(score, 100.0)
    
    def _calculate_global_quality_metrics(self, analysis_results: Dict) -> Dict[str, Any]:
        """Calcula métricas de calidad globales."""
        all_datasets = []
        
        for folder_type in self.dataset_types.keys():
            if folder_type in analysis_results:
                datasets = analysis_results[folder_type].get('datasets', {})
                all_datasets.extend(datasets.values())
        
        if not all_datasets:
            return {}
        
        quality_scores = [ds['quality_score'] for ds in all_datasets if ds.get('quality_score', 0) > 0]
        image_counts = [ds['image_count'] for ds in all_datasets]
        
        return {
            'average_quality': np.mean(quality_scores) if quality_scores else 0,
            'median_quality': np.median(quality_scores) if quality_scores else 0,
            'max_quality': max(quality_scores) if quality_scores else 0,
            'min_quality': min(quality_scores) if quality_scores else 0,
            'high_quality_datasets': len([q for q in quality_scores if q >= 80]),
            'medium_quality_datasets': len([q for q in quality_scores if 60 <= q < 80]),
            'low_quality_datasets': len([q for q in quality_scores if q < 60]),
            'average_images_per_dataset': np.mean(image_counts) if image_counts else 0,
            'total_valid_datasets': len(all_datasets)
        }
    
    def _get_format_from_folder_type(self, folder_type: str) -> str:
        """Convierte el tipo de carpeta al formato correspondiente."""
        mapping = {
            '_YOLO': 'YOLO',
            '_COCO': 'COCO',
            '_UNET': 'UNET',
            '_pure images and masks': 'Classification'
        }
        return mapping.get(folder_type, 'Unknown')
    
    def create_comprehensive_report(self, analysis_data: Dict = None) -> Path:
        """📋 Crea un reporte completo con visualizaciones."""
        if analysis_data is None:
            # Cargar datos existentes o realizar análisis
            analysis_file = self.output_path / 'dental_dataset_analysis_advanced.json'
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    analysis_data = json.load(f)
            else:
                analysis_data = self.analyze_all_datasets()
        
        print("📋 Generando reporte completo...")
        
        # Crear visualizaciones
        self._create_advanced_visualizations(analysis_data)
        
        # Crear reporte HTML
        report_path = self._create_html_report(analysis_data)
        
        # Crear CSV con detalles
        self._create_csv_summary(analysis_data)
        
        print(f"✅ Reporte completo generado en: {report_path}")
        return report_path
    
    def _create_advanced_visualizations(self, analysis_data: Dict):
        """Crea visualizaciones avanzadas."""
        # 1. Dashboard de calidad
        self._create_quality_dashboard(analysis_data)
        
        # 2. Análisis de distribución de imágenes
        self._create_image_distribution_analysis(analysis_data)
        
        # 3. Mapa de calor de formatos vs calidad
        self._create_format_quality_heatmap(analysis_data)
        
        # 4. Análisis de resoluciones
        self._create_resolution_analysis(analysis_data)
    
    def _create_quality_dashboard(self, analysis_data: Dict):
        """Crea dashboard de calidad de datasets."""
        plt.figure(figsize=(20, 12))
        
        # Recopilar datos de calidad
        datasets_info = []
        for folder_type in self.dataset_types.keys():
            if folder_type in analysis_data:
                datasets = analysis_data[folder_type].get('datasets', {})
                for name, info in datasets.items():
                    datasets_info.append({
                        'name': name,
                        'format': self._get_format_from_folder_type(folder_type),
                        'quality': info.get('quality_score', 0),
                        'images': info.get('image_count', 0),
                        'annotations': info.get('annotation_count', 0)
                    })
        
        if not datasets_info:
            return
        
        df = pd.DataFrame(datasets_info)
        
        # Subplot 1: Distribución de calidad
        plt.subplot(2, 3, 1)
        df['quality'].hist(bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Puntuación de Calidad')
        plt.ylabel('Número de Datasets')
        plt.title('Distribución de Calidad de Datasets')
        plt.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Calidad por formato
        plt.subplot(2, 3, 2)
        formats = df['format'].unique()
        quality_by_format = [df[df['format'] == fmt]['quality'].values for fmt in formats]
        plt.boxplot(quality_by_format, labels=formats)
        plt.ylabel('Puntuación de Calidad')
        plt.title('Calidad por Formato')
        plt.xticks(rotation=45)
        
        # Subplot 3: Top 10 datasets por calidad
        plt.subplot(2, 3, 3)
        top_datasets = df.nlargest(10, 'quality')
        plt.barh(range(len(top_datasets)), top_datasets['quality'], color='lightgreen')
        plt.yticks(range(len(top_datasets)), [name[:20] + '...' if len(name) > 20 else name 
                                           for name in top_datasets['name']])
        plt.xlabel('Puntuación de Calidad')
        plt.title('Top 10 Datasets por Calidad')
        plt.gca().invert_yaxis()
        
        # Subplot 4: Scatter plot calidad vs imágenes
        plt.subplot(2, 3, 4)
        colors = [self.colors.get(fmt, '#gray') for fmt in df['format']]
        plt.scatter(df['images'], df['quality'], c=colors, alpha=0.6, s=60)
        plt.xlabel('Número de Imágenes')
        plt.ylabel('Puntuación de Calidad')
        plt.title('Calidad vs Cantidad de Imágenes')
        plt.grid(True, alpha=0.3)
        
        # Agregar leyenda
        for fmt, color in self.colors.items():
            if fmt in df['format'].values:
                plt.scatter([], [], c=color, label=fmt, s=60)
        plt.legend()
        
        # Subplot 5: Ratio anotaciones/imágenes
        plt.subplot(2, 3, 5)
        df['annotation_ratio'] = df['annotations'] / df['images'].replace(0, 1)
        df['annotation_ratio'].hist(bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Ratio Anotaciones/Imágenes')
        plt.ylabel('Número de Datasets')
        plt.title('Distribución de Ratio Anotaciones/Imágenes')
        plt.grid(axis='y', alpha=0.3)
        
        # Subplot 6: Métricas generales
        plt.subplot(2, 3, 6)
        metrics = analysis_data.get('quality_metrics', {})
        metric_names = ['Alta Calidad\n(≥80)', 'Calidad Media\n(60-79)', 'Baja Calidad\n(<60)']
        metric_values = [
            metrics.get('high_quality_datasets', 0),
            metrics.get('medium_quality_datasets', 0),
            metrics.get('low_quality_datasets', 0)
        ]
        
        colors_pie = ['#2ECC71', '#F39C12', '#E74C3C']
        plt.pie(metric_values, labels=metric_names, colors=colors_pie, autopct='%1.1f%%')
        plt.title('Distribución por Nivel de Calidad')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'quality_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_image_distribution_analysis(self, analysis_data: Dict):
        """Análisis de distribución de imágenes por dataset."""
        # Implementación de análisis de distribución
        pass
    
    def _create_format_quality_heatmap(self, analysis_data: Dict):
        """Mapa de calor de formatos vs calidad."""
        # Implementación de heatmap
        pass
    
    def _create_resolution_analysis(self, analysis_data: Dict):
        """Análisis de resoluciones de imágenes."""
        # Implementación de análisis de resoluciones
        pass
    
    def _create_html_report(self, analysis_data: Dict) -> Path:
        """Crea reporte HTML completo."""
        html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Avanzado de Datasets Dentales</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .quality-excellent {{ background-color: #d4edda; }}
        .quality-good {{ background-color: #fff3cd; }}
        .quality-poor {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦷 Reporte Avanzado de Datasets Dentales</h1>
            <p>Análisis completo generado el {analysis_data.get('analysis_date', 'N/A')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{analysis_data.get('total_datasets', 0)}</div>
                <div>Datasets Totales</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{analysis_data.get('total_images', 0):,}</div>
                <div>Imágenes Totales</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(analysis_data.get('format_distribution', {}))}</div>
                <div>Formatos Detectados</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{analysis_data.get('quality_metrics', {}).get('average_quality', 0):.1f}</div>
                <div>Calidad Promedio</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2>📊 Dashboard de Calidad</h2>
            <img src="quality_dashboard.png" alt="Dashboard de Calidad" style="max-width: 100%; height: auto;">
        </div>
        
        <h2>📋 Detalles por Formato</h2>
        <table>
            <thead>
                <tr>
                    <th>Formato</th>
                    <th>Datasets</th>
                    <th>Imágenes Totales</th>
                    <th>Calidad Promedio</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Agregar filas por formato
        for folder_type in self.dataset_types.keys():
            if folder_type in analysis_data:
                folder_data = analysis_data[folder_type]
                format_name = self._get_format_from_folder_type(folder_type)
                
                datasets = folder_data.get('datasets', {})
                total_images = folder_data.get('total_images', 0)
                
                # Calcular calidad promedio
                qualities = [ds.get('quality_score', 0) for ds in datasets.values()]
                avg_quality = np.mean(qualities) if qualities else 0
                
                # Determinar clase CSS para calidad
                quality_class = 'quality-excellent' if avg_quality >= 80 else \
                               'quality-good' if avg_quality >= 60 else 'quality-poor'
                
                html_content += f"""
                <tr class="{quality_class}">
                    <td>{format_name}</td>
                    <td>{len(datasets)}</td>
                    <td>{total_images:,}</td>
                    <td>{avg_quality:.1f}/100</td>
                </tr>
                """
        
        html_content += """
            </tbody>
        </table>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #eee; text-align: center; color: #666;">
            <p>Reporte generado por Dental AI Workflow Manager v2.0</p>
        </div>
    </div>
</body>
</html>
        """
        
        report_path = self.output_path / 'advanced_analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _create_csv_summary(self, analysis_data: Dict):
        """Crea resumen en CSV para análisis posterior."""
        rows = []
        
        for folder_type in self.dataset_types.keys():
            if folder_type in analysis_data:
                folder_data = analysis_data[folder_type]
                datasets = folder_data.get('datasets', {})
                
                for name, info in datasets.items():
                    rows.append({
                        'Dataset': name,
                        'Format': self._get_format_from_folder_type(folder_type),
                        'Images': info.get('image_count', 0),
                        'Annotations': info.get('annotation_count', 0),
                        'Quality_Score': info.get('quality_score', 0),
                        'Valid': info.get('valid', False),
                        'Classes': len(info.get('classes', [])),
                        'Path': info.get('path', '')
                    })
        
        df = pd.DataFrame(rows)
        csv_path = self.output_path / 'datasets_detailed_summary.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Resumen CSV guardado en: {csv_path}")


def main():
    """Función principal para ejecutar el análisis avanzado."""
    print("🔍 ANÁLISIS AVANZADO DE DATASETS DENTALES")
    print("="*50)
    
    # Configurar paths
    base_path = "_dataSets"
    output_path = "StatisticsResults"
    
    # Verificar que existe el directorio base
    if not Path(base_path).exists():
        print(f"❌ Directorio base no encontrado: {base_path}")
        return
    
    # Crear analizador
    analyzer = AdvancedDatasetAnalyzer(base_path, output_path)
    
    # Ejecutar análisis completo
    analysis_results = analyzer.analyze_all_datasets()
    
    # Generar reporte completo
    report_path = analyzer.create_comprehensive_report(analysis_results)
    
    print(f"\n🎉 ANÁLISIS COMPLETO FINALIZADO")
    print(f"📁 Resultados en: {output_path}/")
    print(f"📋 Reporte principal: {report_path}")
    print(f"📊 Dashboard: {output_path}/quality_dashboard.png")
    print(f"📈 CSV detallado: {output_path}/datasets_detailed_summary.csv")


if __name__ == "__main__":
    main()
