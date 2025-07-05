"""
🧠 Analizador Inteligente de Categorías Dentales
===============================================

Sistema avanzado para detectar y analizar categorías en datasets dentales
con mapeo inteligente de clases y análisis de distribución.

Author: Anton Sychev
Version: 3.0
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from tqdm import tqdm


class SmartCategoryAnalyzer:
    """🧠 Analizador inteligente de categorías dentales."""
    
    def __init__(self, unified_classes: Dict[str, List[str]]):
        """Inicializar analizador."""
        self.unified_classes = unified_classes
        self.detected_categories = {}
        self.category_stats = {}
        self.unmatched_classes = set()
        
        # Patrones comunes para detección inteligente
        self.dental_patterns = {
            'caries': [r'caries', r'cavit', r'decay', r'hole', r'carries'],
            'tooth': [r'tooth', r'teeth', r'diente', r'molar', r'premolar', r'canine', r'incisor'],
            'crown': [r'crown', r'cap', r'corona'],
            'filling': [r'fill', r'restor', r'amalgam', r'composite'],
            'root_canal': [r'root.*canal', r'endodontic', r'treated.*tooth'],
            'implant': [r'implant', r'screw', r'fixture'],
            'periapical': [r'periapical', r'lesion', r'abscess', r'infection'],
            'bone': [r'bone.*loss', r'alveolar', r'periodontal'],
            'impacted': [r'impacted', r'wisdom', r'third.*molar'],
            'orthodontic': [r'bracket', r'brace', r'wire', r'orthodontic'],
            'prosthetic': [r'denture', r'prosthetic', r'artificial']
        }
    
    def analyze_dataset_categories(self, dataset_path: Path, format_type: str) -> Dict[str, Any]:
        """🔍 Analizar categorías en un dataset específico."""
        categories_info = {
            'path': str(dataset_path),
            'format': format_type,
            'detected_classes': {},
            'unified_mapping': {},
            'class_counts': {},
            'quality_indicators': {},
            'recommendations': []
        }
        
        try:
            if format_type == 'YOLO':
                categories_info.update(self._analyze_yolo_categories(dataset_path))
            elif format_type == 'COCO':
                categories_info.update(self._analyze_coco_categories(dataset_path))
            elif format_type == 'Classification':
                categories_info.update(self._analyze_classification_categories(dataset_path))
            elif format_type == 'Segmentation_Images':
                categories_info.update(self._analyze_segmentation_categories(dataset_path))
            
            # Aplicar mapeo inteligente
            self._apply_intelligent_mapping(categories_info)
            
            # Calcular estadísticas
            self._calculate_category_statistics(categories_info)
            
            # Generar recomendaciones
            self._generate_recommendations(categories_info)
            
        except Exception as e:
            print(f"⚠️ Error analizando categorías en {dataset_path}: {e}")
        
        return categories_info
    
    def _analyze_yolo_categories(self, dataset_path: Path) -> Dict[str, Any]:
        """Analizar categorías en dataset YOLO."""
        yolo_info = {
            'detected_classes': {},
            'class_counts': {},
            'annotation_files': []
        }
        
        # Buscar archivos de clases
        classes_files = list(dataset_path.rglob('*.names')) + list(dataset_path.rglob('classes.txt'))
        
        if classes_files:
            # Leer desde archivo de clases
            classes_file = classes_files[0]
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f.readlines() if line.strip()]
                
                for i, class_name in enumerate(classes):
                    yolo_info['detected_classes'][i] = class_name
                    
            except Exception as e:
                print(f"⚠️ Error leyendo {classes_file}: {e}")
        
        # Analizar archivos de anotaciones para contar
        annotation_files = list(dataset_path.rglob('*.txt'))
        annotation_files = [f for f in annotation_files if f.name not in ['classes.txt', 'README.txt']]
        
        class_counter = Counter()
        
        for ann_file in annotation_files[:100]:  # Muestra limitada para velocidad
            try:
                with open(ann_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            class_counter[class_id] += 1
            except Exception:
                continue
        
        # Mapear conteos a nombres de clase
        for class_id, count in class_counter.items():
            class_name = yolo_info['detected_classes'].get(class_id, f'class_{class_id}')
            yolo_info['class_counts'][class_name] = count
        
        yolo_info['annotation_files'] = len(annotation_files)
        
        return yolo_info
    
    def _analyze_coco_categories(self, dataset_path: Path) -> Dict[str, Any]:
        """Analizar categorías en dataset COCO."""
        coco_info = {
            'detected_classes': {},
            'class_counts': {},
            'annotation_files': []
        }
        
        # Buscar archivos JSON de COCO
        json_files = list(dataset_path.rglob('*.json'))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                
                # Extraer categorías
                if 'categories' in coco_data:
                    for cat in coco_data['categories']:
                        cat_id = cat.get('id', len(coco_info['detected_classes']))
                        cat_name = cat.get('name', f'category_{cat_id}')
                        coco_info['detected_classes'][cat_id] = cat_name
                
                # Contar anotaciones por categoría
                if 'annotations' in coco_data:
                    for ann in coco_data['annotations']:
                        cat_id = ann.get('category_id')
                        if cat_id in coco_info['detected_classes']:
                            cat_name = coco_info['detected_classes'][cat_id]
                            coco_info['class_counts'][cat_name] = coco_info['class_counts'].get(cat_name, 0) + 1
                
                break  # Solo procesar el primer archivo JSON válido
                
            except Exception as e:
                continue
        
        coco_info['annotation_files'] = len(json_files)
        
        return coco_info
    
    def _analyze_classification_categories(self, dataset_path: Path) -> Dict[str, Any]:
        """Analizar categorías en dataset de clasificación."""
        class_info = {
            'detected_classes': {},
            'class_counts': {},
            'annotation_files': 0
        }
        
        # Buscar estructura de directorios por clase
        subdirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        for i, subdir in enumerate(subdirs):
            class_name = subdir.name
            
            # Contar imágenes en el directorio
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_count = len([f for f in subdir.rglob('*') if f.suffix.lower() in image_extensions])
            
            if image_count > 0:
                class_info['detected_classes'][i] = class_name
                class_info['class_counts'][class_name] = image_count
        
        return class_info
    
    def _analyze_segmentation_categories(self, dataset_path: Path) -> Dict[str, Any]:
        """Analizar categorías en dataset de segmentación."""
        seg_info = {
            'detected_classes': {},
            'class_counts': {},
            'annotation_files': 0
        }
        
        # Buscar máscaras y extraer información
        mask_files = []
        for ext in ['.png', '.jpg', '.tiff', '.bmp']:
            mask_files.extend(dataset_path.rglob(f'*mask*{ext}'))
            mask_files.extend(dataset_path.rglob(f'*label*{ext}'))
        
        # Analizar nombres de archivos para detectar clases
        class_names = set()
        for mask_file in mask_files:
            # Extraer clase del nombre del archivo
            filename = mask_file.stem.lower()
            
            # Buscar patrones conocidos
            for pattern_name, patterns in self.dental_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, filename, re.IGNORECASE):
                        class_names.add(pattern_name)
                        break
        
        # Si no se encuentran patrones, usar nombres de directorio
        if not class_names:
            for subdir in dataset_path.iterdir():
                if subdir.is_dir():
                    class_names.add(subdir.name.lower())
        
        for i, class_name in enumerate(sorted(class_names)):
            seg_info['detected_classes'][i] = class_name
            seg_info['class_counts'][class_name] = len([f for f in mask_files if class_name in f.name.lower()])
        
        seg_info['annotation_files'] = len(mask_files)
        
        return seg_info
    
    def _apply_intelligent_mapping(self, categories_info: Dict[str, Any]) -> None:
        """Aplicar mapeo inteligente de clases."""
        unified_mapping = {}
        
        for class_id, class_name in categories_info['detected_classes'].items():
            unified_name = self._find_unified_class(class_name)
            unified_mapping[class_name] = unified_name
            
            if unified_name == class_name.lower() and unified_name not in self.unified_classes:
                self.unmatched_classes.add(class_name)
        
        categories_info['unified_mapping'] = unified_mapping
    
    def _find_unified_class(self, class_name: str) -> str:
        """Encontrar clase unificada para un nombre dado."""
        class_name_clean = class_name.lower().strip()
        
        # Búsqueda exacta
        for unified, variants in self.unified_classes.items():
            if any(variant.lower() == class_name_clean for variant in variants):
                return unified
        
        # Búsqueda por patrones
        for pattern_name, patterns in self.dental_patterns.items():
            for pattern in patterns:
                if re.search(pattern, class_name_clean, re.IGNORECASE):
                    # Verificar si el patrón corresponde a una clase unificada
                    if pattern_name in self.unified_classes:
                        return pattern_name
        
        # Búsqueda fuzzy (contiene)
        for unified, variants in self.unified_classes.items():
            for variant in variants:
                if variant.lower() in class_name_clean or class_name_clean in variant.lower():
                    return unified
        
        return class_name_clean
    
    def _calculate_category_statistics(self, categories_info: Dict[str, Any]) -> None:
        """Calcular estadísticas de categorías."""
        stats = {
            'total_classes': len(categories_info['detected_classes']),
            'total_annotations': sum(categories_info['class_counts'].values()),
            'class_distribution': {},
            'balance_score': 0.0,
            'coverage_score': 0.0
        }
        
        if stats['total_annotations'] > 0:
            # Distribución por clase
            for class_name, count in categories_info['class_counts'].items():
                stats['class_distribution'][class_name] = {
                    'count': count,
                    'percentage': (count / stats['total_annotations']) * 100
                }
            
            # Puntuación de balance (0-100)
            counts = list(categories_info['class_counts'].values())
            if len(counts) > 1:
                stats['balance_score'] = 100 - (np.std(counts) / np.mean(counts) * 100)
                stats['balance_score'] = max(0, min(100, stats['balance_score']))
            else:
                stats['balance_score'] = 100
            
            # Puntuación de cobertura (basada en clases unificadas detectadas)
            unified_classes_found = set(categories_info['unified_mapping'].values())
            total_unified_classes = len(self.unified_classes)
            stats['coverage_score'] = (len(unified_classes_found) / total_unified_classes) * 100
        
        categories_info['quality_indicators'] = stats
    
    def _generate_recommendations(self, categories_info: Dict[str, Any]) -> None:
        """Generar recomendaciones para el dataset."""
        recommendations = []
        stats = categories_info['quality_indicators']
        
        # Recomendaciones de balance
        if stats['balance_score'] < 50:
            recommendations.append({
                'type': 'balance',
                'priority': 'high',
                'message': 'Dataset muy desbalanceado. Considera augmentación de datos para clases minoritarias.'
            })
        elif stats['balance_score'] < 75:
            recommendations.append({
                'type': 'balance',
                'priority': 'medium',
                'message': 'Dataset moderadamente desbalanceado. Revisa distribución de clases.'
            })
        
        # Recomendaciones de cobertura
        if stats['coverage_score'] < 30:
            recommendations.append({
                'type': 'coverage',
                'priority': 'high',
                'message': 'Baja cobertura de categorías dentales. Considera agregar más tipos de datos.'
            })
        
        # Recomendaciones de tamaño
        if stats['total_annotations'] < 100:
            recommendations.append({
                'type': 'size',
                'priority': 'high',
                'message': 'Dataset muy pequeño. Necesita más muestras para entrenamiento efectivo.'
            })
        elif stats['total_annotations'] < 500:
            recommendations.append({
                'type': 'size',
                'priority': 'medium',
                'message': 'Dataset pequeño. Considera augmentación o recolección de más datos.'
            })
        
        # Recomendaciones de clases no mapeadas
        unmapped_classes = [k for k, v in categories_info['unified_mapping'].items() if v == k.lower()]
        if unmapped_classes:
            recommendations.append({
                'type': 'mapping',
                'priority': 'medium',
                'message': f'Clases no mapeadas encontradas: {", ".join(unmapped_classes[:3])}{"..." if len(unmapped_classes) > 3 else ""}. Revisa mapeo de clases.'
            })
        
        categories_info['recommendations'] = recommendations
    
    def generate_category_report(self, all_categories: Dict[str, Any], output_path: Path) -> None:
        """📊 Generar reporte completo de categorías."""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': {
                'total_datasets': len(all_categories),
                'unified_categories_found': {},
                'format_distribution': Counter(),
                'overall_quality': 0.0
            },
            'detailed_analysis': all_categories,
            'recommendations': {
                'high_priority': [],
                'medium_priority': [],
                'low_priority': []
            },
            'unmatched_classes': list(self.unmatched_classes)
        }
        
        # Calcular resumen
        all_unified_classes = set()
        total_quality_score = 0
        
        for dataset_path, info in all_categories.items():
            report['summary']['format_distribution'][info.get('format', 'unknown')] += 1
            
            # Recopilar clases unificadas
            for class_name, unified_name in info.get('unified_mapping', {}).items():
                all_unified_classes.add(unified_name)
            
            # Acumular puntuaciones de calidad
            quality_score = info.get('quality_indicators', {}).get('balance_score', 0)
            total_quality_score += quality_score
            
            # Recopilar recomendaciones por prioridad
            for rec in info.get('recommendations', []):
                priority = rec.get('priority', 'low')
                report['recommendations'][f'{priority}_priority'].append({
                    'dataset': dataset_path,
                    'message': rec['message']
                })
        
        # Calcular métricas globales
        for unified_class in all_unified_classes:
            count = sum(1 for info in all_categories.values() 
                       if unified_class in info.get('unified_mapping', {}).values())
            report['summary']['unified_categories_found'][unified_class] = count
        
        if all_categories:
            report['summary']['overall_quality'] = total_quality_score / len(all_categories)
        
        # Guardar reporte
        report_path = output_path / "category_analysis_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generar resumen en markdown
        self._generate_markdown_report(report, output_path)
        
        print(f"📊 Reporte de categorías guardado en: {report_path}")
    
    def _generate_markdown_report(self, report: Dict[str, Any], output_path: Path) -> None:
        """Generar reporte en formato markdown."""
        md_content = f"""# 📊 Reporte de Análisis de Categorías Dentales

**Fecha de análisis:** {report['timestamp']}

## 📋 Resumen Ejecutivo

- **Datasets analizados:** {report['summary']['total_datasets']}
- **Categorías unificadas encontradas:** {len(report['summary']['unified_categories_found'])}
- **Calidad promedio:** {report['summary']['overall_quality']:.1f}/100

## 🏷️ Categorías Detectadas

"""
        
        for category, count in sorted(report['summary']['unified_categories_found'].items()):
            md_content += f"- **{category}**: {count} dataset(s)\n"
        
        md_content += f"""

## 📊 Distribución por Formato

"""
        
        for format_type, count in report['summary']['format_distribution'].items():
            md_content += f"- **{format_type}**: {count} dataset(s)\n"
        
        # Recomendaciones de alta prioridad
        if report['recommendations']['high_priority']:
            md_content += f"""

## ⚠️ Recomendaciones de Alta Prioridad

"""
            for rec in report['recommendations']['high_priority'][:10]:  # Top 10
                md_content += f"- **{Path(rec['dataset']).name}**: {rec['message']}\n"
        
        # Clases no mapeadas
        if report['unmatched_classes']:
            md_content += f"""

## 🔍 Clases No Mapeadas

Las siguientes clases fueron encontradas pero no están en el mapeo unificado:

"""
            for cls in sorted(report['unmatched_classes'])[:20]:  # Top 20
                md_content += f"- `{cls}`\n"
        
        # Guardar markdown
        md_path = output_path / "category_analysis_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"📝 Reporte markdown guardado en: {md_path}")
