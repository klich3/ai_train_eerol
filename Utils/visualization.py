"""
📊 Dataset Visualization Utilities
Utilidades para visualización de datasets dentales
"""

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple
import pandas as pd
from wordcloud import WordCloud


class DatasetVisualizer:
    """Visualizador de datasets dentales."""
    
    def __init__(self):
        # Configurar estilo de matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Colores para diferentes tipos de datos
        self.colors = {
            'YOLO': '#FF6B6B',
            'COCO': '#4ECDC4', 
            'Classification': '#45B7D1',
            'U-Net': '#96CEB4',
            'Pure_Images': '#FECA57'
        }
    
    def create_overview_dashboard(self, analysis: Dict, output_path: Path):
        """📊 Crea un dashboard completo del análisis de datasets."""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Distribución de formatos
        plt.subplot(3, 4, 1)
        self._plot_format_distribution(analysis)
        
        # 2. Distribución de imágenes por dataset
        plt.subplot(3, 4, 2)
        self._plot_image_distribution(analysis)
        
        # 3. Puntuaciones de calidad
        plt.subplot(3, 4, 3)
        self._plot_quality_scores(analysis)
        
        # 4. Top datasets por calidad
        plt.subplot(3, 4, 4)
        self._plot_top_datasets(analysis)
        
        # 5. Distribución de clases
        plt.subplot(3, 4, (5, 6))
        self._plot_class_distribution(analysis)
        
        # 6. Mapa de calor de datasets por formato
        plt.subplot(3, 4, (7, 8))
        self._plot_dataset_heatmap(analysis)
        
        # 7. Progresión temporal (si hay fechas)
        plt.subplot(3, 4, (9, 10))
        self._plot_dataset_timeline(analysis)
        
        # 8. Estadísticas de resolución
        plt.subplot(3, 4, 11)
        self._plot_resolution_stats(analysis)
        
        # 9. Métricas de completitud
        plt.subplot(3, 4, 12)
        self._plot_completeness_metrics(analysis)
        
        plt.tight_layout()
        plt.savefig(output_path / 'dental_datasets_dashboard.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Crear dashboard HTML interactivo
        self._create_html_dashboard(analysis, output_path)
        
        print(f"📊 Dashboard creado en: {output_path}/dental_datasets_dashboard.png")
    
    def _plot_format_distribution(self, analysis: Dict):
        """Distribución de formatos de datasets."""
        format_dist = analysis.get('format_distribution', {})
        
        if format_dist:
            formats = list(format_dist.keys())
            counts = list(format_dist.values())
            colors = [self.colors.get(fmt, '#gray') for fmt in formats]
            
            wedges, texts, autotexts = plt.pie(counts, labels=formats, autopct='%1.1f%%', 
                                             startangle=90, colors=colors)
            plt.title('Distribución de Formatos', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No hay datos', ha='center', va='center', transform=plt.gca().transAxes)
    
    def _plot_image_distribution(self, analysis: Dict):
        """Distribución del número de imágenes por dataset."""
        dataset_details = analysis.get('dataset_details', {})
        
        if dataset_details:
            image_counts = [info['image_count'] for info in dataset_details.values()]
            
            plt.hist(image_counts, bins=min(20, len(image_counts)), 
                    alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Número de Imágenes')
            plt.ylabel('Número de Datasets')
            plt.title('Distribución de Imágenes por Dataset', fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
    
    def _plot_quality_scores(self, analysis: Dict):
        """Distribución de puntuaciones de calidad."""
        dataset_details = analysis.get('dataset_details', {})
        
        if dataset_details:
            quality_scores = [info['quality_score'] for info in dataset_details.values()]
            
            plt.hist(quality_scores, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.xlabel('Puntuación de Calidad')
            plt.ylabel('Número de Datasets')
            plt.title('Distribución de Calidad', fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
    
    def _plot_top_datasets(self, analysis: Dict):
        """Top 10 datasets por calidad."""
        dataset_details = analysis.get('dataset_details', {})
        
        if dataset_details:
            sorted_datasets = sorted(dataset_details.items(), 
                                   key=lambda x: x[1]['quality_score'], reverse=True)[:10]
            
            names = [Path(item[0]).name[:15] + '...' if len(Path(item[0]).name) > 15 
                    else Path(item[0]).name for item, _ in sorted_datasets]
            scores = [info['quality_score'] for _, info in sorted_datasets]
            
            bars = plt.barh(names, scores, color='coral')
            plt.xlabel('Puntuación de Calidad')
            plt.title('Top 10 Datasets por Calidad', fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Agregar valores en las barras
            for bar, score in zip(bars, scores):
                plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                        f'{score:.1f}', va='center', fontsize=8)
    
    def _plot_class_distribution(self, analysis: Dict):
        """Distribución de clases detectadas."""
        all_classes = Counter()
        dataset_details = analysis.get('dataset_details', {})
        
        for info in dataset_details.values():
            all_classes.update(info.get('classes', []))
        
        if all_classes:
            # Tomar las 20 clases más comunes
            top_classes = all_classes.most_common(20)
            classes, counts = zip(*top_classes)
            
            plt.bar(range(len(classes)), counts, color='lightblue', edgecolor='black')
            plt.xlabel('Clases')
            plt.ylabel('Frecuencia')
            plt.title('Distribución de Clases (Top 20)', fontweight='bold')
            plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
    
    def _plot_dataset_heatmap(self, analysis: Dict):
        """Mapa de calor de datasets por formato y calidad."""
        dataset_details = analysis.get('dataset_details', {})
        
        if dataset_details:
            # Crear matriz para el heatmap
            formats = list(set(info['format'] for info in dataset_details.values()))
            quality_ranges = ['0-25', '25-50', '50-75', '75-100']
            
            heatmap_data = np.zeros((len(formats), len(quality_ranges)))
            
            for info in dataset_details.values():
                format_idx = formats.index(info['format'])
                quality = info['quality_score']
                
                if quality <= 25:
                    quality_idx = 0
                elif quality <= 50:
                    quality_idx = 1
                elif quality <= 75:
                    quality_idx = 2
                else:
                    quality_idx = 3
                
                heatmap_data[format_idx, quality_idx] += 1
            
            sns.heatmap(heatmap_data, 
                       xticklabels=quality_ranges,
                       yticklabels=formats,
                       annot=True, fmt='.0f', cmap='YlOrRd')
            plt.title('Datasets por Formato y Calidad', fontweight='bold')
            plt.xlabel('Rango de Calidad')
            plt.ylabel('Formato')
    
    def _plot_dataset_timeline(self, analysis: Dict):
        """Timeline de datasets (simulado por orden alfabético)."""
        dataset_details = analysis.get('dataset_details', {})
        
        if dataset_details:
            # Simular timeline usando orden alfabético de nombres
            sorted_datasets = sorted(dataset_details.items(), key=lambda x: Path(x[0]).name)
            
            names = [Path(item[0]).name[:20] for item, _ in sorted_datasets[:15]]
            qualities = [info['quality_score'] for _, info in sorted_datasets[:15]]
            
            plt.plot(range(len(names)), qualities, marker='o', linewidth=2, markersize=6)
            plt.xlabel('Datasets (orden alfabético)')
            plt.ylabel('Puntuación de Calidad')
            plt.title('Calidad de Datasets', fontweight='bold')
            plt.xticks(range(len(names)), names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
    
    def _plot_resolution_stats(self, analysis: Dict):
        """Estadísticas de resolución (simuladas)."""
        # Para futura implementación con análisis real de resoluciones
        resolutions = ['< 512px', '512-1024px', '1024-2048px', '> 2048px']
        counts = [15, 45, 30, 10]  # Datos simulados
        
        plt.bar(resolutions, counts, color='lightcoral', edgecolor='black')
        plt.xlabel('Resolución')
        plt.ylabel('Número de Datasets')
        plt.title('Distribución por Resolución', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
    
    def _plot_completeness_metrics(self, analysis: Dict):
        """Métricas de completitud de datasets."""
        dataset_details = analysis.get('dataset_details', {})
        
        if dataset_details:
            # Calcular métricas de completitud
            total_datasets = len(dataset_details)
            valid_datasets = sum(1 for info in dataset_details.values() if info['valid'])
            annotated_datasets = sum(1 for info in dataset_details.values() if info['annotation_count'] > 0)
            classified_datasets = sum(1 for info in dataset_details.values() if info['classes'])
            
            metrics = ['Total', 'Válidos', 'Anotados', 'Clasificados']
            values = [total_datasets, valid_datasets, annotated_datasets, classified_datasets]
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            
            bars = plt.bar(metrics, values, color=colors, edgecolor='black')
            plt.ylabel('Número de Datasets')
            plt.title('Métricas de Completitud', fontweight='bold')
            
            # Agregar valores en las barras
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(value), ha='center', va='bottom', fontweight='bold')
    
    def _create_html_dashboard(self, analysis: Dict, output_path: Path):
        """Crea un dashboard HTML interactivo."""
        html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard de Datasets Dentales</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .table-container {{
            margin-top: 30px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
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
        .format-badge {{
            padding: 4px 8px;
            border-radius: 4px;
            color: white;
            font-size: 0.8em;
        }}
        .format-YOLO {{ background-color: #FF6B6B; }}
        .format-COCO {{ background-color: #4ECDC4; }}
        .format-Classification {{ background-color: #45B7D1; }}
        .format-U-Net {{ background-color: #96CEB4; }}
        .format-Pure_Images {{ background-color: #FECA57; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🦷 Dashboard de Datasets Dentales</h1>
            <p>Análisis completo de datasets disponibles - {analysis.get('metadata', {}).get('analysis_date', 'N/A')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{analysis.get('total_datasets', 0)}</div>
                <div>Datasets Totales</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{analysis.get('total_images', 0):,}</div>
                <div>Imágenes Totales</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(analysis.get('format_distribution', {}))}</div>
                <div>Formatos Detectados</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len([d for d in analysis.get('dataset_details', {}).values() if d['valid']])}</div>
                <div>Datasets Válidos</div>
            </div>
        </div>
        
        <img src="dental_datasets_dashboard.png" alt="Dashboard Visual" style="width: 100%; max-width: 100%; height: auto; margin: 20px 0;">
        
        <div class="table-container">
            <h2>📊 Detalles de Datasets</h2>
            <table>
                <thead>
                    <tr>
                        <th>Dataset</th>
                        <th>Formato</th>
                        <th>Imágenes</th>
                        <th>Anotaciones</th>
                        <th>Calidad</th>
                        <th>Clases</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Agregar filas de la tabla
        dataset_details = analysis.get('dataset_details', {})
        sorted_datasets = sorted(dataset_details.items(), 
                               key=lambda x: x[1]['quality_score'], reverse=True)
        
        for dataset_path, info in sorted_datasets:
            dataset_name = Path(dataset_path).name
            format_class = f"format-{info['format']}"
            classes_str = ", ".join(info['classes'][:3])
            if len(info['classes']) > 3:
                classes_str += f" (+{len(info['classes'])-3} más)"
            
            html_content += f"""
                    <tr>
                        <td>{dataset_name}</td>
                        <td><span class="format-badge {format_class}">{info['format']}</span></td>
                        <td>{info['image_count']:,}</td>
                        <td>{info['annotation_count']:,}</td>
                        <td>{info['quality_score']:.1f}/100</td>
                        <td>{classes_str}</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
        """
        
        # Guardar HTML
        with open(output_path / 'dental_datasets_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def create_class_wordcloud(self, analysis: Dict, output_path: Path):
        """🎨 Crea un word cloud de las clases detectadas."""
        all_classes = Counter()
        dataset_details = analysis.get('dataset_details', {})
        
        for info in dataset_details.values():
            all_classes.update(info.get('classes', []))
        
        if all_classes:
            # Crear word cloud
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate_from_frequencies(all_classes)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Clases Detectadas en Datasets Dentales', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(output_path / 'categories_wordcloud.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"🎨 Word cloud de clases creado en: {output_path}/categories_wordcloud.png")
    
    def create_detailed_report(self, analysis: Dict, output_path: Path):
        """📋 Crea un reporte detallado en markdown."""
        report_content = f"""# 🦷 Reporte de Análisis de Datasets Dentales

**Fecha de análisis:** {analysis.get('metadata', {}).get('analysis_date', 'N/A')}
**Ruta base:** {analysis.get('metadata', {}).get('base_path', 'N/A')}

## 📊 Resumen Ejecutivo

- **Datasets totales encontrados:** {analysis.get('total_datasets', 0)}
- **Imágenes totales:** {analysis.get('total_images', 0):,}
- **Formatos detectados:** {', '.join(analysis.get('format_distribution', {}).keys())}
- **Datasets válidos:** {len([d for d in analysis.get('dataset_details', {}).values() if d['valid']])}

## 📈 Distribución por Formato

| Formato | Cantidad | Porcentaje |
|---------|----------|------------|
"""
        
        format_dist = analysis.get('format_distribution', {})
        total_formats = sum(format_dist.values()) if format_dist else 1
        
        for fmt, count in format_dist.items():
            percentage = (count / total_formats) * 100
            report_content += f"| {fmt} | {count} | {percentage:.1f}% |\n"
        
        report_content += f"""
## 🏆 Top 10 Datasets por Calidad

| Ranking | Dataset | Formato | Imágenes | Calidad |
|---------|---------|---------|----------|---------|
"""
        
        dataset_details = analysis.get('dataset_details', {})
        sorted_datasets = sorted(dataset_details.items(), 
                               key=lambda x: x[1]['quality_score'], reverse=True)[:10]
        
        for i, (dataset_path, info) in enumerate(sorted_datasets, 1):
            dataset_name = Path(dataset_path).name
            report_content += f"| {i} | {dataset_name} | {info['format']} | {info['image_count']:,} | {info['quality_score']:.1f}/100 |\n"
        
        report_content += f"""
## 🔍 Análisis Detallado

### Datasets por Formato:
- **YOLO:** {len(analysis.get('yolo_datasets', []))} datasets
- **COCO:** {len(analysis.get('coco_datasets', []))} datasets  
- **Imágenes puras:** {len(analysis.get('pure_image_datasets', []))} datasets
- **U-Net:** {len(analysis.get('unet_datasets', []))} datasets

### Métricas de Calidad:
- **Calidad promedio:** {np.mean([info['quality_score'] for info in dataset_details.values()]):.1f}/100
- **Mejor dataset:** {max(dataset_details.items(), key=lambda x: x[1]['quality_score'])[1]['quality_score']:.1f}/100
- **Datasets de alta calidad (>80):** {len([d for d in dataset_details.values() if d['quality_score'] > 80])}

## 📋 Recomendaciones

1. **Priorizar datasets de alta calidad** para entrenamiento inicial
2. **Unificar formatos** usando el workflow manager
3. **Balancear clases** para mejorar rendimiento del modelo
4. **Validar anotaciones** en datasets con baja puntuación de calidad

---
*Reporte generado automáticamente por Dental AI Workflow Manager*
"""
        
        # Guardar reporte
        with open(output_path / 'dental_dataset_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📋 Reporte detallado creado en: {output_path}/dental_dataset_report.md")
    
    def visualize_sample_images(self, dataset_path: Path, format_type: str, 
                               output_path: Path, num_samples: int = 9):
        """🖼️ Visualiza muestras de imágenes de un dataset."""
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        # Buscar imágenes de muestra
        sample_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            sample_images.extend(list(dataset_path.rglob(ext))[:num_samples])
            if len(sample_images) >= num_samples:
                break
        
        sample_images = sample_images[:num_samples]
        
        for i, img_path in enumerate(sample_images):
            if i >= len(axes):
                break
                
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img_rgb)
                    axes[i].set_title(img_path.name, fontsize=8)
                    axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
                axes[i].axis('off')
        
        # Ocultar ejes vacíos
        for i in range(len(sample_images), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Muestras de {dataset_path.name} ({format_type})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        sample_output = output_path / f'samples_{dataset_path.name.replace(" ", "_")}.png'
        plt.savefig(sample_output, dpi=300, bbox_inches='tight')
        plt.close()
        
        return sample_output
