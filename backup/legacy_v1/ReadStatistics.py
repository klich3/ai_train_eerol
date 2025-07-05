"""
█▀ █▄█ █▀▀ █░█ █▀▀ █░█
▄█ ░█░ █▄▄ █▀█ ██▄ ▀▄▀

Author: <Anton Sychev> (anton at sychev dot xyz)
ReadStatistics.py (c) 2025
Created:  2025-07-05 17:15:42 
Desc: Previsualizador interactivo de estadísticas de datasets dentales
Docs: Genera gráficos y tablas a partir del análisis JSON de datasets
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DentalDatasetStatisticsViewer:
    """
    Visualizador de estadísticas para datasets dentales analizados.
    Genera gráficos interactivos y tablas de resumen.
    """
    
    def __init__(self, json_file: str = 'dental_dataset_analysis.json'):
        """Inicializa el visualizador con el archivo JSON de análisis."""
        self.json_file = json_file
        self.data = self._load_data()
        self.colors = {
            'YOLO': '#FF6B6B',
            'COCO': '#4ECDC4', 
            'UNET': '#45B7D1',
            'Unknown': '#96CEB4'
        }
        
    def _load_data(self) -> dict:
        """Carga los datos del archivo JSON."""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo {self.json_file}")
            print("Ejecuta primero el script de análisis para generar los datos.")
            return {}
        except json.JSONDecodeError:
            print(f"❌ Error: El archivo {self.json_file} no es un JSON válido")
            return {}
    
    def display_summary(self):
        """Muestra un resumen general de las estadísticas."""
        if not self.data or 'summary' not in self.data:
            print("❌ No hay datos de resumen disponibles")
            return
            
        summary = self.data['summary']
        
        print("\n" + "="*70)
        print("🦷 RESUMEN GENERAL DE DATASETS DENTALES")
        print("="*70)
        print(f"📊 Total de tipos de datasets: {summary.get('total_dataset_types', 0)}")
        print(f"📁 Total de datasets individuales: {summary.get('total_individual_datasets', 0)}")
        print(f"🖼️  Total de imágenes: {summary.get('total_images', 0):,}")
        print(f"🏷️  Categorías únicas: {len(summary.get('all_categories', []))}")
        
        print("\n📈 DISTRIBUCIÓN POR FORMATO:")
        for format_type, count in summary.get('format_distribution', {}).items():
            print(f"   • {format_type}: {count} datasets")
        
        print("\n🤖 ARQUITECTURAS RECOMENDADAS:")
        for arch in summary.get('recommended_architectures', []):
            print(f"   • {arch}")
        print("="*70)
    
    def create_format_distribution_chart(self, save_fig: bool = True):
        """Crea gráfico de distribución por formato."""
        if not self.data or 'summary' not in self.data:
            return
            
        summary = self.data['summary']
        format_dist = summary.get('format_distribution', {})
        
        if not format_dist:
            print("❌ No hay datos de distribución de formatos")
            return
        
        # Preparar datos
        formats = list(format_dist.keys())
        counts = list(format_dist.values())
        
        # Mapear colores
        colors_mapped = []
        for fmt in formats:
            if 'YOLO' in fmt:
                colors_mapped.append(self.colors['YOLO'])
            elif 'COCO' in fmt:
                colors_mapped.append(self.colors['COCO'])
            elif 'U-Net' in fmt:
                colors_mapped.append(self.colors['UNET'])
            else:
                colors_mapped.append(self.colors['Unknown'])
        
        # Crear subplot con matplotlib y plotly
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras
        bars = ax1.bar(range(len(formats)), counts, color=colors_mapped, alpha=0.8)
        ax1.set_title('📊 Distribución de Datasets por Formato', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Formato de Dataset')
        ax1.set_ylabel('Número de Datasets')
        ax1.set_xticks(range(len(formats)))
        ax1.set_xticklabels([f.replace(' ', '\n') for f in formats], rotation=0, ha='center')
        
        # Añadir valores en las barras
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Gráfico circular
        wedges, texts, autotexts = ax2.pie(counts, labels=formats, autopct='%1.1f%%', 
                                          colors=colors_mapped, startangle=90)
        ax2.set_title('🥧 Proporción de Formatos', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('dataset_format_distribution.png', dpi=300, bbox_inches='tight')
            print("📊 Gráfico guardado como: dataset_format_distribution.png")
        
        plt.show()
    
    def create_images_distribution_chart(self, save_fig: bool = True):
        """Crea gráfico de distribución de imágenes por dataset."""
        dataset_data = []
        
        for dataset_type, data in self.data.items():
            if dataset_type.startswith('_') and 'datasets' in data:
                for name, info in data['datasets'].items():
                    dataset_data.append({
                        'name': name[:30] + '...' if len(name) > 30 else name,
                        'full_name': name,
                        'type': data['type'],
                        'images': info.get('image_count', 0),
                        'quality': info.get('dataset_quality', '').split(' ')[0]
                    })
        
        if not dataset_data:
            print("❌ No hay datos de imágenes disponibles")
            return
        
        # Ordenar por número de imágenes
        dataset_data.sort(key=lambda x: x['images'], reverse=True)
        
        # Tomar top 15 datasets
        top_datasets = dataset_data[:15]
        
        # Crear gráfico horizontal
        fig, ax = plt.subplots(figsize=(12, 8))
        
        names = [d['name'] for d in top_datasets]
        images = [d['images'] for d in top_datasets]
        types = [d['type'] for d in top_datasets]
        
        # Mapear colores por tipo
        colors_mapped = []
        for dtype in types:
            if 'YOLO' in dtype:
                colors_mapped.append(self.colors['YOLO'])
            elif 'COCO' in dtype:
                colors_mapped.append(self.colors['COCO'])
            elif 'U-Net' in dtype:
                colors_mapped.append(self.colors['UNET'])
            else:
                colors_mapped.append(self.colors['Unknown'])
        
        bars = ax.barh(range(len(names)), images, color=colors_mapped, alpha=0.8)
        
        ax.set_title('🖼️ Top 15 Datasets por Número de Imágenes', fontsize=14, fontweight='bold')
        ax.set_xlabel('Número de Imágenes')
        ax.set_ylabel('Dataset')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        
        # Añadir valores
        for i, (bar, count) in enumerate(zip(bars, images)):
            ax.text(bar.get_width() + max(images)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', ha='left', va='center', fontweight='bold')
        
        # Invertir orden para que el mayor esté arriba
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('dataset_images_distribution.png', dpi=300, bbox_inches='tight')
            print("📊 Gráfico guardado como: dataset_images_distribution.png")
        
        plt.show()
    
    def create_quality_analysis(self, save_fig: bool = True):
        """Analiza la calidad de los datasets."""
        quality_data = defaultdict(list)
        
        for dataset_type, data in self.data.items():
            if dataset_type.startswith('_') and 'datasets' in data:
                for name, info in data['datasets'].items():
                    quality_str = info.get('dataset_quality', '')
                    if quality_str:
                        quality = quality_str.split(' ')[0]  # Extraer solo la calidad
                        quality_data[quality].append({
                            'name': name,
                            'type': data['type'],
                            'images': info.get('image_count', 0)
                        })
        
        if not quality_data:
            print("❌ No hay datos de calidad disponibles")
            return
        
        # Preparar datos para visualización
        quality_counts = {q: len(datasets) for q, datasets in quality_data.items()}
        quality_avg_images = {q: np.mean([d['images'] for d in datasets]) 
                             for q, datasets in quality_data.items()}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico 1: Distribución de calidad
        qualities = list(quality_counts.keys())
        counts = list(quality_counts.values())
        colors = ['#FF6B6B', '#FFA726', '#66BB6A', '#42A5F5']
        
        bars1 = ax1.bar(qualities, counts, color=colors[:len(qualities)], alpha=0.8)
        ax1.set_title('📈 Distribución de Calidad de Datasets', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Nivel de Calidad')
        ax1.set_ylabel('Número de Datasets')
        
        for bar, count in zip(bars1, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Gráfico 2: Promedio de imágenes por calidad
        avg_images = list(quality_avg_images.values())
        bars2 = ax2.bar(qualities, avg_images, color=colors[:len(qualities)], alpha=0.8)
        ax2.set_title('🖼️ Promedio de Imágenes por Calidad', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Nivel de Calidad')
        ax2.set_ylabel('Promedio de Imágenes')
        
        for bar, avg in zip(bars2, avg_images):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_images)*0.01,
                    f'{avg:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('dataset_quality_analysis.png', dpi=300, bbox_inches='tight')
            print("📊 Gráfico guardado como: dataset_quality_analysis.png")
        
        plt.show()
    
    def create_categories_wordcloud(self, save_fig: bool = True):
        """Crea una nube de palabras con las categorías más comunes."""
        try:
            from wordcloud import WordCloud
        except ImportError:
            print("❌ Para generar la nube de palabras, instala: pip install wordcloud")
            return
        
        if 'summary' not in self.data:
            return
        
        categories = self.data['summary'].get('all_categories', [])
        if not categories:
            print("❌ No hay categorías disponibles")
            return
        
        # Contar frecuencias de categorías
        category_counts = Counter()
        for dataset_type, data in self.data.items():
            if dataset_type.startswith('_') and 'datasets' in data:
                for name, info in data['datasets'].items():
                    cats = info.get('categories', [])
                    category_counts.update(cats)
        
        if not category_counts:
            # Si no hay conteos, usar todas las categorías con peso igual
            category_text = ' '.join(categories)
        else:
            # Crear texto con repeticiones basadas en frecuencia
            category_text = ' '.join([cat for cat, count in category_counts.items() for _ in range(count)])
        
        # Crear wordcloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=50,
            relative_scaling=0.5
        ).generate(category_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('☁️ Nube de Categorías Dentales Más Comunes', fontsize=16, fontweight='bold', pad=20)
        
        if save_fig:
            plt.savefig('categories_wordcloud.png', dpi=300, bbox_inches='tight')
            print("📊 Nube de palabras guardada como: categories_wordcloud.png")
        
        plt.show()
    
    def create_interactive_dashboard(self):
        """Crea un dashboard interactivo con Plotly."""
        if not self.data:
            print("❌ No hay datos disponibles")
            return
        
        # Preparar datos
        dataset_info = []
        for dataset_type, data in self.data.items():
            if dataset_type.startswith('_') and 'datasets' in data:
                for name, info in data['datasets'].items():
                    dataset_info.append({
                        'name': name,
                        'type': data['type'],
                        'images': info.get('image_count', 0),
                        'annotations': info.get('annotation_count', 0),
                        'quality': info.get('dataset_quality', '').split(' ')[0],
                        'categories_count': len(info.get('categories', []))
                    })
        
        df = pd.DataFrame(dataset_info)
        
        if df.empty:
            print("❌ No se pudieron procesar los datos")
            return
        
        # Crear subplot con múltiples gráficos
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Imágenes vs Anotaciones', 'Distribución por Tipo', 
                          'Calidad de Datasets', 'Categorías por Dataset'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Gráfico 1: Scatter plot Imágenes vs Anotaciones
        fig.add_trace(
            go.Scatter(
                x=df['images'], 
                y=df['annotations'],
                mode='markers',
                marker=dict(size=10, opacity=0.7),
                text=df['name'],
                name='Datasets',
                hovertemplate='<b>%{text}</b><br>Imágenes: %{x}<br>Anotaciones: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Gráfico 2: Pie chart por tipo
        type_counts = df['type'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                name="Distribución"
            ),
            row=1, col=2
        )
        
        # Gráfico 3: Calidad de datasets
        quality_counts = df['quality'].value_counts()
        fig.add_trace(
            go.Bar(
                x=quality_counts.index,
                y=quality_counts.values,
                name="Calidad",
                marker_color=['#FF6B6B', '#FFA726', '#66BB6A', '#42A5F5']
            ),
            row=2, col=1
        )
        
        # Gráfico 4: Categorías vs Imágenes
        fig.add_trace(
            go.Scatter(
                x=df['categories_count'],
                y=df['images'],
                mode='markers',
                marker=dict(size=8, opacity=0.7),
                text=df['name'],
                name='Cat vs Imgs',
                hovertemplate='<b>%{text}</b><br>Categorías: %{x}<br>Imágenes: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Actualizar layout
        fig.update_layout(
            title_text="🦷 Dashboard Interactivo - Análisis de Datasets Dentales",
            showlegend=False,
            height=800
        )
        
        # Actualizar ejes
        fig.update_xaxes(title_text="Número de Imágenes", row=1, col=1)
        fig.update_yaxes(title_text="Número de Anotaciones", row=1, col=1)
        fig.update_xaxes(title_text="Nivel de Calidad", row=2, col=1)
        fig.update_yaxes(title_text="Cantidad de Datasets", row=2, col=1)
        fig.update_xaxes(title_text="Número de Categorías", row=2, col=2)
        fig.update_yaxes(title_text="Número de Imágenes", row=2, col=2)
        
        # Mostrar gráfico
        fig.show()
        
        # Guardar como HTML
        fig.write_html("dental_datasets_dashboard.html")
        print("📊 Dashboard interactivo guardado como: dental_datasets_dashboard.html")
    
    def generate_summary_table(self):
        """Genera una tabla resumen de todos los datasets."""
        dataset_info = []
        
        for dataset_type, data in self.data.items():
            if dataset_type.startswith('_') and 'datasets' in data:
                for name, info in data['datasets'].items():
                    dataset_info.append({
                        'Dataset': name[:40] + '...' if len(name) > 40 else name,
                        'Tipo': data['type'].replace(' ', '\n'),
                        'Imágenes': f"{info.get('image_count', 0):,}",
                        'Anotaciones': f"{info.get('annotation_count', 0):,}",
                        'Categorías': len(info.get('categories', [])),
                        'Calidad': info.get('dataset_quality', '').split(' ')[0],
                        'Uso Recomendado': info.get('recommended_use', '')[:50] + '...' if len(info.get('recommended_use', '')) > 50 else info.get('recommended_use', '')
                    })
        
        if not dataset_info:
            print("❌ No hay datos disponibles para la tabla")
            return
        
        df = pd.DataFrame(dataset_info)
        
        # Ordenar por número de imágenes (convertir a int para ordenar)
        df['Images_int'] = df['Imágenes'].str.replace(',', '').astype(int)
        df = df.sort_values('Images_int', ascending=False).drop('Images_int', axis=1)
        
        # Mostrar tabla
        print("\n" + "="*150)
        print("📋 TABLA RESUMEN DE DATASETS")
        print("="*150)
        print(df.to_string(index=False, max_colwidth=50))
        print("="*150)
        
        # Guardar como CSV
        df.to_csv('datasets_summary_table.csv', index=False)
        print("📊 Tabla guardada como: datasets_summary_table.csv")
    
    def run_complete_analysis(self, save_figures: bool = True):
        """Ejecuta el análisis completo y genera todas las visualizaciones."""
        print("🚀 Iniciando análisis completo de estadísticas...")
        
        # Mostrar resumen
        self.display_summary()
        
        # Generar todas las visualizaciones
        print("\n📊 Generando gráficos...")
        self.create_format_distribution_chart(save_figures)
        self.create_images_distribution_chart(save_figures)
        self.create_quality_analysis(save_figures)
        self.create_categories_wordcloud(save_figures)
        
        # Tabla resumen
        print("\n📋 Generando tabla resumen...")
        self.generate_summary_table()
        
        # Dashboard interactivo
        print("\n🎯 Creando dashboard interactivo...")
        self.create_interactive_dashboard()
        
        print("\n✅ Análisis completo terminado!")
        print("📁 Archivos generados:")
        print("   • dataset_format_distribution.png")
        print("   • dataset_images_distribution.png") 
        print("   • dataset_quality_analysis.png")
        print("   • categories_wordcloud.png")
        print("   • datasets_summary_table.csv")
        print("   • dental_datasets_dashboard.html")


def main():
    """Función principal para ejecutar el previsualizador."""
    print("🦷 PREVISUALIZADOR DE ESTADÍSTICAS DE DATASETS DENTALES")
    print("="*60)
    
    # Crear visualizador
    viewer = DentalDatasetStatisticsViewer()
    
    if not viewer.data:
        print("❌ No se pudieron cargar los datos. Ejecuta primero 'script.py'")
        return
    
    # Menú interactivo
    while True:
        print("\n🎯 ¿Qué análisis quieres ver?")
        print("1. 📊 Resumen general")
        print("2. 📈 Distribución por formato")
        print("3. 🖼️  Distribución de imágenes")
        print("4. 📉 Análisis de calidad")
        print("5. ☁️  Nube de categorías")
        print("6. 🎛️  Dashboard interactivo")
        print("7. 📋 Tabla resumen")
        print("8. 🚀 Análisis completo")
        print("9. ❌ Salir")
        
        choice = input("\n👉 Selecciona una opción (1-9): ").strip()
        
        if choice == '1':
            viewer.display_summary()
        elif choice == '2':
            viewer.create_format_distribution_chart()
        elif choice == '3':
            viewer.create_images_distribution_chart()
        elif choice == '4':
            viewer.create_quality_analysis()
        elif choice == '5':
            viewer.create_categories_wordcloud()
        elif choice == '6':
            viewer.create_interactive_dashboard()
        elif choice == '7':
            viewer.generate_summary_table()
        elif choice == '8':
            viewer.run_complete_analysis()
        elif choice == '9':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida. Intenta de nuevo.")


if __name__ == "__main__":
    main()

