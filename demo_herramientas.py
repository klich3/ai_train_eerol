#!/usr/bin/env python3
"""
🦷 Demo de Herramientas Integradas
Demuestra el uso de las herramientas de análisis y estadísticas integradas
"""

import sys
from pathlib import Path
import json

# Agregar rutas
sys.path.append(str(Path(__file__).parent / "Utils"))

def demo_analisis_completo():
    """Demo del análisis completo integrado."""
    
    print("🦷 DEMO: ANÁLISIS COMPLETO DE DATASETS DENTALES")
    print("=" * 60)
    
    try:
        from advanced_analysis import analyze_dental_datasets
        
        datasets_path = "_dataSets"
        statistics_path = "StatisticsResults"
        
        print(f"📁 Analizando datasets en: {datasets_path}")
        print(f"📊 Resultados se guardarán en: {statistics_path}")
        print()
        print("🔄 Ejecutando análisis... (esto puede tomar unos minutos)")
        
        # Ejecutar análisis completo
        results = analyze_dental_datasets(
            datasets_path=datasets_path,
            output_dir=statistics_path,
            generate_visuals=True
        )
        
        # Mostrar resumen
        if 'summary' in results:
            summary = results['summary']
            
            print("\n📈 RESULTADOS DEL ANÁLISIS:")
            print("=" * 40)
            print(f"📁 Tipos de datasets: {summary.get('total_dataset_types', 0)}")
            print(f"📊 Datasets totales: {summary.get('total_datasets', 0)}")
            print(f"🖼️  Imágenes totales: {summary.get('total_images', 0):,}")
            
            # Distribución por formato
            if 'format_distribution' in summary:
                print(f"\n📋 DISTRIBUCIÓN POR FORMATO:")
                for format_type, count in summary['format_distribution'].items():
                    print(f"   • {format_type}: {count} datasets")
            
            # Top categorías
            if 'top_categories' in summary:
                print(f"\n🏷️  TOP 5 CATEGORÍAS:")
                top_cats = dict(sorted(summary['top_categories'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5])
                for i, (cat, count) in enumerate(top_cats.items(), 1):
                    print(f"   {i}. {cat}: {count} datasets")
            
            # Calidad de imágenes
            if 'quality_overview' in summary:
                print(f"\n🎯 RESUMEN DE CALIDAD:")
                quality_data = summary['quality_overview']
                total_quality = sum(quality_data.values())
                if total_quality > 0:
                    for quality, count in quality_data.items():
                        percentage = (count / total_quality) * 100
                        print(f"   • {quality.capitalize()}: {count:,} imágenes ({percentage:.1f}%)")
            
            print(f"\n📁 ARCHIVOS GENERADOS EN {statistics_path}/:")
            statistics_dir = Path(statistics_path)
            if statistics_dir.exists():
                files = list(statistics_dir.iterdir())
                for file_path in sorted(files):
                    if file_path.is_file():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        print(f"   📄 {file_path.name} ({size_mb:.1f} MB)")
            
            print(f"\n🌐 DASHBOARD INTERACTIVO:")
            dashboard_path = statistics_dir / "dental_datasets_dashboard.html"
            if dashboard_path.exists():
                print(f"   ✅ Disponible en: {dashboard_path}")
                print(f"   🔗 Abre en tu navegador para explorar los datos")
            else:
                print(f"   ⚠️  No se pudo generar el dashboard")
            
            print(f"\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
            
        else:
            print("⚠️  No se pudo obtener el resumen del análisis")
            
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("   Asegúrate de que las dependencias estén instaladas:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")

def demo_estadisticas_visuales():
    """Demo de generación de estadísticas visuales."""
    
    print("🦷 DEMO: GENERACIÓN DE ESTADÍSTICAS VISUALES")
    print("=" * 60)
    
    try:
        from advanced_analysis import DentalDatasetStatisticsViewer
        
        statistics_path = "StatisticsResults"
        
        # Verificar si hay datos de análisis
        analysis_file = Path(statistics_path) / "dental_dataset_analysis.json"
        
        if not analysis_file.exists():
            print("❌ No se encontraron datos de análisis.")
            print("   Ejecuta primero: python demo_herramientas.py --analisis")
            return
        
        print(f"📊 Generando visualizaciones desde: {analysis_file}")
        
        # Crear visualizador
        viewer = DentalDatasetStatisticsViewer(
            json_file=str(analysis_file),
            output_dir=statistics_path
        )
        
        # Generar todos los reportes
        print("🎨 Generando reportes y visualizaciones...")
        viewer.generate_all_reports()
        
        print("\n✅ VISUALIZACIONES GENERADAS:")
        visualizations = [
            "dataset_overview.png",
            "format_distribution.png", 
            "categories_analysis.png",
            "quality_analysis.png",
            "size_distribution.png",
            "datasets_summary_table.csv",
            "dataset_report.md",
            "dental_datasets_dashboard.html"
        ]
        
        statistics_dir = Path(statistics_path)
        for viz_file in visualizations:
            viz_path = statistics_dir / viz_file
            if viz_path.exists():
                print(f"   ✅ {viz_file}")
            else:
                print(f"   ⚠️  {viz_file} (no generado)")
        
        print(f"\n📁 Todos los archivos están en: {statistics_path}/")
        
    except Exception as e:
        print(f"❌ Error generando visualizaciones: {e}")

def demo_resumen_rapido():
    """Demo de resumen rápido de resultados existentes."""
    
    print("🦷 DEMO: RESUMEN RÁPIDO")
    print("=" * 30)
    
    statistics_path = Path("StatisticsResults")
    analysis_file = statistics_path / "dental_dataset_analysis.json"
    
    if not analysis_file.exists():
        print("❌ No hay datos de análisis disponibles.")
        print("   Ejecuta: python demo_herramientas.py --analisis")
        return
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'summary' in data:
            summary = data['summary']
            timestamp = data.get('analysis_timestamp', 'Desconocido')
            
            print(f"📅 Último análisis: {timestamp}")
            print(f"📁 Datasets: {summary.get('total_datasets', 0)}")
            print(f"🖼️  Imágenes: {summary.get('total_images', 0):,}")
            print(f"🏷️  Categorías: {len(summary.get('top_categories', {}))}")
            
            # Archivos disponibles
            print(f"\n📄 ARCHIVOS DISPONIBLES:")
            if statistics_path.exists():
                files = list(statistics_path.iterdir())
                for file_path in sorted(files):
                    if file_path.is_file():
                        print(f"   📄 {file_path.name}")
            
        else:
            print("⚠️  Datos de resumen no disponibles")
            
    except Exception as e:
        print(f"❌ Error leyendo datos: {e}")

def main():
    """Función principal del demo."""
    
    if len(sys.argv) > 1:
        option = sys.argv[1]
        
        if option == "--analisis":
            demo_analisis_completo()
        elif option == "--visuales":
            demo_estadisticas_visuales()
        elif option == "--resumen":
            demo_resumen_rapido()
        else:
            print("❌ Opción no válida")
            show_help()
    else:
        show_help()

def show_help():
    """Muestra la ayuda del demo."""
    
    print("🦷 DEMO DE HERRAMIENTAS INTEGRADAS")
    print("=" * 40)
    print()
    print("Opciones disponibles:")
    print("  --analisis    Ejecutar análisis completo de datasets")
    print("  --visuales    Generar visualizaciones y reportes")
    print("  --resumen     Mostrar resumen rápido")
    print()
    print("Ejemplos de uso:")
    print("  python demo_herramientas.py --analisis")
    print("  python demo_herramientas.py --visuales")
    print("  python demo_herramientas.py --resumen")
    print()
    print("🔄 Flujo recomendado:")
    print("  1. python demo_herramientas.py --analisis")
    print("  2. python demo_herramientas.py --visuales")
    print("  3. Explorar StatisticsResults/")
    print()
    print("📚 Documentación completa en: Wiki/")

if __name__ == "__main__":
    main()
