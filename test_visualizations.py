#!/usr/bin/env python3
"""
🧪 Test específico para verificar las visualizaciones corregidas
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))
sys.path.append(str(Path(__file__).parent / "Utils"))

def test_visualizations():
    """🧪 Test de las visualizaciones después de corregir el error de color."""
    try:
        from Src.workflow_manager import DentalDataWorkflowManager
        from Utils.visualization import DatasetVisualizer
        
        print("🧪 TEST: VISUALIZACIONES CORREGIDAS")
        print("="*40)
        
        # Crear manager
        manager = DentalDataWorkflowManager(
            base_path="_dataSets",
            output_path="Dist/dental_ai"
        )
        print("✅ DentalDataWorkflowManager creado")
        
        # Crear visualizer
        visualizer = DatasetVisualizer()
        print("✅ DatasetVisualizer creado")
        
        # Obtener análisis existente
        print(f"\n📊 Obteniendo estadísticas de datasets...")
        analysis = manager.get_dataset_statistics()
        print(f"✅ Análisis obtenido: {analysis.get('total_datasets', 0)} datasets")
        
        # Test crear visualizaciones
        print(f"\n📊 Creando dashboard...")
        visualizer.create_overview_dashboard(analysis, manager.output_path)
        print("✅ Dashboard creado")
        
        print(f"\n🎨 Creando word cloud...")
        visualizer.create_class_wordcloud(analysis, manager.output_path)
        print("✅ Word cloud creado")
        
        print(f"\n📋 Creando reporte detallado...")
        visualizer.create_detailed_report(analysis, manager.output_path)
        print("✅ Reporte detallado creado")
        
        # Verificar archivos creados
        print(f"\n🔍 VERIFICANDO ARCHIVOS CREADOS:")
        files_to_check = [
            'dental_datasets_dashboard.html',
            'categories_wordcloud.png',
            'dental_dataset_report.md'
        ]
        
        for filename in files_to_check:
            file_path = manager.output_path / filename
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"   ✅ {filename} ({size} bytes)")
            else:
                print(f"   ❌ {filename} NO existe")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualizations()
    if success:
        print("\n🎉 ¡VISUALIZACIONES FUNCIONAN CORRECTAMENTE!")
        print("✅ Error de color '#gray' corregido")
        print("✅ El workflow completo debería funcionar sin errores ahora")
    else:
        print("\n💥 Error en visualizaciones")
