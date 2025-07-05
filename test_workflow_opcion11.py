#!/usr/bin/env python3
"""
🧪 Test específico para replicar exactamente el workflow de la opción 11
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))
sys.path.append(str(Path(__file__).parent / "Utils"))

def test_workflow_opcion_11():
    """🧪 Test que replica exactamente la opción 11 de main.py"""
    try:
        from Src.workflow_manager import DentalDataWorkflowManager
        from Utils.visualization import DatasetVisualizer
        
        print("🧪 TEST: REPLICAR OPCIÓN 11 COMPLETA")
        print("="*45)
        
        # Inicializar como en main.py
        manager = DentalDataWorkflowManager("_dataSets", "Dist/dental_ai")
        visualizer = DatasetVisualizer()
        
        print("✅ Managers inicializados")
        
        # Ejecutar workflow completo (como en opción 11)
        print(f"\n🚀 Ejecutando workflow completo...")
        manager.run_complete_workflow()
        
        print(f"\n📊 Creando visualizaciones adicionales...")
        
        # Crear visualizaciones adicionales (aquí puede estar el error)
        analysis = manager.get_dataset_statistics()
        print("✅ Análisis obtenido")
        
        print("📊 Creando dashboard...")
        visualizer.create_overview_dashboard(analysis, manager.output_path)
        print("✅ Dashboard creado")
        
        print("🎨 Creando word cloud...")
        visualizer.create_class_wordcloud(analysis, manager.output_path)
        print("✅ Word cloud creado")
        
        print("📋 Creando reporte detallado...")
        visualizer.create_detailed_report(analysis, manager.output_path)
        print("✅ Reporte detallado creado")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR ENCONTRADO: {e}")
        print(f"💡 Este es el mismo error que aparece en el workflow")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_workflow_opcion_11()
    if success:
        print("\n🎉 ¡WORKFLOW OPCIÓN 11 COMPLETADO SIN ERRORES!")
    else:
        print("\n💥 Error replicado - ahora sabemos dónde está el problema")
