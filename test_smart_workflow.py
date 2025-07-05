#!/usr/bin/env python3
"""
Script de prueba del Smart Dental Workflow
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

from Src.smart_workflow_manager import SmartDentalWorkflowManager

def test_analisis_rapido():
    """Probar análisis rápido con debug."""
    print("🧠 TESTING SMART DENTAL AI WORKFLOW MANAGER v3.0")
    print("="*60)
    
    try:
        # Verificar directorio
        base_path = Path("_dataSets")
        print(f"📂 Verificando directorio base: {base_path}")
        print(f"   Existe: {base_path.exists()}")
        
        if base_path.exists():
            subdirs = [d for d in base_path.iterdir() if d.is_dir()]
            print(f"   Subdirectorios: {len(subdirs)}")
            for subdir in subdirs[:5]:  # Mostrar primeros 5
                print(f"     • {subdir.name}")
        
        # Inicializar workflow
        print(f"\n🚀 Inicializando SmartDentalWorkflowManager...")
        workflow = SmartDentalWorkflowManager(
            base_path="_dataSets",
            output_path="Dist/dental_ai_test"
        )
        print(f"✅ Workflow inicializado")
        print(f"   Base path: {workflow.base_path}")
        print(f"   Output path: {workflow.output_path}")
        print(f"   Clases unificadas: {len(workflow.unified_classes)}")
        
        # Ejecutar análisis
        print(f"\n🔍 Ejecutando análisis rápido...")
        workflow._scan_and_analyze()
        
        # Mostrar resultados
        if workflow.current_analysis:
            print(f"\n📊 Resultados del análisis:")
            print(f"   • Datasets encontrados: {workflow.current_analysis.get('total_datasets', 0)}")
            print(f"   • Imágenes totales: {workflow.current_analysis.get('total_images', 0):,}")
            print(f"   • Categorías detectadas: {len(workflow.available_categories)}")
            
            # Mostrar algunas categorías
            if workflow.available_categories:
                print(f"\n🏷️ Primeras categorías detectadas:")
                for i, (category, info) in enumerate(list(workflow.available_categories.items())[:5], 1):
                    print(f"   {i}. {category}: {info['total_samples']} muestras")
        else:
            print("⚠️ No se obtuvieron resultados del análisis")
        
        print(f"\n✅ Prueba completada exitosamente")
        
    except Exception as e:
        print(f"\n❌ Error en la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analisis_rapido()
