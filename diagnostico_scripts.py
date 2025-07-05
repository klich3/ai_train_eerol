#!/usr/bin/env python3
"""
🔍 Diagnóstico del problema de generación de scripts
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def diagnosticar_problema():
    """Diagnosticar por qué no se generan archivos en la opción 7."""
    print("🔍 DIAGNÓSTICO DEL PROBLEMA DE GENERACIÓN DE SCRIPTS")
    print("=" * 60)
    
    # Verificar estructura de salida
    output_path = Path("Dist/dental_ai")
    print(f"📂 Verificando directorio de salida: {output_path}")
    print(f"   Existe: {output_path.exists()}")
    
    if output_path.exists():
        print("   Contenido:")
        for item in output_path.iterdir():
            print(f"     • {item.name}")
    
    # Verificar subdirectorios específicos
    datasets_dir = output_path / "datasets"
    scripts_dir = output_path / "scripts"
    
    print(f"\n📊 Verificando subdirectorios:")
    print(f"   datasets/: {datasets_dir.exists()}")
    print(f"   scripts/: {scripts_dir.exists()}")
    
    if datasets_dir.exists():
        print(f"   datasets/ contiene:")
        for item in datasets_dir.iterdir():
            print(f"     • {item.name}")
            
        # Verificar carpetas específicas de formato
        yolo_dir = datasets_dir / "yolo"
        coco_dir = datasets_dir / "coco"
        unet_dir = datasets_dir / "unet"
        
        print(f"\n🔄 Verificando formatos:")
        print(f"   yolo/: {yolo_dir.exists()}")
        print(f"   coco/: {coco_dir.exists()}")
        print(f"   unet/: {unet_dir.exists()}")
    
    # Simular el proceso de generación de scripts
    print(f"\n🧪 SIMULANDO PROCESO DE GENERACIÓN DE SCRIPTS:")
    
    try:
        from Src.smart_workflow_manager import SmartDentalWorkflowManager
        
        manager = SmartDentalWorkflowManager(
            base_path="_dataSets",
            output_path="Dist/dental_ai_diagnostico"
        )
        
        print("✅ Manager inicializado")
        
        # Verificar estado interno
        print(f"📊 Estado interno del manager:")
        print(f"   current_analysis: {manager.current_analysis is not None}")
        print(f"   available_categories: {len(manager.available_categories)}")
        print(f"   selected_datasets: {len(manager.selected_datasets)}")
        print(f"   conversion_results: {len(manager.conversion_results)}")
        
        # Crear directorios de prueba
        test_output = Path("Dist/dental_ai_diagnostico")
        test_scripts = test_output / "scripts"
        test_datasets = test_output / "datasets"
        test_yolo = test_datasets / "yolo"
        
        test_output.mkdir(parents=True, exist_ok=True)
        test_scripts.mkdir(exist_ok=True)
        test_datasets.mkdir(exist_ok=True)
        test_yolo.mkdir(exist_ok=True)
        
        print(f"\n🧪 Creando directorios de prueba:")
        print(f"   {test_output} ✅")
        print(f"   {test_scripts} ✅")
        print(f"   {test_datasets} ✅")
        print(f"   {test_yolo} ✅")
        
        # Probar generación de scripts
        print(f"\n📝 Probando generación de scripts...")
        manager.output_path = test_output
        manager._generate_training_scripts()
        
        # Verificar si se crearon archivos
        print(f"\n✅ Verificando archivos creados:")
        if test_scripts.exists():
            scripts_created = list(test_scripts.iterdir())
            print(f"   Scripts creados: {len(scripts_created)}")
            for script in scripts_created:
                print(f"     • {script.name}")
        else:
            print("   ❌ No se creó el directorio scripts")
            
    except Exception as e:
        print(f"❌ Error en la simulación: {e}")
        import traceback
        traceback.print_exc()
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    print("1. Antes de generar scripts (opción 7), debes:")
    print("   • Ejecutar análisis (opción 1)")
    print("   • Seleccionar datasets (opción 3)")
    print("   • Convertir formatos (opción 4)")
    print()
    print("2. O usar el workflow completo (opción 8)")
    print()
    print("3. Verificar que tienes permisos de escritura en Dist/")

def mostrar_flujo_correcto():
    """Mostrar el flujo correcto para generar scripts."""
    print(f"\n🎯 FLUJO CORRECTO PARA GENERAR SCRIPTS:")
    print("=" * 50)
    print()
    print("1️⃣ python smart_dental_workflow.py")
    print("2️⃣ Opción 1: 🔍 Escanear y analizar datasets")
    print("3️⃣ Opción 3: 📦 Seleccionar datasets")
    print("4️⃣ Opción 4: 🔄 Convertir formatos")
    print("5️⃣ Opción 7: 📝 Generar scripts de entrenamiento")
    print()
    print("O simplemente:")
    print("1️⃣ python smart_dental_workflow.py")
    print("2️⃣ Opción 8: 🚀 Workflow completo")

if __name__ == "__main__":
    diagnosticar_problema()
    mostrar_flujo_correcto()
