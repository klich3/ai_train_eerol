#!/usr/bin/env python3
"""
🧪 Script para crear datasets y probar la opción 7
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def crear_estructura_datasets_rapida():
    """Crear estructura rápida para probar."""
    print("🚀 CREANDO ESTRUCTURA RÁPIDA PARA PROBAR OPCIÓN 7")
    print("=" * 60)
    
    # Crear estructura en Dist/dental_ai/datasets/
    base_output = Path("Dist/dental_ai/datasets")
    
    # Crear datasets básicos
    datasets_a_crear = {
        'yolo': {
            'train/images': [],
            'train/labels': [],
            'val/images': [],
            'val/labels': [],
            'data.yaml': """# Dental AI Dataset
path: .
train: train/images
val: val/images

names:
  0: caries
  1: tooth
  2: implant
  3: crown
  4: filling
"""
        },
        'coco': {
            'train': [],
            'val': [],
            'annotations': [],
            'annotations/instances_train.json': '{"images": [], "annotations": [], "categories": []}',
            'annotations/instances_val.json': '{"images": [], "annotations": [], "categories": []}'
        },
        'unet': {
            'train/images': [],
            'train/masks': [],
            'val/images': [],
            'val/masks': []
        },
        'classification': {
            'train/caries': [],
            'train/tooth_healthy': [],
            'train/implant': [],
            'val/caries': [],
            'val/tooth_healthy': [],
            'val/implant': []
        }
    }
    
    datasets_creados = []
    
    for dataset_name, estructura in datasets_a_crear.items():
        dataset_path = base_output / dataset_name
        print(f"📁 Creando dataset: {dataset_name}")
        
        for ruta, contenido in estructura.items():
            full_path = dataset_path / ruta
            
            if isinstance(contenido, list):  # Es un directorio
                full_path.mkdir(parents=True, exist_ok=True)
            else:  # Es un archivo
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(contenido)
        
        datasets_creados.append(dataset_name)
        print(f"   ✅ {dataset_name} creado")
    
    return datasets_creados

def probar_generacion_scripts():
    """Probar la generación de scripts."""
    print("\n🧪 PROBANDO GENERACIÓN DE SCRIPTS...")
    
    try:
        from Src.smart_workflow_manager import SmartDentalWorkflowManager
        
        # Inicializar manager
        manager = SmartDentalWorkflowManager(
            base_path="_dataSets",
            output_path="Dist/dental_ai"
        )
        
        print("✅ Manager inicializado")
        
        # Probar generación de scripts
        print("📝 Ejecutando _generate_training_scripts()...")
        manager._generate_training_scripts()
        
        # Verificar si se crearon archivos
        scripts_dir = Path("Dist/dental_ai/scripts")
        if scripts_dir.exists():
            scripts = list(scripts_dir.iterdir())
            print(f"✅ Scripts generados: {len(scripts)}")
            for script in scripts:
                print(f"   📄 {script.name}")
        else:
            print("❌ No se creó el directorio scripts")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal."""
    # Crear estructura de datasets
    datasets_creados = crear_estructura_datasets_rapida()
    
    if datasets_creados:
        print(f"\n✅ Datasets creados: {', '.join(datasets_creados)}")
        
        # Probar generación de scripts
        exito = probar_generacion_scripts()
        
        if exito:
            print("\n🎉 ¡ÉXITO! Todo funcionando correctamente")
            print("\n🎯 PRÓXIMOS PASOS:")
            print("1. 📝 Ejecuta: python smart_dental_workflow.py")
            print("2. 🎯 Selecciona opción 7: Generar scripts de entrenamiento")
            print("3. 🚀 Los scripts estarán en: Dist/dental_ai/scripts/")
        else:
            print("\n❌ Hubo errores. Revisa los logs arriba.")
    else:
        print("❌ No se pudieron crear los datasets")

if __name__ == "__main__":
    main()
