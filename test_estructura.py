#!/usr/bin/env python3
"""
Prueba de creación de estructura dental-ai
"""

import sys
from pathlib import Path

# Agregar la carpeta Src al path para importar módulos
sys.path.append(str(Path(__file__).parent / "Src"))

from Src.workflow_manager import DentalDataWorkflowManager

def test_create_structure():
    """Prueba la creación de estructura directamente."""
    
    print("🧪 PROBANDO CREACIÓN DE ESTRUCTURA DENTAL-AI")
    print("="*50)
    
    # Configurar rutas
    base_path = "_dataSets"
    output_path = "Dist/dental_ai"
    
    try:
        # Inicializar manager
        print(f"📂 Inicializando con rutas:")
        print(f"   Fuente: {base_path}")
        print(f"   Salida: {output_path}")
        
        manager = DentalDataWorkflowManager(base_path, output_path)
        
        print(f"\n✅ Manager inicializado correctamente")
        print(f"📁 Output path: {manager.output_path}")
        print(f"🏗️ Structure generator: {type(manager.structure_generator)}")
        
        # Crear estructura
        print(f"\n🚀 Creando estructura dental-ai...")
        manager.create_dental_ai_structure()
        
        print(f"\n🎉 ¡ESTRUCTURA CREADA EXITOSAMENTE!")
        
        # Verificar estructura creada
        print(f"\n📊 VERIFICANDO ESTRUCTURA CREADA:")
        print(f"="*40)
        
        if manager.output_path.exists():
            for item in manager.output_path.iterdir():
                if item.is_dir():
                    print(f"📁 {item.name}/")
                    # Mostrar subdirectorios
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            print(f"   📁 {subitem.name}/")
                        else:
                            print(f"   📄 {subitem.name}")
                else:
                    print(f"📄 {item.name}")
        else:
            print(f"❌ El directorio de salida no existe")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_create_structure()
