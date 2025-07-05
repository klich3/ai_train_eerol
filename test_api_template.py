#!/usr/bin/env python3
"""
🧪 Test específico para la opción 10 - Crear template de API
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def test_create_api_template():
    """🧪 Test para crear template de API."""
    try:
        from Src.workflow_manager import DentalDataWorkflowManager
        
        print("🧪 TEST: CREAR TEMPLATE DE API")
        print("="*35)
        
        # Crear manager
        manager = DentalDataWorkflowManager(
            base_path="_dataSets",
            output_path="Dist/dental_ai"
        )
        print("✅ DentalDataWorkflowManager creado")
        
        # Test crear template de API
        print(f"\n🌐 Creando template de API...")
        manager.create_api_template()
        print("✅ Template de API creado exitosamente")
        
        # Verificar archivos creados
        print(f"\n🔍 VERIFICANDO ARCHIVOS CREADOS:")
        api_path = Path("Dist/dental_ai/api")
        
        if api_path.exists():
            print(f"✅ Directorio api/ existe")
            
            # Verificar archivos principales
            archivos_esperados = ['main.py', 'requirements.txt']
            for archivo in archivos_esperados:
                archivo_path = api_path / archivo
                if archivo_path.exists():
                    print(f"✅ {archivo} existe")
                else:
                    print(f"❌ {archivo} NO existe")
            
            # Listar todos los archivos en api/
            archivos_api = list(api_path.glob("*"))
            print(f"\n📋 Archivos en api/:")
            for archivo in archivos_api:
                if archivo.is_file():
                    print(f"   📝 {archivo.name}")
                elif archivo.is_dir():
                    print(f"   📁 {archivo.name}/")
        else:
            print(f"❌ Directorio api/ NO existe")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_create_api_template()
    if success:
        print("\n🎉 ¡TEST API TEMPLATE EXITOSO!")
        print("✅ La opción 10 de main.py debería funcionar ahora")
    else:
        print("\n💥 Error en test API template")
