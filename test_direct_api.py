#!/usr/bin/env python3
"""
🧪 Test directo del ScriptTemplateGenerator para API
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def test_direct_api_creation():
    """🧪 Test directo de creación de API."""
    try:
        from Src.script_templates import ScriptTemplateGenerator
        
        print("🧪 TEST DIRECTO: API TEMPLATE GENERATOR")
        print("="*45)
        
        output_path = Path("Results")
        generator = ScriptTemplateGenerator(output_path)
        print("✅ ScriptTemplateGenerator creado")
        
        # Crear API template
        print(f"\n🌐 Creando API template...")
        generator.create_api_template()
        print("✅ API template creado")
        
        # Verificar archivos
        api_path = output_path / "api"
        print(f"\n🔍 Verificando archivos en {api_path}:")
        
        if api_path.exists():
            for item in api_path.iterdir():
                if item.is_file():
                    size = item.stat().st_size
                    print(f"   📝 {item.name} ({size} bytes)")
                elif item.is_dir():
                    print(f"   📁 {item.name}/")
        else:
            print(f"❌ Directorio {api_path} no existe")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_api_creation()
    if success:
        print("\n🎉 ¡API TEMPLATE CREADO EXITOSAMENTE!")
    else:
        print("\n💥 Error en creación de API template")
