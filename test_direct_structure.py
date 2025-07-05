#!/usr/bin/env python3
"""
🧪 Test directo del StructureGenerator
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def test_structure_generator():
    """🧪 Prueba directa del StructureGenerator."""
    try:
        from Src.structure_generator import StructureGenerator
        
        print("🧪 TEST: STRUCTURE GENERATOR DIRECTO")
        print("="*40)
        
        output_path = Path("Dist/dental_ai")
        print(f"📂 Output path: {output_path}")
        
        # Crear StructureGenerator
        generator = StructureGenerator(output_path)
        print("✅ StructureGenerator creado")
        
        # Mostrar estructura que se va a crear
        print(f"\n📋 ESTRUCTURA A CREAR:")
        for main_dir, content in generator.dental_ai_structure.items():
            print(f"📁 {main_dir}/")
            if isinstance(content, dict):
                for sub_dir, description in content.items():
                    if '.' not in sub_dir:  # Solo directorios, no archivos
                        print(f"   📁 {sub_dir}/ - {description}")
        
        # Crear estructura
        print(f"\n🏗️ Creando estructura...")
        result = generator.create_dental_ai_structure()
        
        print(f"✅ Estructura creada exitosamente!")
        print(f"📊 Directorios creados: {len(result) if result else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_structure_generator()
    if success:
        print("\n🎉 Test exitoso!")
    else:
        print("\n💥 Test falló!")
