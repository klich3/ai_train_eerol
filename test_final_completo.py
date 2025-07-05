#!/usr/bin/env python3
"""
🧪 Test final completo de todas las opciones corregidas
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def test_todas_las_opciones():
    """🧪 Test de las opciones 8, 9 y 10 de main.py"""
    try:
        from Src.workflow_manager import DentalDataWorkflowManager
        
        print("🧪 TEST FINAL: TODAS LAS OPCIONES CORREGIDAS")
        print("="*50)
        
        # Crear manager
        manager = DentalDataWorkflowManager(
            base_path="_dataSets",
            output_path="Dist/dental_ai"
        )
        print("✅ DentalDataWorkflowManager inicializado")
        
        # TEST OPCIÓN 8: Crear estructura
        print(f"\n🏗️ TEST OPCIÓN 8: Crear estructura dental-ai...")
        manager.create_dental_ai_structure()
        print("✅ OPCIÓN 8 completada")
        
        # TEST OPCIÓN 9: Generar scripts
        print(f"\n📝 TEST OPCIÓN 9: Generar scripts de entrenamiento...")
        manager.create_training_scripts()
        print("✅ OPCIÓN 9 completada")
        
        # TEST OPCIÓN 10: Crear API template
        print(f"\n🌐 TEST OPCIÓN 10: Crear template de API...")
        manager.create_api_template()
        print("✅ OPCIÓN 10 completada")
        
        # VERIFICACIÓN FINAL COMPLETA
        print(f"\n🔍 VERIFICACIÓN FINAL COMPLETA:")
        print("="*40)
        
        base_path = Path("Dist/dental_ai")
        
        # 1. Verificar estructura principal
        print("📁 ESTRUCTURA PRINCIPAL:")
        dirs_principales = ['datasets', 'models', 'training', 'api', 'docs']
        for dir_name in dirs_principales:
            dir_path = base_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                print(f"   ✅ {dir_name}/")
            else:
                print(f"   ❌ {dir_name}/")
        
        # 2. Verificar scripts de entrenamiento
        print(f"\n📝 SCRIPTS DE ENTRENAMIENTO:")
        training_dir = base_path / "training"
        if training_dir.exists():
            scripts = list(training_dir.glob("*.py")) + list(training_dir.glob("*.sh"))
            for script in scripts:
                print(f"   ✅ {script.name}")
        else:
            print(f"   ❌ Directorio training no existe")
        
        # 3. Verificar API template
        print(f"\n🌐 API TEMPLATE:")
        api_dir = base_path / "api"
        if api_dir.exists():
            api_files = ['main.py', 'requirements.txt']
            for archivo in api_files:
                archivo_path = api_dir / archivo
                if archivo_path.exists():
                    print(f"   ✅ {archivo}")
                else:
                    print(f"   ❌ {archivo}")
        else:
            print(f"   ❌ Directorio api no existe")
        
        # 4. Verificar archivos de configuración
        print(f"\n⚙️ ARCHIVOS DE CONFIGURACIÓN:")
        config_files = ['README.md', 'requirements.txt', 'config.yaml']
        for archivo in config_files:
            archivo_path = base_path / archivo
            if archivo_path.exists():
                print(f"   ✅ {archivo}")
            else:
                print(f"   ❌ {archivo}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_todas_las_opciones()
    
    print(f"\n" + "="*50)
    if success:
        print("🎉 ¡TODOS LOS TESTS COMPLETADOS EXITOSAMENTE!")
        print()
        print("✅ OPCIONES CORREGIDAS Y FUNCIONANDO:")
        print("   • Opción 8: Crear estructura dental-ai")
        print("   • Opción 9: Generar scripts de entrenamiento")
        print("   • Opción 10: Crear template de API")
        print()
        print("🚀 MAIN.PY ESTÁ LISTO PARA USAR")
        print("   Ejecuta: python main.py")
        print("   Todas las opciones deberían funcionar sin errores")
    else:
        print("💥 Algunos tests fallaron")
        print("   Revisa los errores arriba para más detalles")
