#!/usr/bin/env python3
"""
🧪 Probar generación de scripts corregida
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def test_script_generation():
    """Probar la generación de scripts mejorada."""
    print("🧪 PROBANDO GENERACIÓN DE SCRIPTS MEJORADA")
    print("=" * 50)
    
    try:
        from Src.smart_workflow_manager import SmartDentalWorkflowManager
        
        # Inicializar con tu estructura existente
        manager = SmartDentalWorkflowManager(
            base_path="_dataSets",
            output_path="Results"
        )
        
        print("✅ Manager inicializado")
        print(f"📂 Output path: {manager.output_path}")
        
        # Verificar datasets existentes
        datasets_dir = manager.output_path / "datasets"
        print(f"\n📊 Verificando datasets en: {datasets_dir}")
        
        if datasets_dir.exists():
            for item in datasets_dir.iterdir():
                if item.is_dir():
                    print(f"   📁 {item.name}")
        
        # Probar generación de scripts
        print(f"\n📝 Ejecutando generación de scripts...")
        manager._generate_training_scripts()
        
        # Verificar archivos generados
        scripts_dir = manager.output_path / "scripts"
        print(f"\n✅ Verificando archivos generados en: {scripts_dir}")
        
        if scripts_dir.exists():
            scripts = list(scripts_dir.iterdir())
            print(f"   📄 Archivos generados: {len(scripts)}")
            for script in scripts:
                print(f"      • {script.name}")
        else:
            print("❌ No se generó directorio scripts")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_script_generation()
