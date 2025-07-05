#!/usr/bin/env python3
"""
Test de importaciones individuales.
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def test_importaciones():
    """Test de cada importación por separado."""
    print("🔧 PROBANDO IMPORTACIONES INDIVIDUALES...")
    
    try:
        print("1. Importando DataAnalyzer...")
        from Src.data_analyzer import DataAnalyzer
        print("✅ DataAnalyzer OK")
        
        print("2. Importando DataProcessor...")
        from Src.data_processor import DataProcessor
        print("✅ DataProcessor OK")
        
        print("3. Importando StructureGenerator...")
        from Src.structure_generator import StructureGenerator
        print("✅ StructureGenerator OK")
        
        print("4. Importando ScriptTemplateGenerator...")
        from Src.script_templates import ScriptTemplateGenerator
        print("✅ ScriptTemplateGenerator OK")
        
        print("5. Importando SmartCategoryAnalyzer...")
        from Src.smart_category_analyzer import SmartCategoryAnalyzer
        print("✅ SmartCategoryAnalyzer OK")
        
        print("6. Importando Utils...")
        from Utils.data_augmentation import DataBalancer, QualityChecker
        print("✅ Utils.data_augmentation OK")
        
        from Utils.visualization import DatasetVisualizer
        print("✅ Utils.visualization OK")
        
        print("7. Finalmente SmartDentalWorkflowManager...")
        from Src.smart_workflow_manager import SmartDentalWorkflowManager
        print("✅ SmartDentalWorkflowManager OK")
        
        print("\n🎉 Todas las importaciones exitosas!")
        
    except Exception as e:
        print(f"❌ Error en importación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_importaciones()
