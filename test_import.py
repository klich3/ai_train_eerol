#!/usr/bin/env python3
"""
Script de prueba para verificar importaciones.
"""

import sys
import os
from pathlib import Path

# Agregar ruta de módulos
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "Src"))

print("🔍 Probando importaciones...")
print(f"Python path: {sys.path[:3]}...")
print(f"Directorio actual: {os.getcwd()}")

try:
    print("1. Probando dependencias básicas...")
    import cv2
    print("✅ OpenCV importado")
    import matplotlib.pyplot as plt
    print("✅ Matplotlib importado")
    import pandas as pd
    print("✅ Pandas importado")
    
    print("2. Importando data_analyzer...")
    from Src.data_analyzer import DataAnalyzer
    print("✅ DataAnalyzer importado correctamente")
    
    print("3. Importando data_processor...")
    from Src.data_processor import DataProcessor
    print("✅ DataProcessor importado correctamente")
    
    print("4. Importando smart_category_analyzer...")
    from Src.smart_category_analyzer import SmartCategoryAnalyzer
    print("✅ SmartCategoryAnalyzer importado correctamente")
    
    print("5. Importando smart_workflow_manager...")
    from Src.smart_workflow_manager import SmartDentalWorkflowManager
    print("✅ SmartDentalWorkflowManager importado correctamente")
    
    print("\n🎉 Todas las importaciones exitosas!")
    
    # Probar inicialización
    print("\n6. Probando inicialización...")
    manager = SmartDentalWorkflowManager()
    print("✅ SmartDentalWorkflowManager inicializado correctamente")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
