#!/usr/bin/env python3
"""
Test mínimo sin matplotlib.
"""

def test_minimal():
    """Test sin dependencias pesadas."""
    print("🔧 TEST MÍNIMO...")
    
    try:
        print("1. Importaciones básicas...")
        import os
        import json
        from pathlib import Path
        from collections import defaultdict, Counter
        print("✅ Básicas OK")
        
        print("2. Importando cv2...")
        import cv2
        print("✅ cv2 OK")
        
        print("3. Importando numpy...")
        import numpy as np
        print("✅ numpy OK")
        
        print("4. Importando pandas...")
        import pandas as pd
        print("✅ pandas OK")
        
        print("5. Probando crear DataAnalyzer directamente...")
        
        # Copiar la clase DataAnalyzer sin las importaciones problemáticas
        class SimpleDataAnalyzer:
            def __init__(self, base_path, unified_classes):
                self.base_path = Path(base_path)
                self.unified_classes = unified_classes
                
            def scan_datasets(self):
                print("   Escaneando...")
                return {
                    'total_datasets': 0,
                    'total_images': 0,
                    'dataset_details': {},
                    'format_distribution': Counter()
                }
        
        print("6. Creando analyzer...")
        analyzer = SimpleDataAnalyzer("_dataSets", {})
        print("✅ Analyzer creado")
        
        print("7. Ejecutando scan...")
        result = analyzer.scan_datasets()
        print("✅ Scan ejecutado")
        
        print("\n🎉 Test mínimo exitoso!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal()
