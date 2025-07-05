#!/usr/bin/env python3
"""
Script de prueba para verificar que DataWorkflowManager funciona sin errores.
"""

import sys
from pathlib import Path

try:
    from DataWorkflowManager import DentalDataWorkflowManager
    
    print("✅ Importación exitosa de DentalDataWorkflowManager")
    
    # Crear instancia con rutas por defecto
    manager = DentalDataWorkflowManager(
        base_path="_dataSets",
        output_path="dental-ai"
    )
    
    print("✅ Instancia creada exitosamente")
    
    # Verificar que los métodos existen
    methods_to_check = [
        'load_analysis_data',
        'analyze_class_distribution', 
        'recommend_dataset_fusion',
        'generate_training_recommendations',
        'create_api_template'
    ]
    
    for method_name in methods_to_check:
        if hasattr(manager, method_name):
            print(f"✅ Método {method_name} existe")
        else:
            print(f"❌ Método {method_name} NO existe")
    
    print("\n🧪 Pruebas básicas completadas")
    print("📋 Para prueba completa, ejecuta: python DataWorkflowManager.py")
    
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("   Verifica que el archivo DataWorkflowManager.py existe y es válido")
except Exception as e:
    print(f"❌ Error general: {e}")
    import traceback
    traceback.print_exc()
