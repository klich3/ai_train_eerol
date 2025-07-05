#!/usr/bin/env python3
"""
🔧 Generador dinámico de data.yaml para YOLO
===========================================

Genera un data.yaml con rutas absolutas dinámicas que funciona
en cualquier sistema (macOS, Linux, Windows).
"""

import os
import yaml
from pathlib import Path

def generate_dynamic_data_yaml():
    """Genera data.yaml con rutas absolutas dinámicas."""
    
    # Detectar el directorio base del proyecto
    script_dir = Path(__file__).parent
    
    # Buscar dental_ai en la estructura actual
    dental_ai_path = script_dir / "Dist" / "dental_ai"
    
    if not dental_ai_path.exists():
        # Buscar hacia arriba hasta encontrar dental_ai
        current_dir = script_dir
        while current_dir.parent != current_dir:
            potential_path = current_dir / "Dist" / "dental_ai"
            if potential_path.exists():
                dental_ai_path = potential_path
                break
            current_dir = current_dir.parent
    
    if not dental_ai_path.exists():
        # Si no encontramos dental_ai, usar variables de entorno
        if os.getenv("HOME"):
            # Linux/macOS
            dental_ai_path = Path(os.getenv("HOME")) / "ai" / "XRAY" / "Dist" / "dental_ai"
        elif os.getenv("USERPROFILE"):
            # Windows
            dental_ai_path = Path(os.getenv("USERPROFILE")) / "ai" / "XRAY" / "Dist" / "dental_ai"
        else:
            # Fallback - usar directorio actual
            dental_ai_path = script_dir / "Dist" / "dental_ai"
    
    # Ruta del dataset
    dataset_path = dental_ai_path / "datasets" / "detection_combined"
    
    print(f"📁 Directorio dental_ai: {dental_ai_path}")
    print(f"📁 Dataset path: {dataset_path}")
    
    # Verificar que existe
    if not dataset_path.exists():
        print(f"❌ Error: No existe {dataset_path}")
        return None
    
    # Leer el data.yaml original
    original_yaml = dataset_path / "data.yaml"
    if not original_yaml.exists():
        print(f"❌ Error: No existe {original_yaml}")
        return None
    
    with open(original_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Actualizar con ruta absoluta
    data['path'] = str(dataset_path.absolute())
    
    # Crear data.yaml temporal con rutas absolutas
    temp_yaml = dataset_path / "data_absolute.yaml"
    
    with open(temp_yaml, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✅ Generado: {temp_yaml}")
    print(f"📊 Ruta absoluta: {data['path']}")
    
    return str(temp_yaml)

if __name__ == "__main__":
    result = generate_dynamic_data_yaml()
    if result:
        print(f"📝 Archivo generado: {result}")
    else:
        print("❌ Error generando data.yaml")
