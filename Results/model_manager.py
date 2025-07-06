#!/usr/bin/env python3
"""
🔧 Model Manager
Utilidad para gestionar modelos entrenados
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime
import argparse

class ModelManager:
    """Gestor de modelos dentales."""
    
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
    def scan_for_models(self):
        """Escanear en busca de modelos entrenados."""
        print("🔍 Escaneando modelos entrenados...")
        
        # Ubicaciones posibles
        search_paths = [
            self.base_dir / "datasets" / "**" / "logs" / "**" / "weights",
            self.base_dir / "training" / "logs" / "**" / "weights",
            self.base_dir / "models" / "**",
            Path.home() / "runs" / "detect" / "**" / "weights",  # YOLO default
        ]
        
        models_found = {}
        
        for search_path in search_paths:
            if search_path.parent.exists():
                pt_files = list(search_path.parent.rglob("*.pt"))
                
                for pt_file in pt_files:
                    if any(name in pt_file.name for name in ["best", "last", "final"]):
                        # Obtener información del modelo
                        model_info = self._get_model_info(pt_file)
                        models_found[str(pt_file)] = model_info
        
        return models_found
    
    def _get_model_info(self, model_path):
        """Obtener información de un modelo."""
        model_path = Path(model_path)
        
        info = {
            'name': model_path.name,
            'path': str(model_path),
            'size_mb': model_path.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
            'type': 'YOLO' if model_path.suffix == '.pt' else 'Unknown'
        }
        
        # Intentar obtener más información
        try:
            # Buscar archivos relacionados
            weights_dir = model_path.parent
            results_dir = weights_dir.parent
            
            # Buscar results.csv o results.png
            if (results_dir / "results.csv").exists():
                info['has_metrics'] = True
            if (results_dir / "results.png").exists():
                info['has_plots'] = True
                
            # Buscar data.yaml asociado
            dataset_dirs = [
                results_dir.parent.parent,  # Subir dos niveles
                results_dir.parent.parent.parent,  # Subir tres niveles
            ]
            
            for dataset_dir in dataset_dirs:
                if (dataset_dir / "data.yaml").exists():
                    info['dataset'] = str(dataset_dir)
                    break
                    
        except Exception:
            pass
        
        return info
    
    def list_models(self):
        """Listar todos los modelos encontrados."""
        models = self.scan_for_models()
        
        if not models:
            print("❌ No se encontraron modelos entrenados")
            return []
        
        print(f"📋 Encontrados {len(models)} modelos:")
        print("-" * 80)
        
        for i, (path, info) in enumerate(models.items()):
            print(f"{i+1:2d}. {info['name']}")
            print(f"    📁 Ruta: {path}")
            print(f"    📊 Tamaño: {info['size_mb']:.1f} MB")
            print(f"    📅 Modificado: {info['modified'][:19]}")
            if 'dataset' in info:
                print(f"    🗂️ Dataset: {info['dataset']}")
            if info.get('has_metrics'):
                print(f"    📈 Métricas: ✅")
            print()
        
        return list(models.items())
    
    def organize_model(self, model_path, name=None):
        """Organizar un modelo en la estructura estándar."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        # Generar nombre si no se proporciona
        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"dental_model_{timestamp}"
        
        # Crear directorio de destino
        dest_dir = self.models_dir / "yolo_detect" / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copiar modelo
        dest_model = dest_dir / "best.pt"
        shutil.copy2(model_path, dest_model)
        
        # Copiar archivos relacionados
        source_dir = model_path.parent
        related_files = ["last.pt", "results.csv", "results.png", "confusion_matrix.png"]
        
        for file_name in related_files:
            source_file = source_dir / file_name
            if source_file.exists():
                dest_file = dest_dir / file_name
                shutil.copy2(source_file, dest_file)
        
        # Crear metadata
        metadata = {
            'name': name,
            'created': datetime.now().isoformat(),
            'source_path': str(model_path),
            'model_size_mb': model_path.stat().st_size / (1024 * 1024),
            'type': 'YOLO',
            'organized_by': 'ModelManager'
        }
        
        metadata_file = dest_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Crear README
        readme_content = f"""# {name}

## Información del Modelo
- **Tipo**: YOLO Detection
- **Creado**: {metadata['created'][:19]}
- **Tamaño**: {metadata['model_size_mb']:.1f} MB
- **Origen**: {metadata['source_path']}

## Archivos
- `best.pt` - Mejor modelo entrenado
- `last.pt` - Último checkpoint (si disponible)
- `results.csv` - Métricas de entrenamiento (si disponible)
- `results.png` - Gráficos de entrenamiento (si disponible)
- `metadata.json` - Metadatos del modelo

## Uso
```python
from ultralytics import YOLO

# Cargar modelo
model = YOLO('{dest_model}')

# Hacer predicción
results = model('imagen.jpg')
```

## Pruebas
```bash
# Prueba rápida
python ../../quick_test.py

# Prueba completa
python ../../test_model.py --model {dest_model}

# Demo visual
python ../../visual_demo.py
```
"""
        
        readme_file = dest_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Modelo organizado en: {dest_dir}")
        print(f"📝 Archivos copiados:")
        for file in dest_dir.iterdir():
            if file.is_file():
                print(f"   - {file.name}")
        
        return dest_dir
    
    def create_model_index(self):
        """Crear índice de todos los modelos organizados."""
        organized_models = []
        
        yolo_detect_dir = self.models_dir / "yolo_detect"
        if yolo_detect_dir.exists():
            for model_dir in yolo_detect_dir.iterdir():
                if model_dir.is_dir():
                    metadata_file = model_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        metadata['path'] = str(model_dir)
                        organized_models.append(metadata)
        
        # Crear índice
        index = {
            'generated': datetime.now().isoformat(),
            'total_models': len(organized_models),
            'models': organized_models
        }
        
        index_file = self.models_dir / "index.json"
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        print(f"📋 Índice de modelos creado: {index_file}")
        print(f"📊 Total modelos organizados: {len(organized_models)}")
        
        return index

def main():
    parser = argparse.ArgumentParser(description="Gestor de modelos dentales")
    parser.add_argument("--scan", action="store_true", help="Escanear modelos")
    parser.add_argument("--list", action="store_true", help="Listar modelos")
    parser.add_argument("--organize", help="Organizar modelo (ruta)")
    parser.add_argument("--name", help="Nombre para el modelo organizado")
    parser.add_argument("--index", action="store_true", help="Crear índice de modelos")
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    if args.scan or args.list:
        models = manager.list_models()
        
    elif args.organize:
        if not Path(args.organize).exists():
            print(f"❌ Error: Modelo no encontrado: {args.organize}")
            return
        
        dest_dir = manager.organize_model(args.organize, args.name)
        manager.create_model_index()
        
    elif args.index:
        manager.create_model_index()
        
    else:
        # Ejecutar interfaz interactiva
        print("🔧 MODEL MANAGER")
        print("===============")
        
        while True:
            print("\n📋 Opciones:")
            print("1. 🔍 Escanear y listar modelos")
            print("2. 📦 Organizar modelo")
            print("3. 📋 Crear índice")
            print("0. ❌ Salir")
            
            choice = input("\n🎯 Selecciona una opción: ").strip()
            
            if choice == "1":
                models = manager.list_models()
                
            elif choice == "2":
                models = manager.list_models()
                if models:
                    try:
                        idx = int(input("📝 Número de modelo a organizar: ")) - 1
                        if 0 <= idx < len(models):
                            model_path = models[idx][0]
                            name = input("📝 Nombre para el modelo (opcional): ").strip() or None
                            dest_dir = manager.organize_model(model_path, name)
                        else:
                            print("❌ Número inválido")
                    except ValueError:
                        print("❌ Entrada inválida")
                        
            elif choice == "3":
                manager.create_model_index()
                
            elif choice == "0":
                print("👋 ¡Hasta luego!")
                break
                
            else:
                print("❌ Opción inválida")

if __name__ == "__main__":
    main()
