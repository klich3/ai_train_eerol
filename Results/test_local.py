#!/usr/bin/env python3
"""
🧪 Test Local - Herramienta Simple para Probar Sin CUDA
Analiza resultados de modelos sin necesidad de PyTorch/CUDA
"""

import json
import os
import sys
from pathlib import Path
import random

def print_header(title):
    """Imprimir encabezado bonito."""
    print("\n" + "="*60)
    print(f"🔬 {title}")
    print("="*60)

def list_available_models():
    """Listar modelos disponibles."""
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ No se encontró directorio de modelos")
        return []
    
    models = []
    for model_file in models_dir.rglob("*.pt"):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        models.append({
            'path': model_file,
            'name': model_file.name,
            'size': f"{size_mb:.1f} MB"
        })
    
    if models:
        print("📋 Modelos disponibles:")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model['name']} ({model['size']})")
    else:
        print("❌ No se encontraron modelos (.pt)")
    
    return models

def list_available_datasets():
    """Listar datasets disponibles."""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("❌ No se encontró directorio de datasets")
        return []
    
    datasets = []
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            data_yaml = dataset_dir / "data.yaml"
            if data_yaml.exists():
                # Contar imágenes
                train_imgs = len(list((dataset_dir / "train" / "images").glob("*"))) if (dataset_dir / "train" / "images").exists() else 0
                val_imgs = len(list((dataset_dir / "val" / "images").glob("*"))) if (dataset_dir / "val" / "images").exists() else 0
                test_imgs = len(list((dataset_dir / "test" / "images").glob("*"))) if (dataset_dir / "test" / "images").exists() else 0
                
                datasets.append({
                    'path': dataset_dir,
                    'name': dataset_dir.name,
                    'train': train_imgs,
                    'val': val_imgs,
                    'test': test_imgs
                })
    
    if datasets:
        print("📊 Datasets disponibles:")
        for i, ds in enumerate(datasets, 1):
            print(f"   {i}. {ds['name']}")
            print(f"      Train: {ds['train']}, Val: {ds['val']}, Test: {ds['test']}")
    else:
        print("❌ No se encontraron datasets")
    
    return datasets

def list_test_batches():
    """Listar lotes de prueba disponibles."""
    batches_dir = Path("batches")
    if not batches_dir.exists():
        print("❌ No se encontró directorio de lotes")
        return []
    
    batches = []
    for batch_dir in batches_dir.iterdir():
        if batch_dir.is_dir():
            batch_info_file = batch_dir / "batch_info.json"
            if batch_info_file.exists():
                try:
                    with open(batch_info_file, 'r') as f:
                        info = json.load(f)
                    
                    # Contar imágenes en el lote
                    image_count = 0
                    for split_dir in batch_dir.iterdir():
                        if split_dir.is_dir():
                            images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
                            image_count += len(images)
                    
                    batches.append({
                        'path': batch_dir,
                        'name': batch_dir.name,
                        'info': info,
                        'images': image_count
                    })
                except:
                    pass
    
    if batches:
        print("📦 Lotes de prueba disponibles:")
        for i, batch in enumerate(batches, 1):
            print(f"   {i}. {batch['name']}")
            print(f"      Modelo: {batch['info'].get('model_name', 'N/A')}")
            print(f"      Dataset: {batch['info'].get('dataset_name', 'N/A')}")
            print(f"      Imágenes: {batch['images']}")
    else:
        print("❌ No se encontraron lotes de prueba")
    
    return batches

def list_results():
    """Listar resultados disponibles."""
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ No se encontró directorio de resultados")
        return []
    
    results = []
    for result_dir in results_dir.iterdir():
        if result_dir.is_dir():
            results_file = result_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    
                    # Contar imágenes predichas
                    predicted_imgs = len(list(result_dir.glob("predicted_*.jpg"))) + len(list(result_dir.glob("predicted_*.png")))
                    
                    results.append({
                        'path': result_dir,
                        'name': result_dir.name,
                        'data': data,
                        'predicted_images': predicted_imgs
                    })
                except:
                    pass
    
    if results:
        print("📊 Resultados disponibles:")
        for i, result in enumerate(results, 1):
            total_detections = sum(len(r.get('detections', [])) for r in result['data'].get('results', []))
            print(f"   {i}. {result['name']}")
            print(f"      Imágenes: {result['data'].get('total_images', 0)}")
            print(f"      Detecciones: {total_detections}")
            print(f"      Imágenes marcadas: {result['predicted_images']}")
    else:
        print("❌ No se encontraron resultados")
    
    return results

def create_simple_batch():
    """Crear un lote de prueba simple."""
    print_header("Crear Lote de Prueba")
    
    # Listar datasets
    datasets = list_available_datasets()
    if not datasets:
        return
    
    # Seleccionar dataset
    while True:
        try:
            choice = input(f"\n🎯 Selecciona dataset (1-{len(datasets)}) o 'q' para salir: ").strip()
            if choice.lower() == 'q':
                return
            
            dataset_idx = int(choice) - 1
            if 0 <= dataset_idx < len(datasets):
                selected_dataset = datasets[dataset_idx]
                break
            else:
                print("❌ Selección inválida")
        except ValueError:
            print("❌ Por favor ingresa un número")
    
    # Seleccionar split
    splits = ['val', 'test', 'train']
    available_splits = []
    for split in splits:
        split_dir = selected_dataset['path'] / split / "images"
        if split_dir.exists() and list(split_dir.glob("*")):
            available_splits.append(split)
    
    if not available_splits:
        print("❌ No se encontraron imágenes en el dataset")
        return
    
    print(f"\n📂 Splits disponibles: {', '.join(available_splits)}")
    split = input(f"🎯 Selecciona split ({available_splits[0]} por defecto): ").strip() or available_splits[0]
    
    if split not in available_splits:
        print(f"❌ Split '{split}' no disponible. Usando '{available_splits[0]}'")
        split = available_splits[0]
    
    # Número de imágenes
    max_images = int(input("🖼️ Número de imágenes a incluir (5 por defecto): ").strip() or "5")
    
    # Crear lote
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = f"{selected_dataset['name']}_{split}_{timestamp}"
    batch_dir = Path("batches") / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Copiar imágenes
    images_dir = selected_dataset['path'] / split / "images"
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    selected_images = image_files[:max_images]
    
    split_dir = batch_dir / split
    split_dir.mkdir(exist_ok=True)
    
    print(f"\n📋 Copiando {len(selected_images)} imágenes...")
    for img_file in selected_images:
        dest = split_dir / img_file.name
        import shutil
        shutil.copy2(img_file, dest)
        print(f"   ✅ {img_file.name}")
    
    # Crear info del lote
    batch_info = {
        'batch_id': batch_name,
        'dataset_name': selected_dataset['name'],
        'split': split,
        'total_images': len(selected_images),
        'created': timestamp
    }
    
    with open(batch_dir / "batch_info.json", 'w') as f:
        json.dump(batch_info, f, indent=2)
    
    print(f"\n✅ Lote creado: {batch_name}")
    print(f"📁 Ubicación: {batch_dir}")
    print(f"🖼️ Imágenes: {len(selected_images)}")

def create_demo_results_for_batch():
    """Crear resultados demo para un lote existente."""
    print_header("Crear Resultados Demo")
    
    batches = list_test_batches()
    if not batches:
        return
    
    # Seleccionar lote
    while True:
        try:
            choice = input(f"\n🎯 Selecciona lote (1-{len(batches)}) o 'q' para salir: ").strip()
            if choice.lower() == 'q':
                return
            
            batch_idx = int(choice) - 1
            if 0 <= batch_idx < len(batches):
                selected_batch = batches[batch_idx]
                break
            else:
                print("❌ Selección inválida")
        except ValueError:
            print("❌ Por favor ingresa un número")
    
    # Crear resultados demo
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_name = f"demo_{selected_batch['name']}_{timestamp}"
    results_dir = Path("results") / results_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Simular resultados
    batch_dir = selected_batch['path']
    dental_classes = ["tooth", "caries", "filling", "crown", "implant", "root"]
    
    # Buscar imágenes
    image_files = []
    for split_dir in batch_dir.glob("*/"):
        if split_dir.is_dir() and split_dir.name not in ["__pycache__"]:
            images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            image_files.extend(images)
    
    results_data = {
        'batch_info': selected_batch['info'],
        'total_images': len(image_files),
        'results': []
    }
    
    print(f"\n🎭 Simulando resultados para {len(image_files)} imágenes...")
    
    for img_file in image_files:
        num_detections = random.randint(1, 5)  # Al menos 1 detección
        detections = []
        
        for i in range(num_detections):
            detection = {
                'class_id': random.randint(0, len(dental_classes)-1),
                'class': random.choice(dental_classes),
                'confidence': round(random.uniform(0.4, 0.95), 3),
                'bbox': [
                    random.randint(50, 300),   # x1
                    random.randint(50, 300),   # y1
                    random.randint(350, 600),  # x2
                    random.randint(350, 600)   # y2
                ]
            }
            detections.append(detection)
        
        results_data['results'].append({
            'image': img_file.name,
            'detections': detections
        })
        
        print(f"   📸 {img_file.name}: {num_detections} detecciones")
    
    # Guardar resultados
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Resultados demo creados: {results_name}")
    print(f"📁 Ubicación: {results_dir}")

def analyze_results():
    """Analizar resultados disponibles."""
    print_header("Analizar Resultados")
    
    results = list_results()
    if not results:
        return
    
    # Seleccionar resultado
    while True:
        try:
            choice = input(f"\n🎯 Selecciona resultado (1-{len(results)}) o 'q' para salir: ").strip()
            if choice.lower() == 'q':
                return
            
            result_idx = int(choice) - 1
            if 0 <= result_idx < len(results):
                selected_result = results[result_idx]
                break
            else:
                print("❌ Selección inválida")
        except ValueError:
            print("❌ Por favor ingresa un número")
    
    # Mostrar análisis detallado
    data = selected_result['data']
    results_list = data.get('results', [])
    
    print(f"\n📊 ANÁLISIS DETALLADO - {selected_result['name']}")
    print("="*60)
    
    # Estadísticas generales
    total_images = len(results_list)
    total_detections = sum(len(r.get('detections', [])) for r in results_list)
    avg_detections = total_detections / total_images if total_images > 0 else 0
    
    print(f"📈 ESTADÍSTICAS GENERALES:")
    print(f"   Total imágenes: {total_images}")
    print(f"   Total detecciones: {total_detections}")
    print(f"   Promedio por imagen: {avg_detections:.2f}")
    
    # Análisis por clase
    class_counts = {}
    confidences = []
    
    for result in results_list:
        for det in result.get('detections', []):
            class_name = det.get('class', 'unknown')
            confidence = det.get('confidence', 0)
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            confidences.append(confidence)
    
    if class_counts:
        print(f"\n🏷️ DETECCIONES POR CLASE:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        print(f"\n🎯 CONFIANZA:")
        print(f"   Promedio: {avg_conf:.3f}")
        print(f"   Mínima: {min_conf:.3f}")
        print(f"   Máxima: {max_conf:.3f}")
    
    # Mostrar imágenes con más detecciones
    print(f"\n🖼️ IMÁGENES CON MÁS DETECCIONES:")
    sorted_results = sorted(results_list, key=lambda x: len(x.get('detections', [])), reverse=True)
    
    for i, result in enumerate(sorted_results[:5]):
        detections = result.get('detections', [])
        print(f"   {i+1}. {result['image']}: {len(detections)} detecciones")
        for j, det in enumerate(detections[:3]):  # Mostrar máximo 3
            print(f"      - {det.get('class', 'unknown')}: {det.get('confidence', 0):.3f}")
        if len(detections) > 3:
            print(f"      ... y {len(detections)-3} más")

def show_individual_results():
    """Mostrar resultados de imágenes individuales."""
    print_header("Ver Resultados Individuales")
    
    results = list_results()
    if not results:
        return
    
    # Seleccionar resultado
    while True:
        try:
            choice = input(f"\n🎯 Selecciona resultado (1-{len(results)}) o 'q' para salir: ").strip()
            if choice.lower() == 'q':
                return
            
            result_idx = int(choice) - 1
            if 0 <= result_idx < len(results):
                selected_result = results[result_idx]
                break
            else:
                print("❌ Selección inválida")
        except ValueError:
            print("❌ Por favor ingresa un número")
    
    # Mostrar imágenes
    data = selected_result['data']
    results_list = data.get('results', [])
    
    print(f"\n🖼️ RESULTADOS INDIVIDUALES - {selected_result['name']}")
    print("="*60)
    
    for i, result in enumerate(results_list, 1):
        detections = result.get('detections', [])
        print(f"\n📸 {i}. {result['image']}")
        
        if detections:
            print(f"   🎯 {len(detections)} detecciones encontradas:")
            for j, det in enumerate(detections):
                bbox = det.get('bbox', [0, 0, 0, 0])
                print(f"      {j+1}. {det.get('class', 'unknown')}")
                print(f"         Confianza: {det.get('confidence', 0):.3f}")
                print(f"         Posición: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        else:
            print("   ❌ No se encontraron detecciones")
        
        # Preguntar si continuar
        if i < len(results_list):
            continue_choice = input("\n⏩ Presiona Enter para continuar, 'q' para salir: ").strip()
            if continue_choice.lower() == 'q':
                break

def main_menu():
    """Menú principal."""
    while True:
        print_header("TEST LOCAL - Sin CUDA/PyTorch")
        print("🎛️ OPCIONES:")
        print("1. 📋 Ver modelos disponibles")
        print("2. 📊 Ver datasets disponibles")
        print("3. 📦 Ver lotes de prueba")
        print("4. 📊 Ver resultados disponibles")
        print("5. 🎭 Crear lote de prueba")
        print("6. 🧪 Crear resultados demo")
        print("7. 📈 Analizar resultados")
        print("8. 🔍 Ver resultados individuales")
        print("0. 🚪 Salir")
        
        choice = input("\n🎯 Selecciona una opción: ").strip()
        
        if choice == "1":
            print_header("Modelos Disponibles")
            list_available_models()
            input("\n⏩ Presiona Enter para continuar...")
            
        elif choice == "2":
            print_header("Datasets Disponibles")
            list_available_datasets()
            input("\n⏩ Presiona Enter para continuar...")
            
        elif choice == "3":
            print_header("Lotes de Prueba")
            list_test_batches()
            input("\n⏩ Presiona Enter para continuar...")
            
        elif choice == "4":
            print_header("Resultados Disponibles")
            list_results()
            input("\n⏩ Presiona Enter para continuar...")
            
        elif choice == "5":
            create_simple_batch()
            input("\n⏩ Presiona Enter para continuar...")
            
        elif choice == "6":
            create_demo_results_for_batch()
            input("\n⏩ Presiona Enter para continuar...")
            
        elif choice == "7":
            analyze_results()
            input("\n⏩ Presiona Enter para continuar...")
            
        elif choice == "8":
            show_individual_results()
            input("\n⏩ Presiona Enter para continuar...")
            
        elif choice == "0":
            print("\n👋 ¡Hasta luego!")
            break
            
        else:
            print("❌ Opción no válida")
            input("\n⏩ Presiona Enter para continuar...")

if __name__ == "__main__":
    main_menu()
