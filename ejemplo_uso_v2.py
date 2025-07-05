#!/usr/bin/env python3
"""
🦷 Ejemplo de uso del Dental AI Workflow Manager v2.0
======================================================

Este script demuestra cómo usar el sistema modularizado para:
- Analizar datasets dentales
- Crear estructura dental-ai
- Procesar y fusionar datasets
- Generar scripts de entrenamiento

Author: Anton Sychev
Version: 2.0 (Modular)
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

from Src.workflow_manager import DentalDataWorkflowManager
from Utils.visualization import DatasetVisualizer
from Utils.data_augmentation import DataBalancer, QualityChecker


def ejemplo_basico():
    """📝 Ejemplo básico de uso del sistema."""
    print("🦷 EJEMPLO BÁSICO - DENTAL AI WORKFLOW MANAGER v2.0")
    print("="*60)
    
    # 1. Inicializar el manager
    print("\n1️⃣ Inicializando Workflow Manager...")
    manager = DentalDataWorkflowManager(
        base_path="_dataSets",  # Directorio con datasets originales
        output_path="Dist/dental_ai"  # Salida en Dist/dental_ai
    )
    
    # 2. Crear estructura básica
    print("\n2️⃣ Creando estructura dental-ai...")
    manager.create_dental_ai_structure()
    
    # 3. Analizar datasets disponibles
    print("\n3️⃣ Analizando datasets disponibles...")
    analysis = manager.scan_and_analyze_datasets()
    
    print(f"✅ Análisis completado:")
    print(f"   📊 Datasets encontrados: {analysis['total_datasets']}")
    print(f"   🖼️ Imágenes totales: {analysis['total_images']:,}")
    print(f"   📋 Formatos detectados: {list(analysis['format_distribution'].keys())}")
    
    # 4. Crear visualizaciones
    print("\n4️⃣ Creando visualizaciones...")
    visualizer = DatasetVisualizer()
    visualizer.create_overview_dashboard(analysis, manager.output_path)
    visualizer.create_class_wordcloud(analysis, manager.output_path)
    
    # 5. Generar scripts de entrenamiento
    print("\n5️⃣ Generando scripts de entrenamiento...")
    manager.create_training_scripts()
    
    # 6. Crear template de API
    print("\n6️⃣ Creando template de API...")
    manager.create_api_template()
    
    print(f"\n🎉 ¡Ejemplo básico completado!")
    print(f"📂 Revisa los resultados en: {manager.output_path}")


def ejemplo_procesamiento_avanzado():
    """🔄 Ejemplo de procesamiento avanzado de datasets."""
    print("\n\n🔄 EJEMPLO AVANZADO - PROCESAMIENTO DE DATASETS")
    print("="*55)
    
    # Inicializar manager
    manager = DentalDataWorkflowManager(
        base_path="_dataSets",
        output_path="Dist/dental_ai"
    )
    
    # 1. Analizar calidad de datasets
    print("\n1️⃣ Analizando calidad de datasets...")
    analysis = manager.scan_and_analyze_datasets()
    
    quality_checker = QualityChecker()
    
    high_quality_datasets = []
    for dataset_path, info in analysis['dataset_details'].items():
        if info['quality_score'] >= 70:  # Solo datasets de buena calidad
            high_quality_datasets.append((dataset_path, info))
    
    print(f"🏆 Datasets de alta calidad encontrados: {len(high_quality_datasets)}")
    
    # 2. Procesar datasets por formato
    print("\n2️⃣ Procesando datasets por formato...")
    
    # Fusionar YOLO datasets
    yolo_datasets = [path for path, info in high_quality_datasets if info['format'] == 'YOLO']
    if yolo_datasets:
        print(f"   🔄 Fusionando {len(yolo_datasets)} datasets YOLO...")
        yolo_stats = manager.merge_yolo_datasets(yolo_datasets)
        print(f"   ✅ YOLO: {yolo_stats.get('total_images', 0)} imágenes procesadas")
    
    # Fusionar COCO datasets
    coco_datasets = [path for path, info in high_quality_datasets if info['format'] == 'COCO']
    if coco_datasets:
        print(f"   🔄 Fusionando {len(coco_datasets)} datasets COCO...")
        coco_stats = manager.merge_coco_datasets(coco_datasets)
        print(f"   ✅ COCO: {coco_stats.get('total_images', 0)} imágenes procesadas")
    
    # Crear dataset de clasificación
    classification_datasets = [path for path, info in high_quality_datasets 
                             if info['format'] == 'Classification']
    if classification_datasets:
        print(f"   📁 Creando dataset de clasificación...")
        class_stats = manager.create_classification_dataset(classification_datasets)
        print(f"   ✅ Clasificación: {class_stats.get('total_images', 0)} imágenes procesadas")
    
    # 3. Balancear datasets (augmentación)
    print("\n3️⃣ Balanceando datasets con augmentación...")
    balancer = DataBalancer(target_samples_per_class=500)
    
    # Balancear dataset YOLO si existe
    yolo_path = manager.output_path / "datasets" / "detection_combined"
    if yolo_path.exists():
        balance_stats = balancer.balance_yolo_dataset(yolo_path)
        print(f"   ⚖️ Muestras augmentadas: {balance_stats.get('total_augmented', 0)}")
    
    print(f"\n🎉 ¡Procesamiento avanzado completado!")


def ejemplo_personalizado():
    """⚙️ Ejemplo de configuración personalizada."""
    print("\n\n⚙️ EJEMPLO PERSONALIZADO - CONFIGURACIÓN AVANZADA")
    print("="*55)
    
    # Crear manager con configuración personalizada
    manager = DentalDataWorkflowManager(
        base_path="_dataSets",
        output_path="Dist/dental_ai_custom"
    )
    
    # Personalizar configuración
    manager.workflow_config.update({
        'train_ratio': 0.8,  # Más datos para entrenamiento
        'val_ratio': 0.15,
        'test_ratio': 0.05,
        'min_samples_per_class': 50,  # Mínimo más alto
        'max_augmentation_factor': 3
    })
    
    # Personalizar resoluciones
    manager.standard_resolutions.update({
        'yolo': (1024, 1024),  # Resolución más alta
        'coco': (1280, 1280),
        'unet': (768, 768)
    })
    
    # Agregar nuevas clases unificadas
    manager.unified_classes.update({
        'orthodontic': ['brackets', 'braces', 'orthodontic', 'wire'],
        'prosthetic': ['denture', 'prosthetic', 'artificial_tooth']
    })
    
    print("⚙️ Configuración personalizada aplicada:")
    print(f"   📊 Split train/val/test: {manager.workflow_config['train_ratio']}/{manager.workflow_config['val_ratio']}/{manager.workflow_config['test_ratio']}")
    print(f"   🎯 Resolución YOLO: {manager.standard_resolutions['yolo']}")
    print(f"   🏷️ Clases unificadas: {len(manager.unified_classes)}")
    
    # Ejecutar workflow con configuración personalizada
    print(f"\n🚀 Ejecutando workflow con configuración personalizada...")
    manager.run_complete_workflow()
    
    print(f"\n🎉 ¡Configuración personalizada completada!")
    print(f"📂 Resultados en: {manager.output_path}")


def ejemplo_uso_modulos():
    """🧩 Ejemplo de uso directo de módulos individuales."""
    print("\n\n🧩 EJEMPLO MODULAR - USO DIRECTO DE MÓDULOS")
    print("="*50)
    
    # Importar módulos individuales
    from Src.data_analyzer import DataAnalyzer
    from Src.data_processor import DataProcessor
    from Utils.visualization import DatasetVisualizer
    
    # 1. Usar solo el analizador
    print("\n1️⃣ Usando DataAnalyzer independientemente...")
    
    unified_classes = {
        'caries': ['caries', 'cavity', 'decay'],
        'tooth': ['tooth', 'teeth', 'diente']
    }
    
    analyzer = DataAnalyzer(Path("_dataSets"), unified_classes)
    analysis = analyzer.scan_datasets()
    
    print(f"   📊 Análisis directo: {analysis['total_datasets']} datasets")
    
    # 2. Usar solo el procesador
    print("\n2️⃣ Usando DataProcessor independientemente...")
    
    standard_resolutions = {'yolo': (640, 640)}
    safety_config = {'read_only_source': True, 'verify_copy': True}
    
    processor = DataProcessor(unified_classes, standard_resolutions, safety_config)
    
    # Ejemplo de unificación de clases
    unified_name = processor.unify_class_names("CARIES")
    print(f"   🏷️ Clase unificada: 'CARIES' -> '{unified_name}'")
    
    # 3. Usar solo el visualizador
    print("\n3️⃣ Usando DatasetVisualizer independientemente...")
    
    visualizer = DatasetVisualizer()
    output_path = Path("Dist/dental_ai")
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer.create_detailed_report(analysis, output_path)
    print(f"   📋 Reporte creado en: {output_path}/dental_dataset_report.md")
    
    print(f"\n🎉 ¡Uso modular completado!")


def main():
    """🚀 Función principal con menú de ejemplos."""
    print("🦷 EJEMPLOS DE USO - DENTAL AI WORKFLOW MANAGER v2.0")
    print("="*60)
    print("Selecciona qué ejemplo ejecutar:")
    print()
    print("1. 📝 Ejemplo básico (recomendado)")
    print("2. 🔄 Procesamiento avanzado")
    print("3. ⚙️ Configuración personalizada")
    print("4. 🧩 Uso modular de componentes")
    print("5. 🚀 Ejecutar todos los ejemplos")
    print("0. ❌ Salir")
    
    choice = input("\n🎯 Selecciona una opción (1-5): ").strip()
    
    try:
        if choice == '1':
            ejemplo_basico()
        elif choice == '2':
            ejemplo_procesamiento_avanzado()
        elif choice == '3':
            ejemplo_personalizado()
        elif choice == '4':
            ejemplo_uso_modulos()
        elif choice == '5':
            print("🚀 Ejecutando todos los ejemplos...")
            ejemplo_basico()
            ejemplo_procesamiento_avanzado()
            ejemplo_personalizado()
            ejemplo_uso_modulos()
        elif choice == '0':
            print("👋 ¡Hasta luego!")
            return
        else:
            print("❌ Opción no válida")
            return
        
        print(f"\n" + "="*60)
        print("✅ ¡EJEMPLOS COMPLETADOS EXITOSAMENTE!")
        print("📂 Revisa los directorios de salida para ver los resultados")
        print("📋 Consulta DENTAL_AI_GUIDE.md para más información")
        print("🎯 Usa main.py para el sistema completo interactivo")
        
    except Exception as e:
        print(f"\n❌ Error ejecutando ejemplo: {e}")
        print("💡 Asegúrate de tener instaladas todas las dependencias:")
        print("   pip install -r requirements.txt")


if __name__ == "__main__":
    main()
