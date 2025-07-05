#!/usr/bin/env python3
"""
🦷 Script de Ejemplo - Dental AI Workflow Manager (Legacy)

NOTA: Este es el ejemplo legacy. Para la versión 2.0 modular, usa:
- ejemplo_uso_v2.py (Sistema modular completo)
- Utils/advanced_analysis.py (Análisis avanzado integrado)
- StatisticsResults/ (Resultados de análisis)
- Wiki/ (Documentación actualizada)

Este script demuestra el uso del sistema legacy.
"""

from DataWorkflowManager import DentalDataWorkflowManager
from pathlib import Path
import json
import sys
import os

# Agregar rutas para los nuevos módulos
sys.path.append(str(Path(__file__).parent / "Src"))
sys.path.append(str(Path(__file__).parent / "Utils"))

def ejemplo_workflow_completo():
    """Ejemplo de workflow completo usando el manager."""
    
    print("🦷 EJEMPLO DE WORKFLOW DENTAL-AI")
    print("="*50)
    
    # Inicializar manager
    manager = DentalDataWorkflowManager(
        base_path="_dataSets",
        output_path="dental-ai"
    )
    
    # 1. Crear estructura dental-ai
    print("\n🏗️ 1. Creando estructura dental-ai...")
    manager.create_dental_ai_structure()
    
    # 2. Cargar datos de análisis
    print("\n📊 2. Cargando análisis de datasets...")
    data = manager.load_analysis_data()
    
    if not data:
        print("❌ No se encontró dental_dataset_analysis.json")
        print("Ejecuta primero: python demo_herramientas.py --analisis")
        print("O usa: from Utils.advanced_analysis import analyze_dental_datasets")
        return
    
    # 3. Análisis de distribución
    print("\n🔍 3. Analizando distribución de clases...")
    class_distribution, total_samples = manager.analyze_class_distribution(data)
    
    # 4. Identificar desbalances
    print("\n⚖️ 4. Identificando clases desbalanceadas...")
    imbalanced_classes = manager.identify_imbalanced_classes(total_samples)
    
    # 5. Recomendar fusiones
    print("\n🔗 5. Recomendando fusiones de datasets...")
    fusion_groups = manager.recommend_dataset_fusion(data)
    
    # 6. Crear estrategia de entrenamiento
    print("\n🎯 6. Generando estrategia de entrenamiento...")
    strategy = manager.create_balanced_split_strategy(total_samples)
    recommendations = manager.generate_training_recommendations(class_distribution, fusion_groups)
    
    # 7. Ejemplo de creación de dataset YOLO
    print("\n🧱 7. Ejemplo de creación de dataset YOLO...")
    
    # Obtener primeros datasets YOLO disponibles como ejemplo
    yolo_datasets = []
    for dataset_type, dataset_info in data.items():
        if dataset_type.startswith('_YOLO'):
            yolo_datasets.extend(list(dataset_info.get('datasets', {}).keys())[:2])
    
    if yolo_datasets:
        print(f"   Datasets de ejemplo: {yolo_datasets}")
        
        # Crear dataset unificado pequeño como ejemplo
        try:
            result = manager.create_unified_yolo_dataset(
                data, 
                yolo_datasets[:2], 
                "ejemplo_deteccion"
            )
            print(f"   ✅ Dataset de ejemplo creado: {result}")
        except Exception as e:
            print(f"   ⚠️ Error en ejemplo: {e}")
    
    # 8. Crear plantilla de API
    print("\n🌐 8. Creando plantilla de API...")
    manager.create_api_template()
    
    # 9. Guardar configuración
    print("\n💾 9. Guardando configuración del workflow...")
    manager.save_workflow_config(strategy, recommendations)
    
    # Resumen final
    print("\n" + "="*50)
    print("✅ EJEMPLO COMPLETADO")
    print("="*50)
    print(f"📁 Estructura creada en: dental-ai/")
    print(f"📊 Datasets analizados: {len(data)}")
    print(f"🏷️ Clases encontradas: {len(total_samples)}")
    print(f"⚖️ Clases desbalanceadas: {len(imbalanced_classes)}")
    print()
    print("🚀 PRÓXIMOS PASOS:")
    print("1. Revisar dental-ai/README.md")
    print("2. Explorar dental-ai/datasets/")
    print("3. Ejecutar scripts en dental-ai/training/")
    print("4. Probar API en dental-ai/api/")
    print()
    print("📖 Consulta DENTAL_AI_GUIDE.md para más detalles")

def ejemplo_creacion_datasets():
    """Ejemplo específico de creación de diferentes tipos de datasets."""
    
    print("🦷 EJEMPLO DE CREACIÓN DE DATASETS")
    print("="*40)
    
    manager = DentalDataWorkflowManager()
    data = manager.load_analysis_data()
    
    if not data:
        print("❌ No se encontró análisis de datasets")
        return
    
    # Mostrar datasets disponibles por tipo
    print("\n📋 DATASETS DISPONIBLES:")
    
    yolo_datasets = []
    coco_datasets = []
    classification_datasets = []
    
    for dataset_type, dataset_info in data.items():
        if not dataset_type.startswith('_'):
            continue
            
        format_type = manager._get_format_type(dataset_info.get('type', ''))
        datasets = list(dataset_info.get('datasets', {}).keys())
        
        if format_type == 'YOLO':
            yolo_datasets.extend(datasets)
        elif format_type == 'COCO':
            coco_datasets.extend(datasets)
        else:
            classification_datasets.extend(datasets)
    
    print(f"\n🧱 YOLO (Detección): {len(yolo_datasets)} datasets")
    for i, dataset in enumerate(yolo_datasets[:5]):
        print(f"   {i+1}. {dataset}")
    if len(yolo_datasets) > 5:
        print(f"   ... y {len(yolo_datasets) - 5} más")
    
    print(f"\n🎨 COCO (Segmentación): {len(coco_datasets)} datasets")
    for i, dataset in enumerate(coco_datasets[:5]):
        print(f"   {i+1}. {dataset}")
    if len(coco_datasets) > 5:
        print(f"   ... y {len(coco_datasets) - 5} más")
    
    print(f"\n📂 Clasificación: {len(classification_datasets)} datasets")
    for i, dataset in enumerate(classification_datasets[:5]):
        print(f"   {i+1}. {dataset}")
    if len(classification_datasets) > 5:
        print(f"   ... y {len(classification_datasets) - 5} más")
    
    print("\n💡 PARA CREAR DATASETS UNIFICADOS:")
    print("   python DataWorkflowManager.py")
    print("   Seleccionar opción 3, 4 o 5 según el tipo deseado")

def verificar_estructura_dental_ai():
    """Verifica si la estructura dental-ai existe y está completa."""
    
    print("🔍 VERIFICACIÓN DE ESTRUCTURA DENTAL-AI")
    print("="*40)
    
    dental_ai_path = Path("dental-ai")
    
    if not dental_ai_path.exists():
        print("❌ La estructura dental-ai no existe")
        print("💡 Ejecuta: python DataWorkflowManager.py -> Opción 10")
        return False
    
    # Verificar directorios principales
    required_dirs = [
        "datasets",
        "models", 
        "training",
        "api",
        "docs"
    ]
    
    print("📁 Verificando directorios principales:")
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = dental_ai_path / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ (faltante)")
            all_exist = False
    
    # Verificar subdirectorios de datasets
    datasets_dir = dental_ai_path / "datasets"
    if datasets_dir.exists():
        print("\n📊 Subdirectorios de datasets:")
        dataset_types = ["detection_combined", "segmentation_coco", "segmentation_bitmap", "classification"]
        
        for dt in dataset_types:
            dt_path = datasets_dir / dt
            if dt_path.exists():
                datasets = list(dt_path.iterdir())
                print(f"   ✅ {dt}/ ({len(datasets)} datasets)")
            else:
                print(f"   📁 {dt}/ (vacío)")
    
    # Verificar archivos principales
    print("\n📄 Archivos principales:")
    important_files = [
        "README.md",
        "requirements.txt", 
        "config.yaml"
    ]
    
    for file_name in important_files:
        file_path = dental_ai_path / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name} (faltante)")
    
    if all_exist:
        print("\n✅ Estructura dental-ai completa y lista para usar")
    else:
        print("\n⚠️ Estructura incompleta. Recomendación:")
        print("   python DataWorkflowManager.py -> Opción 10")
    
    return all_exist

def ejemplo_herramientas_v2():
    """Demuestra el uso de las nuevas herramientas integradas v2.0."""
    
    print("🦷 EJEMPLO DE HERRAMIENTAS v2.0 (MODULAR)")
    print("="*50)
    
    print("\n📊 ANÁLISIS AVANZADO INTEGRADO:")
    print("   Las herramientas legacy han sido integradas en Utils/advanced_analysis.py")
    print("   Ofrecen funcionalidad expandida y mejor rendimiento")
    print()
    print("   🔧 Para ejecutar análisis completo:")
    print("   python -c \"from Utils.advanced_analysis import analyze_dental_datasets; analyze_dental_datasets('_dataSets')\"")
    print()
    print("   📁 Resultados se guardan en: StatisticsResults/")
    print("      • dental_dataset_analysis.json (datos completos)")
    print("      • *.png (gráficos y visualizaciones)")
    print("      • *.csv (tablas de datos)")
    print("      • *.html (dashboard interactivo)")
    
    print("\n🏗️ SISTEMA MODULAR v2.0:")
    print("   📁 Src/ - Módulos principales del sistema")
    print("      • workflow_manager.py - Gestor principal")
    print("      • data_analyzer.py - Analizador de datos") 
    print("      • data_processor.py - Procesador de datasets")
    print("      • structure_generator.py - Generador de estructuras")
    print()
    print("   🔧 Utils/ - Herramientas y utilidades")
    print("      • advanced_analysis.py - Análisis avanzado integrado")
    print("      • visualization.py - Visualización de datos")
    print("      • data_augmentation.py - Augmentación de datos")
    print("      • dental_format_converter.py - Convertidor de formatos")
    
    print("\n📚 DOCUMENTACIÓN CENTRALIZADA:")
    print("   📁 Wiki/ - Toda la documentación centralizada")
    print("      • README.md - Índice principal")
    print("      • USAGE_EXAMPLES.md - Ejemplos de uso")
    print("      • API_REFERENCE.md - Referencia de la API")
    print("      • WORKFLOW_GUIDE.md - Guía del workflow")
    
    print("\n🚀 EJEMPLOS DE USO RÁPIDO:")
    print()
    print("   # Análisis completo de datasets")
    print("   python ejemplo_uso_v2.py --quick")
    print()
    print("   # Workflow completo v2.0")
    print("   python ejemplo_uso_v2.py")
    print()
    print("   # Solo análisis de estadísticas")
    print("   python -c \"from Utils.advanced_analysis import *; analyze_dental_datasets('_dataSets')\"")
    
    # Verificar si las nuevas herramientas están disponibles
    print("\n🔍 VERIFICACIÓN DE HERRAMIENTAS v2.0:")
    
    advanced_analysis_path = Path("Utils/advanced_analysis.py")
    if advanced_analysis_path.exists():
        print("   ✅ Utils/advanced_analysis.py - Disponible")
    else:
        print("   ❌ Utils/advanced_analysis.py - No encontrado")
    
    statistics_dir = Path("StatisticsResults")
    if statistics_dir.exists():
        print("   ✅ StatisticsResults/ - Disponible")
        files_count = len(list(statistics_dir.iterdir()))
        print(f"      {files_count} archivos de resultados")
    else:
        print("   ❌ StatisticsResults/ - No encontrado")
    
    wiki_dir = Path("Wiki")
    if wiki_dir.exists():
        print("   ✅ Wiki/ - Disponible")
        md_files = len(list(wiki_dir.glob("*.md")))
        print(f"      {md_files} archivos de documentación")
    else:
        print("   ❌ Wiki/ - No encontrado")
    
    ejemplo_v2_path = Path("ejemplo_uso_v2.py")
    if ejemplo_v2_path.exists():
        print("   ✅ ejemplo_uso_v2.py - Disponible")
    else:
        print("   ❌ ejemplo_uso_v2.py - No encontrado")
    
    print("\n💡 MIGRACIÓN A v2.0:")
    print("   • Las herramientas legacy siguen funcionando")
    print("   • Las nuevas herramientas ofrecen más funcionalidades") 
    print("   • Los resultados ahora se centralizan en StatisticsResults/")
    print("   • La documentación está en Wiki/")
    print("   • Consulta Wiki/MIGRACION_V2.md para más detalles")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "completo":
            ejemplo_workflow_completo()
        elif sys.argv[1] == "datasets":
            ejemplo_creacion_datasets()
        elif sys.argv[1] == "verificar":
            verificar_estructura_dental_ai()
        elif sys.argv[1] == "herramientas_v2":
            ejemplo_herramientas_v2()
        else:
            print("Opciones: completo, datasets, verificar, herramientas_v2")
    else:
        print("🦷 EJEMPLOS DE USO DENTAL-AI")
        print("="*30)
        print("python ejemplo_uso.py completo   # Workflow completo")
        print("python ejemplo_uso.py datasets  # Ver datasets disponibles") 
        print("python ejemplo_uso.py verificar # Verificar estructura")
        print("python ejemplo_uso.py herramientas_v2 # Ejemplos de herramientas v2.0")
        print()
        print("O ejecuta directamente:")
        print("python DataWorkflowManager.py    # Menú interactivo")
