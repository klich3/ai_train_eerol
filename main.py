#!/usr/bin/env python3
"""
🦷 DENTAL AI WORKFLOW MANAGER
===============================

Sistema modular para gestión de datasets dentales
- Análisis automático de datasets
- Fusión y unificación de formatos
- Generación de estructura dental-ai
- Scripts de entrenamiento y API

Author: Anton Sychev (anton at sychev dot xyz)
Created: 2025-01-XX
Version: 2.0 (Modular)
"""

import os
import sys
from pathlib import Path

# Agregar la carpeta Src al path para importar módulos
sys.path.append(str(Path(__file__).parent / "Src"))

from Src.workflow_manager import DentalDataWorkflowManager
from Utils.visualization import DatasetVisualizer
from Utils.data_augmentation import DataBalancer, QualityChecker


def print_banner():
    """🎨 Imprime el banner del sistema."""
    banner = """
    ████████ ████████ ██████   ██████  ██      
    ██       ██       ██   ██ ██    ██ ██      
    ██████   █████    ██████  ██    ██ ██      
    ██       ██       ██   ██ ██    ██ ██      
    ████████ ████████ ██   ██  ██████  ████████
    
    🦷 DENTAL AI WORKFLOW MANAGER v2.0
    ===================================
    
    ✨ Sistema Modular para Datasets Dentales
    📊 Análisis • 🔄 Fusión • 🏗️ Estructura • 🚀 API
    
    🛡️ MODO SEGURO: Solo lectura en origen
    📂 Salida: Dist/dental_ai/
    """
    print(banner)


def show_main_menu():
    """📋 Muestra el menú principal."""
    menu = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🎛️ MENÚ PRINCIPAL                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  📊 ANÁLISIS                                                 ║
    ║  1. Escanear y analizar datasets                             ║
    ║  2. Crear visualizaciones y dashboard                        ║
    ║  3. Verificar calidad de datasets                            ║
    ║                                                              ║
    ║  🔄 PROCESAMIENTO                                            ║
    ║  4. Fusionar datasets YOLO                                   ║
    ║  5. Fusionar datasets COCO                                   ║
    ║  6. Crear dataset de clasificación                           ║
    ║  7. Balancear datasets (augmentación)                        ║
    ║                                                              ║
    ║  🏗️ ESTRUCTURA Y SCRIPTS                                     ║
    ║  8. Crear estructura dental-ai completa                      ║
    ║  9. Generar scripts de entrenamiento                         ║
    ║  10. Crear template de API                                   ║
    ║                                                              ║
    ║  🚀 WORKFLOW COMPLETO                                        ║
    ║  11. Ejecutar workflow completo automático                   ║
    ║                                                              ║
    ║  ℹ️ INFORMACIÓN                                              ║
    ║  12. Ver configuración actual                                ║
    ║  13. Estadísticas de datasets                                ║
    ║  14. Ayuda y documentación                                   ║
    ║                                                              ║
    ║  0. Salir                                                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(menu)


def show_config_info(manager: DentalDataWorkflowManager):
    """ℹ️ Muestra información de configuración."""
    print(f"""
    🛡️ CONFIGURACIÓN ACTUAL
    =====================
    
    📂 Directorio fuente: {manager.base_path}
    📁 Directorio salida: {manager.output_path}
    
    🔒 CONFIGURACIÓN DE SEGURIDAD:
    • Modo solo lectura: {manager.safety_config['read_only_source']}
    • Verificación de copia: {manager.safety_config['verify_copy']}
    • Backup habilitado: {manager.safety_config['backup_enabled']}
    • Preservar estructura: {manager.safety_config['preserve_original_structure']}
    
    ⚙️ CONFIGURACIÓN DE WORKFLOW:
    • Ratio train/val/test: {manager.workflow_config['train_ratio']}/{manager.workflow_config['val_ratio']}/{manager.workflow_config['test_ratio']}
    • Mínimo muestras por clase: {manager.workflow_config['min_samples_per_class']}
    • Factor máximo de augmentación: {manager.workflow_config['max_augmentation_factor']}
    
    🎯 RESOLUCIONES ESTÁNDAR:
    • YOLO: {manager.standard_resolutions['yolo']}
    • COCO: {manager.standard_resolutions['coco']}
    • U-Net: {manager.standard_resolutions['unet']}
    
    🏷️ CLASES UNIFICADAS: {len(manager.unified_classes)} categorías principales
    """)


def show_help():
    """❓ Muestra información de ayuda."""
    help_text = """
    🆘 AYUDA Y DOCUMENTACIÓN
    ========================
    
    📖 DESCRIPCIÓN:
    El Dental AI Workflow Manager es un sistema modular para gestionar
    datasets dentales, desde el análisis hasta la preparación para entrenamiento.
    
    🔄 FLUJO DE TRABAJO RECOMENDADO:
    1. Escanear datasets (opción 1)
    2. Crear visualizaciones (opción 2)
    3. Crear estructura dental-ai (opción 8)
    4. Fusionar datasets por formato (opciones 4-6)
    5. Generar scripts de entrenamiento (opción 9)
    6. Crear API template (opción 10)
    
    🛡️ GARANTÍAS DE SEGURIDAD:
    • Todos los datasets originales permanecen INTACTOS
    • Solo operaciones de LECTURA en directorios fuente
    • Todas las modificaciones se hacen en Dist/dental_ai/
    • Verificación de integridad en todas las copias
    
    📁 ESTRUCTURA DE SALIDA:
    Dist/dental_ai/
    ├── datasets/          # Datasets procesados
    ├── models/           # Modelos entrenados
    ├── training/         # Scripts de entrenamiento
    ├── api/              # API de inferencia
    └── docs/             # Documentación
    
    📋 ARCHIVOS GENERADOS:
    • dental_dataset_analysis.json - Análisis completo
    • datasets_summary_table.csv - Tabla resumen
    • dental_datasets_dashboard.html - Dashboard interactivo
    • workflow_report.json - Reporte final
    
    💡 CONSEJOS:
    • Usa el workflow completo (opción 11) para automatizar todo
    • Revisa las visualizaciones antes de procesar
    • Los datasets de alta calidad (>80) son prioritarios
    • La augmentación ayuda a balancear clases minoritarias
    
    📞 SOPORTE:
    Consulta DENTAL_AI_GUIDE.md para documentación completa
    """
    print(help_text)


def main():
    """🚀 Función principal del sistema."""
    print_banner()
    
    # Configurar rutas por defecto
    base_path = "_dataSets"
    output_path = "Dist/dental_ai"
    
    # Verificar si existe el directorio fuente
    if not Path(base_path).exists():
        print(f"⚠️ Directorio fuente no encontrado: {base_path}")
        print("📁 Asegúrate de ejecutar el script desde el directorio correcto")
        return
    
    # Preguntar al usuario si quiere usar rutas personalizadas
    print(f"📂 Directorio fuente por defecto: {base_path}")
    print(f"📁 Directorio salida por defecto: {output_path}")
    
    custom_paths = input("\n¿Usar rutas personalizadas? (s/N): ").strip().lower()
    
    if custom_paths in ['s', 'si', 'sí', 'yes', 'y']:
        base_path = input("📂 Directorio fuente: ").strip() or base_path
        output_path = input("📁 Directorio salida: ").strip() or output_path
    
    # Inicializar manager principal
    print(f"\n🎛️ Inicializando Workflow Manager...")
    manager = DentalDataWorkflowManager(base_path, output_path)
    
    # Inicializar utilidades
    visualizer = DatasetVisualizer()
    balancer = DataBalancer()
    quality_checker = QualityChecker()
    
    print(f"✅ Sistema inicializado correctamente")
    print(f"📂 Fuente: {manager.base_path}")
    print(f"📁 Salida: {manager.output_path}")
    
    # Loop principal del menú
    while True:
        show_main_menu()
        choice = input("\n🎯 Selecciona una opción: ").strip()
        
        try:
            if choice == '1':
                print(f"\n🔍 ESCANEANDO Y ANALIZANDO DATASETS...")
                analysis = manager.scan_and_analyze_datasets()
                print(f"\n✅ Análisis completado:")
                print(f"   📊 Datasets encontrados: {analysis['total_datasets']}")
                print(f"   🖼️ Imágenes totales: {analysis['total_images']:,}")
                print(f"   📋 Formatos: {list(analysis['format_distribution'].keys())}")
            
            elif choice == '2':
                print(f"\n📊 CREANDO VISUALIZACIONES Y DASHBOARD...")
                
                # Cargar análisis existente o crear uno nuevo
                analysis = manager.get_dataset_statistics()
                
                # Crear dashboard completo
                visualizer.create_overview_dashboard(analysis, manager.output_path)
                visualizer.create_class_wordcloud(analysis, manager.output_path)
                visualizer.create_detailed_report(analysis, manager.output_path)
                
                print(f"\n✅ Visualizaciones creadas:")
                print(f"   📊 Dashboard: {manager.output_path}/dental_datasets_dashboard.html")
                print(f"   🎨 Word cloud: {manager.output_path}/categories_wordcloud.png")
                print(f"   📋 Reporte: {manager.output_path}/dental_dataset_report.md")
            
            elif choice == '3':
                print(f"\n🔍 VERIFICANDO CALIDAD DE DATASETS...")
                analysis = manager.get_dataset_statistics()
                
                print(f"\n📊 REPORTE DE CALIDAD:")
                print(f"="*50)
                
                dataset_details = analysis.get('dataset_details', {})
                sorted_datasets = sorted(dataset_details.items(), 
                                       key=lambda x: x[1]['quality_score'], reverse=True)
                
                for dataset_path, info in sorted_datasets[:10]:  # Top 10
                    dataset_name = Path(dataset_path).name
                    quality = info['quality_score']
                    
                    if quality >= 80:
                        status = "🟢 EXCELENTE"
                    elif quality >= 60:
                        status = "🟡 BUENO"
                    elif quality >= 40:
                        status = "🟠 REGULAR"
                    else:
                        status = "🔴 BAJO"
                    
                    print(f"   {status} {dataset_name}: {quality:.1f}/100")
                
                print(f"\n💡 Usa datasets con calidad >80 para mejores resultados")
            
            elif choice == '4':
                print(f"\n🔄 FUSIONANDO DATASETS YOLO...")
                confirm = input("¿Proceder con la fusión? (s/N): ").strip().lower()
                
                if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                    stats = manager.merge_yolo_datasets()
                    
                    if stats:
                        print(f"\n✅ Fusión YOLO completada:")
                        print(f"   📊 Imágenes procesadas: {stats['total_images']:,}")
                        print(f"   🏷️ Anotaciones: {stats['total_annotations']:,}")
                        print(f"   📋 Datasets fusionados: {len(stats['datasets_processed'])}")
                        print(f"   📂 Ubicación: {manager.output_path}/datasets/detection_combined/")
                    else:
                        print("⚠️ No se encontraron datasets YOLO para fusionar")
                else:
                    print("❌ Operación cancelada")
            
            elif choice == '5':
                print(f"\n🔄 FUSIONANDO DATASETS COCO...")
                confirm = input("¿Proceder con la fusión? (s/N): ").strip().lower()
                
                if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                    stats = manager.merge_coco_datasets()
                    
                    if stats:
                        print(f"\n✅ Fusión COCO completada:")
                        print(f"   📊 Imágenes procesadas: {stats['total_images']:,}")
                        print(f"   🏷️ Anotaciones: {stats['total_annotations']:,}")
                        print(f"   📋 Datasets fusionados: {len(stats['datasets_processed'])}")
                        print(f"   📂 Ubicación: {manager.output_path}/datasets/segmentation_coco/")
                    else:
                        print("⚠️ No se encontraron datasets COCO para fusionar")
                else:
                    print("❌ Operación cancelada")
            
            elif choice == '6':
                print(f"\n📁 CREANDO DATASET DE CLASIFICACIÓN...")
                confirm = input("¿Proceder con la creación? (s/N): ").strip().lower()
                
                if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                    stats = manager.create_classification_dataset()
                    
                    if stats:
                        print(f"\n✅ Dataset de clasificación creado:")
                        print(f"   📊 Imágenes procesadas: {stats['total_images']:,}")
                        print(f"   📋 Clases detectadas: {len(stats['class_distribution'])}")
                        print(f"   📂 Ubicación: {manager.output_path}/datasets/classification/")
                    else:
                        print("⚠️ No se encontraron datasets de imágenes para clasificación")
                else:
                    print("❌ Operación cancelada")
            
            elif choice == '7':
                print(f"\n⚖️ BALANCEANDO DATASETS (AUGMENTACIÓN)...")
                
                # Mostrar datasets disponibles para balancear
                detection_path = manager.output_path / "datasets" / "detection_combined"
                classification_path = manager.output_path / "datasets" / "classification"
                
                available_datasets = []
                if detection_path.exists():
                    available_datasets.append(("YOLO Detection", detection_path))
                if classification_path.exists():
                    available_datasets.append(("Classification", classification_path))
                
                if not available_datasets:
                    print("⚠️ No hay datasets procesados para balancear")
                    print("💡 Ejecuta primero las opciones 4-6 para crear datasets")
                else:
                    print(f"\n📋 Datasets disponibles para balancear:")
                    for i, (name, path) in enumerate(available_datasets, 1):
                        print(f"   {i}. {name}")
                    
                    choice_balance = input("Selecciona dataset (número): ").strip()
                    
                    try:
                        idx = int(choice_balance) - 1
                        if 0 <= idx < len(available_datasets):
                            name, path = available_datasets[idx]
                            print(f"\n⚖️ Balanceando {name}...")
                            
                            if "YOLO" in name:
                                stats = balancer.balance_yolo_dataset(path)
                            else:
                                print("🚧 Balanceo de clasificación en desarrollo")
                                stats = {'total_augmented': 0}
                            
                            print(f"✅ Balanceo completado: {stats.get('total_augmented', 0)} muestras augmentadas")
                        else:
                            print("❌ Selección inválida")
                    except ValueError:
                        print("❌ Entrada inválida")
            
            elif choice == '8':
                print(f"\n🏗️ CREANDO ESTRUCTURA DENTAL-AI COMPLETA...")
                confirm = input("¿Proceder? (s/N): ").strip().lower()
                
                if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                    manager.create_dental_ai_structure()
                    
                    print(f"\n✅ Estructura dental-ai creada en: {manager.output_path}")
                    print(f"📂 Directorios principales:")
                    print(f"   • datasets/ (datasets procesados)")
                    print(f"   • models/ (modelos entrenados)")
                    print(f"   • training/ (scripts y configuraciones)")
                    print(f"   • api/ (API de inferencia)")
                    print(f"   • docs/ (documentación)")
                else:
                    print("❌ Operación cancelada")
            
            elif choice == '9':
                print(f"\n📝 GENERANDO SCRIPTS DE ENTRENAMIENTO...")
                confirm = input("¿Proceder? (s/N): ").strip().lower()
                
                if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                    manager.create_training_scripts()
                    
                    print(f"\n✅ Scripts de entrenamiento generados:")
                    print(f"   📝 YOLO: {manager.output_path}/training/scripts/train_yolo.py")
                    print(f"   📝 U-Net: {manager.output_path}/training/scripts/train_unet.py")
                    print(f"   📝 Classification: {manager.output_path}/training/scripts/train_classification.py")
                    print(f"   ⚙️ Configuraciones en: {manager.output_path}/training/configs/")
                else:
                    print("❌ Operación cancelada")
            
            elif choice == '10':
                print(f"\n🌐 CREANDO TEMPLATE DE API...")
                confirm = input("¿Proceder? (s/N): ").strip().lower()
                
                if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                    manager.create_api_template()
                    
                    print(f"\n✅ Template de API creado:")
                    print(f"   📝 Archivo principal: {manager.output_path}/api/main.py")
                    print(f"   📋 Requirements: {manager.output_path}/api/requirements.txt")
                    print(f"\n🚀 Para usar la API:")
                    print(f"   cd {manager.output_path}/api")
                    print(f"   pip install -r requirements.txt")
                    print(f"   python main.py")
                    print(f"   Navega a: http://localhost:8000/docs")
                else:
                    print("❌ Operación cancelada")
            
            elif choice == '11':
                print(f"\n🚀 EJECUTANDO WORKFLOW COMPLETO AUTOMÁTICO...")
                print(f"Esta operación realizará:")
                print(f"   1. 🏗️ Crear estructura dental-ai")
                print(f"   2. 🔍 Analizar todos los datasets")
                print(f"   3. 🔄 Fusionar datasets por formato")
                print(f"   4. 📝 Generar scripts de entrenamiento")
                print(f"   5. 🌐 Crear template de API")
                print(f"   6. 📊 Generar visualizaciones y reportes")
                
                confirm = input("\n¿Ejecutar workflow completo? (s/N): ").strip().lower()
                
                if confirm in ['s', 'si', 'sí', 'yes', 'y']:
                    print(f"\n🚀 Iniciando workflow completo...")
                    
                    try:
                        manager.run_complete_workflow()
                        
                        # Crear visualizaciones adicionales
                        analysis = manager.get_dataset_statistics()
                        visualizer.create_overview_dashboard(analysis, manager.output_path)
                        visualizer.create_class_wordcloud(analysis, manager.output_path)
                        visualizer.create_detailed_report(analysis, manager.output_path)
                        
                        print(f"\n🎉 WORKFLOW COMPLETADO EXITOSAMENTE!")
                        print(f"📂 Todos los resultados en: {manager.output_path}")
                        print(f"📊 Revisa el dashboard: {manager.output_path}/dental_datasets_dashboard.html")
                        print(f"📋 Reporte completo: {manager.output_path}/workflow_report.json")
                        
                    except Exception as e:
                        print(f"❌ Error en workflow: {e}")
                        print(f"💡 Revisa los logs para más detalles")
                else:
                    print("❌ Workflow cancelado")
            
            elif choice == '12':
                show_config_info(manager)
            
            elif choice == '13':
                print(f"\n📊 ESTADÍSTICAS DE DATASETS...")
                analysis = manager.get_dataset_statistics()
                
                print(f"📈 RESUMEN ESTADÍSTICO:")
                print(f"="*40)
                print(f"📊 Total datasets: {analysis.get('total_datasets', 0)}")
                print(f"🖼️ Total imágenes: {analysis.get('total_images', 0):,}")
                print(f"📋 Formatos detectados: {len(analysis.get('format_distribution', {}))}")
                
                format_dist = analysis.get('format_distribution', {})
                if format_dist:
                    print(f"\n📊 DISTRIBUCIÓN POR FORMATO:")
                    for fmt, count in format_dist.items():
                        print(f"   {fmt}: {count} datasets")
                
                dataset_details = analysis.get('dataset_details', {})
                if dataset_details:
                    qualities = [info['quality_score'] for info in dataset_details.values()]
                    avg_quality = sum(qualities) / len(qualities)
                    high_quality = len([q for q in qualities if q >= 80])
                    
                    print(f"\n🏆 MÉTRICAS DE CALIDAD:")
                    print(f"   📊 Calidad promedio: {avg_quality:.1f}/100")
                    print(f"   🌟 Datasets alta calidad (≥80): {high_quality}")
                    print(f"   🔝 Mejor dataset: {max(qualities):.1f}/100")
            
            elif choice == '14':
                show_help()
            
            elif choice == '0':
                print(f"\n👋 ¡Gracias por usar Dental AI Workflow Manager!")
                print(f"🛡️ Recuerda: Todos tus datos originales están seguros")
                print(f"🏗️ Tu estructura dental-ai está lista en: {manager.output_path}")
                print(f"📋 Consulta la documentación en DENTAL_AI_GUIDE.md")
                break
            
            else:
                print("❌ Opción no válida. Intenta de nuevo.")
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️ Operación interrumpida por el usuario")
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            print(f"💡 Si el problema persiste, revisa los logs o contacta soporte")
        
        # Pausa antes de volver al menú
        if choice != '0':
            input(f"\n📋 Presiona Enter para volver al menú principal...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n👋 Sistema cerrado por el usuario. ¡Hasta luego!")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        print(f"💡 Revisa que tienes todas las dependencias instaladas")
        print(f"📋 Consulta requirements.txt para la lista completa")
