#!/usr/bin/env python3
"""
🎪 Demo del Smart Dental AI Workflow Manager v3.0
=================================================

Script de demostración que muestra todas las capacidades del nuevo sistema
inteligente de gestión de datasets dentales.

Author: Anton Sychev
Version: 3.0 (Demo)
"""

import sys
import time
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def print_section(title: str, subtitle: str = ""):
    """Imprimir sección con formato bonito."""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    if subtitle:
        print(f"   {subtitle}")
    print("="*60)

def wait_for_user():
    """Esperar input del usuario."""
    input("\n⏸️ Presiona Enter para continuar...")

def demo_smart_workflow():
    """🎪 Demostración completa del smart workflow."""
    print("🎪 DEMO - SMART DENTAL AI WORKFLOW MANAGER v3.0")
    print("="*60)
    print("¡Bienvenido a la demostración del sistema inteligente!")
    print()
    print("🎯 Lo que verás en esta demo:")
    print("   • Análisis automático de datasets")
    print("   • Detección inteligente de categorías")
    print("   • Selección interactiva de datos")
    print("   • Conversión a múltiples formatos")
    print("   • Balanceado inteligente")
    print("   • Verificación de calidad")
    print("   • Generación de scripts de entrenamiento")
    
    wait_for_user()
    
    # Importar aquí para evitar errores si no están disponibles
    try:
        from Src.smart_workflow_manager import SmartDentalWorkflowManager
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        print("💡 Asegúrate de que todos los archivos estén en su lugar")
        return
    
    # Inicializar sistema
    print_section("INICIALIZACIÓN DEL SISTEMA", "Configurando el Smart Workflow Manager")
    
    manager = SmartDentalWorkflowManager(
        base_path="_dataSets",
        output_path="Dist/dental_ai_demo"
    )
    
    print("✅ Sistema inicializado correctamente")
    print(f"📂 Directorio base: {manager.base_path}")
    print(f"📁 Directorio salida: {manager.output_path}")
    print(f"🏷️ Clases unificadas: {len(manager.unified_classes)} categorías")
    print(f"📊 Formatos soportados: YOLO, COCO, U-Net, Clasificación")
    
    wait_for_user()
    
    # Fase 1: Análisis automático
    print_section("FASE 1: ANÁLISIS AUTOMÁTICO", "Escaneando y analizando datasets disponibles")
    
    try:
        print("🔍 Iniciando escaneo de datasets...")
        manager._scan_and_analyze()
        
        print("\n📊 Resultados del análisis:")
        if manager.current_analysis:
            print(f"   • Datasets encontrados: {manager.current_analysis.get('total_datasets', 0)}")
            print(f"   • Imágenes totales: {manager.current_analysis.get('total_images', 0):,}")
            print(f"   • Categorías detectadas: {len(manager.available_categories)}")
            
            # Mostrar distribución por formato
            format_dist = manager.current_analysis.get('format_distribution', {})
            if format_dist:
                print(f"\n📋 Distribución por formato:")
                for fmt, count in format_dist.items():
                    print(f"      {fmt}: {count} dataset(s)")
        else:
            print("⚠️ No se encontraron datasets para analizar")
            print("💡 Asegúrate de que el directorio '_dataSets' existe y contiene datos")
            
    except Exception as e:
        print(f"❌ Error en análisis: {e}")
    
    wait_for_user()
    
    # Fase 2: Análisis de categorías
    print_section("FASE 2: ANÁLISIS DE CATEGORÍAS", "Mostrando categorías dentales detectadas")
    
    try:
        if manager.available_categories:
            print("🏷️ Categorías detectadas:")
            
            sorted_categories = sorted(
                manager.available_categories.items(),
                key=lambda x: x[1]['total_samples'],
                reverse=True
            )
            
            for i, (category, info) in enumerate(sorted_categories[:10], 1):
                print(f"   {i:2d}. {category}")
                print(f"       📊 Muestras: {info['total_samples']:,}")
                print(f"       📋 Formatos: {', '.join(info['formats'])}")
                print(f"       📁 Datasets: {len(info['datasets'])}")
                
                # Mostrar algunos nombres originales
                if 'original_names' in info and info['original_names']:
                    original_sample = list(info['original_names'])[:3]
                    print(f"       🔤 Ejemplos: {', '.join(original_sample)}")
                print()
        else:
            print("⚠️ No se detectaron categorías")
            print("💡 Esto puede deberse a:")
            print("   • Datasets sin anotaciones")
            print("   • Formatos no reconocidos")
            print("   • Estructura de directorios no estándar")
            
    except Exception as e:
        print(f"❌ Error mostrando categorías: {e}")
    
    wait_for_user()
    
    # Fase 3: Selección automática inteligente
    print_section("FASE 3: SELECCIÓN INTELIGENTE", "Seleccionando datasets automáticamente")
    
    try:
        # Seleccionar categorías con suficientes datos
        min_samples = 5  # Umbral bajo para la demo
        selected_count = 0
        
        manager.selected_datasets = {}
        for category, info in manager.available_categories.items():
            if info['total_samples'] >= min_samples:
                manager.selected_datasets[category] = info
                selected_count += 1
        
        print(f"🎯 Criterio de selección: ≥{min_samples} muestras por categoría")
        print(f"✅ Categorías seleccionadas: {selected_count}")
        
        if manager.selected_datasets:
            total_samples = sum(info['total_samples'] for info in manager.selected_datasets.values())
            print(f"📊 Total muestras seleccionadas: {total_samples:,}")
            
            print(f"\n🏷️ Categorías seleccionadas:")
            for category, info in manager.selected_datasets.items():
                print(f"   • {category}: {info['total_samples']:,} muestras")
        else:
            print("⚠️ No hay categorías que cumplan el criterio")
            print("💡 Reduciendo umbral a 1 muestra...")
            
            # Seleccionar todas las categorías disponibles
            manager.selected_datasets = manager.available_categories.copy()
            print(f"✅ Seleccionadas todas las {len(manager.selected_datasets)} categorías")
            
    except Exception as e:
        print(f"❌ Error en selección: {e}")
    
    wait_for_user()
    
    # Fase 4: Análisis de distribución
    print_section("FASE 4: ANÁLISIS DE DISTRIBUCIÓN", "Analizando balance de los datos")
    
    try:
        if manager.selected_datasets:
            print("📊 Distribución actual de datos:")
            manager._show_data_distribution()
            
            # Calcular estadísticas de balance
            samples_list = [info['total_samples'] for info in manager.selected_datasets.values()]
            if samples_list:
                import numpy as np
                mean_samples = np.mean(samples_list)
                std_samples = np.std(samples_list)
                min_samples = min(samples_list)
                max_samples = max(samples_list)
                
                print(f"\n📈 Estadísticas de distribución:")
                print(f"   • Promedio: {mean_samples:.1f} muestras")
                print(f"   • Desviación estándar: {std_samples:.1f}")
                print(f"   • Mínimo: {min_samples} muestras")
                print(f"   • Máximo: {max_samples} muestras")
                
                # Calcular score de balance
                if mean_samples > 0:
                    balance_score = max(0, 100 - (std_samples / mean_samples * 100))
                    print(f"   • Score de balance: {balance_score:.1f}/100")
                    
                    if balance_score >= 80:
                        print("   ✅ Distribución muy equilibrada")
                    elif balance_score >= 60:
                        print("   🟡 Distribución moderadamente equilibrada")
                    else:
                        print("   🔴 Distribución desbalanceada - se recomienda augmentación")
        else:
            print("⚠️ No hay datasets seleccionados para analizar")
            
    except Exception as e:
        print(f"❌ Error en análisis de distribución: {e}")
    
    wait_for_user()
    
    # Fase 5: Conversión de formatos
    print_section("FASE 5: CONVERSIÓN DE FORMATOS", "Preparando datos para entrenamiento")
    
    try:
        if manager.selected_datasets:
            print("🔄 Iniciando conversión a múltiples formatos...")
            print("   • YOLO (detección de objetos)")
            print("   • COCO (detección y segmentación)")
            print("   • Clasificación (directorios por clase)")
            
            # Simular conversión (implementación básica)
            manager._convert_multiple_formats()
            
            print("\n✅ Conversión completada:")
            for format_name, result in manager.conversion_results.items():
                status = result.get('status', 'unknown')
                images = result.get('images', 0)
                print(f"   • {format_name.upper()}: {status}")
                if images > 0:
                    print(f"     Imágenes procesadas: {images}")
        else:
            print("⚠️ No hay datasets seleccionados para convertir")
            
    except Exception as e:
        print(f"❌ Error en conversión: {e}")
    
    wait_for_user()
    
    # Fase 6: Verificación y validación
    print_section("FASE 6: VERIFICACIÓN Y VALIDACIÓN", "Validando resultados del procesamiento")
    
    try:
        print("✅ Iniciando verificación completa...")
        manager._verify_and_validate()
        
        # Verificar estructura creada
        output_path = manager.output_path
        if output_path.exists():
            print(f"\n📁 Estructura creada en: {output_path}")
            
            # Listar directorios principales
            main_dirs = ['datasets', 'scripts', 'reports', 'analysis']
            for dir_name in main_dirs:
                dir_path = output_path / dir_name
                if dir_path.exists():
                    files_count = len(list(dir_path.rglob('*')))
                    print(f"   ✅ {dir_name}/: {files_count} archivos")
                else:
                    print(f"   ⚠️ {dir_name}/: no creado")
        
        print("\n🎯 Validación completada")
        
    except Exception as e:
        print(f"❌ Error en verificación: {e}")
    
    wait_for_user()
    
    # Fase 7: Generación de scripts
    print_section("FASE 7: SCRIPTS DE ENTRENAMIENTO", "Generando scripts listos para usar")
    
    try:
        print("📝 Generando scripts de entrenamiento...")
        manager._generate_training_scripts()
        
        scripts_dir = manager.output_path / "scripts"
        if scripts_dir.exists():
            script_files = list(scripts_dir.glob('*.py'))
            
            print(f"\n✅ Scripts generados en: {scripts_dir}")
            for script_file in script_files:
                print(f"   📝 {script_file.name}")
                
            if script_files:
                print(f"\n🚀 Para entrenar modelos:")
                print(f"   cd {scripts_dir}")
                print(f"   python train_yolo.py")
        else:
            print("⚠️ Directorio de scripts no creado")
            
    except Exception as e:
        print(f"❌ Error generando scripts: {e}")
    
    wait_for_user()
    
    # Fase 8: Reporte final
    print_section("FASE 8: REPORTE FINAL", "Resumen completo del procesamiento")
    
    try:
        print("📋 Generando reporte final...")
        manager._show_analysis_report()
        
        # Resumen de la demo
        print(f"\n🎉 ¡DEMO COMPLETADA EXITOSAMENTE!")
        print(f"\n📊 Resumen de la sesión:")
        
        if manager.current_analysis:
            print(f"   • Datasets analizados: {manager.current_analysis.get('total_datasets', 0)}")
            print(f"   • Imágenes procesadas: {manager.current_analysis.get('total_images', 0):,}")
            
        print(f"   • Categorías detectadas: {len(manager.available_categories)}")
        print(f"   • Categorías seleccionadas: {len(manager.selected_datasets)}")
        print(f"   • Formatos generados: {len(manager.conversion_results)}")
        
        print(f"\n📂 Resultados disponibles en:")
        print(f"   {manager.output_path}")
        
        print(f"\n🎯 Próximos pasos:")
        print(f"   1. Revisar los datasets generados")
        print(f"   2. Ejecutar scripts de entrenamiento")
        print(f"   3. Validar resultados de entrenamiento")
        print(f"   4. Iterar con más datos si es necesario")
        
    except Exception as e:
        print(f"❌ Error en reporte final: {e}")
    
    print("\n" + "="*60)
    print("🎪 FIN DE LA DEMOSTRACIÓN")
    print("="*60)
    print("¡Gracias por probar el Smart Dental AI Workflow Manager!")
    print("📚 Consulta SMART_WORKFLOW_GUIDE.md para más información")
    print("🚀 Usa smart_dental_workflow.py para el sistema completo")

def demo_rapido():
    """⚡ Demo rápido sin interacción."""
    print("⚡ DEMO RÁPIDO - SMART DENTAL AI WORKFLOW")
    print("="*50)
    
    try:
        from Src.smart_workflow_manager import SmartDentalWorkflowManager
        
        manager = SmartDentalWorkflowManager(
            base_path="_dataSets",
            output_path="Dist/dental_ai_demo_rapido"
        )
        
        print("🔍 Analizando datasets...")
        manager._scan_and_analyze()
        
        print("📊 Mostrando categorías...")
        if manager.available_categories:
            print(f"   Categorías encontradas: {len(manager.available_categories)}")
            for category, info in list(manager.available_categories.items())[:5]:
                print(f"   • {category}: {info['total_samples']} muestras")
        
        print("\n✅ Demo rápido completado")
        print(f"📂 Para más detalles revisa: {manager.output_path}")
        
    except Exception as e:
        print(f"❌ Error en demo rápido: {e}")

def main():
    """🚀 Función principal."""
    print("🎪 SELECTOR DE DEMO")
    print("="*30)
    print("1. 🎪 Demo completa interactiva")
    print("2. ⚡ Demo rápida")
    print("0. ❌ Salir")
    
    choice = input("\n🎯 Selecciona demo: ").strip()
    
    if choice == '1':
        demo_smart_workflow()
    elif choice == '2':
        demo_rapido()
    elif choice == '0':
        print("👋 ¡Hasta luego!")
    else:
        print("❌ Opción inválida")

if __name__ == "__main__":
    main()
