#!/usr/bin/env python3
"""
🎯 Demo de Uso del Smart Dental AI Workflow Manager v3.0
========================================================

Este script demuestra todas las formas de usar el sistema inteligente.
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def mostrar_estructura_disponible():
    """📁 Mostrar estructura disponible de datasets."""
    print("🧠 SMART DENTAL AI WORKFLOW MANAGER v3.0")
    print("="*60)
    print()
    
    base_path = Path("_dataSets")
    
    if not base_path.exists():
        print("❌ Directorio '_dataSets' no encontrado")
        print("💡 El sistema necesita que organices tus datasets así:")
        print()
        print("_dataSets/")
        print("├── _YOLO/           # Datasets en formato YOLO")
        print("│   ├── dataset1/")
        print("│   └── dataset2/")
        print("├── _COCO/           # Datasets en formato COCO")
        print("│   ├── dataset3/")
        print("│   └── dataset4/")
        print("├── _pure images and masks/  # Imágenes y máscaras")
        print("│   ├── dataset5/")
        print("│   └── dataset6/")
        print("└── _UNET/           # Datasets para U-Net")
        print("    ├── dataset7/")
        print("    └── dataset8/")
        return False
    
    print("📂 ESTRUCTURA ACTUAL DE DATASETS:")
    print(f"   Base: {base_path}")
    print()
    
    # Mostrar contenido
    total_datasets = 0
    for main_dir in ['_YOLO', '_COCO', '_pure images and masks', '_UNET']:
        dir_path = base_path / main_dir
        if dir_path.exists():
            subdirs = [d for d in dir_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            print(f"📁 {main_dir}: {len(subdirs)} datasets")
            total_datasets += len(subdirs)
            for subdir in subdirs[:3]:  # Mostrar primeros 3
                print(f"   • {subdir.name}")
            if len(subdirs) > 3:
                print(f"   ... y {len(subdirs) - 3} más")
        else:
            print(f"📁 {main_dir}: ❌ No existe")
        print()
    
    print(f"📊 Total de datasets encontrados: {total_datasets}")
    return total_datasets > 0

def demo_modos_de_uso():
    """🎮 Demostrar los diferentes modos de uso."""
    print("🎮 MODOS DE USO DISPONIBLES:")
    print("="*40)
    print()
    
    print("1️⃣ MODO INTERACTIVO (Recomendado)")
    print("   python smart_dental_workflow.py")
    print("   • Menú paso a paso")
    print("   • Selección interactiva de datasets")
    print("   • Control total del proceso")
    print()
    
    print("2️⃣ MODO AUTOMÁTICO")
    print("   python smart_dental_workflow.py --mode auto")
    print("   • Workflow completo sin intervención")
    print("   • Selección automática de datasets")
    print("   • Ideal para producción")
    print()
    
    print("3️⃣ MODO ANÁLISIS RÁPIDO")
    print("   python smart_dental_workflow.py --mode analysis")
    print("   • Solo escaneo y análisis")
    print("   • Sin conversión de datos")
    print("   • Perfecto para exploración")
    print()
    
    print("4️⃣ DEMO INTERACTIVA")
    print("   python demo_smart_workflow.py")
    print("   • Demostración paso a paso")
    print("   • Explicaciones detalladas")
    print("   • Ideal para aprendizaje")
    print()

def demo_uso_programatico():
    """🔧 Demostrar uso programático del sistema."""
    print("🔧 USO PROGRAMÁTICO:")
    print("="*30)
    print()
    
    print("```python")
    print("from Src.smart_workflow_manager import SmartDentalWorkflowManager")
    print()
    print("# Inicializar workflow")
    print("manager = SmartDentalWorkflowManager(")
    print("    base_path='_dataSets',")
    print("    output_path='Dist/dental_ai'")
    print(")")
    print()
    print("# Escanear y analizar datasets")
    print("manager._scan_and_analyze()")
    print()
    print("# Mostrar categorías disponibles")
    print("manager._show_categories_menu()")
    print()
    print("# Ejecutar workflow completo")
    print("manager._run_complete_workflow()")
    print("```")
    print()

def demo_funcionalidades():
    """⚡ Demostrar funcionalidades principales."""
    print("⚡ FUNCIONALIDADES PRINCIPALES:")
    print("="*40)
    print()
    
    print("🔍 ANÁLISIS INTELIGENTE:")
    print("   • Escaneo automático de todos los datasets")
    print("   • Detección de categorías dentales")
    print("   • Análisis de calidad de imágenes")
    print("   • Estadísticas detalladas")
    print()
    
    print("📊 CATEGORÍAS SOPORTADAS:")
    print("   • Caries: caries, decay, cavity")
    print("   • Implantes: implant, screw, titanium")
    print("   • Ortodonticos: bracket, wire, brace")
    print("   • Periodontales: gingivitis, periodontitis")
    print("   • Y muchas más...")
    print()
    
    print("🔄 CONVERSIÓN DE FORMATOS:")
    print("   • YOLO → Detección de objetos")
    print("   • COCO → Segmentación avanzada")
    print("   • U-Net → Segmentación médica")
    print("   • Clasificación → Categorización")
    print()
    
    print("⚖️ BALANCEADO INTELIGENTE:")
    print("   • Detección automática de desbalance")
    print("   • Augmentación de datos")
    print("   • División train/val/test")
    print("   • Verificación de calidad")
    print()

def demo_estructura_salida():
    """📁 Mostrar estructura de salida."""
    print("📁 ESTRUCTURA DE SALIDA:")
    print("="*30)
    print()
    
    print("Dist/dental_ai/")
    print("├── datasets/          # Datasets convertidos")
    print("│   ├── yolo/         # Formato YOLO")
    print("│   ├── coco/         # Formato COCO")
    print("│   ├── unet/         # Formato U-Net")
    print("│   └── classification/")
    print("│")
    print("├── scripts/           # Scripts de entrenamiento")
    print("│   ├── train_yolo.py")
    print("│   ├── train_unet.py")
    print("│   └── train_classifier.py")
    print("│")
    print("├── reports/           # Reportes de análisis")
    print("│   ├── analysis_report.md")
    print("│   ├── categories_report.json")
    print("│   └── quality_metrics.json")
    print("│")
    print("└── analysis/          # Datos de análisis")
    print("    ├── dataset_summary.json")
    print("    ├── class_distribution.png")
    print("    └── quality_analysis.png")
    print()

def demo_comandos_practicos():
    """💡 Mostrar comandos prácticos."""
    print("💡 COMANDOS PRÁCTICOS:")
    print("="*30)
    print()
    
    print("🚀 Para empezar rápidamente:")
    print("   python smart_dental_workflow.py")
    print()
    
    print("📊 Para solo analizar sin procesar:")
    print("   python smart_dental_workflow.py --mode analysis")
    print()
    
    print("🤖 Para workflow automático completo:")
    print("   python smart_dental_workflow.py --mode auto")
    print()
    
    print("🔧 Para ver opciones avanzadas:")
    print("   python smart_dental_workflow.py --help")
    print()
    
    print("📚 Para ver la demo completa:")
    print("   python demo_smart_workflow.py")
    print()

def main():
    """🎯 Función principal de la demo."""
    print()
    
    # 1. Mostrar estructura disponible
    tiene_datasets = mostrar_estructura_disponible()
    print()
    
    # 2. Modos de uso
    demo_modos_de_uso()
    print()
    
    # 3. Uso programático
    demo_uso_programatico()
    print()
    
    # 4. Funcionalidades
    demo_funcionalidades()
    print()
    
    # 5. Estructura de salida
    demo_estructura_salida()
    print()
    
    # 6. Comandos prácticos
    demo_comandos_practicos()
    
    # 7. Recomendaciones
    print("🎯 RECOMENDACIONES:")
    print("="*25)
    print()
    
    if tiene_datasets:
        print("✅ Tienes datasets disponibles!")
        print("👉 Ejecuta: python smart_dental_workflow.py")
        print("   para empezar con el modo interactivo")
    else:
        print("⚠️ No se encontraron datasets")
        print("👉 Organiza tus datasets en la estructura '_dataSets/'")
        print("👉 Luego ejecuta: python smart_dental_workflow.py")
    
    print()
    print("📚 Para más información:")
    print("   • README_SMART.md - Documentación completa")
    print("   • SMART_WORKFLOW_GUIDE.md - Guía detallada")
    print("   • Wiki/ - Documentación técnica")
    print()

if __name__ == "__main__":
    main()
