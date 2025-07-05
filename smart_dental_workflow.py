#!/usr/bin/env python3
"""
🧠 Smart Dental AI Workflow - Sistema Inteligente
================================================

Sistema avanzado para análisis, conversión y preparación de datasets dentales
con menú interactivo, análisis de categorías y verificación de calidad.

Características principales:
- 🔍 Escaneo inteligente de datasets
- 📊 Análisis automático de categorías
- 📦 Selección interactiva de datasets
- 🔄 Conversión a múltiples formatos (YOLO, COCO, U-Net)
- ⚖️ Balanceado inteligente de datos
- ✅ Verificación y validación automática
- 📝 Generación de scripts de entrenamiento

Author: Anton Sychev
Version: 3.0 (Smart Interactive)
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

from Src.smart_workflow_manager import SmartDentalWorkflowManager


def main():
    """🚀 Función principal."""
    print("🧠 SMART DENTAL AI WORKFLOW MANAGER v3.0")
    print("="*60)
    print("Sistema inteligente para preparación de datasets dentales")
    print()
    print("🎯 Funcionalidades:")
    print("   • Escaneo automático de datasets")
    print("   • Análisis de categorías disponibles")
    print("   • Selección interactiva de datos")
    print("   • Conversión a formatos YOLO/COCO/U-Net")
    print("   • Balanceado inteligente de datos")
    print("   • Verificación y validación")
    print("   • Generación de scripts de entrenamiento")
    print()
    
    try:
        # Verificar estructura de directorios
        base_path = Path("_dataSets")
        if not base_path.exists():
            print(f"❌ Directorio de datasets no encontrado: {base_path}")
            print("💡 Asegúrate de que existe el directorio '_dataSets' con tus datasets")
            return
        
        # Inicializar y ejecutar workflow inteligente
        workflow = SmartDentalWorkflowManager(
            base_path="_dataSets",
            output_path="Dist/dental_ai"
        )
        
        workflow.run_interactive_workflow()
        
    except KeyboardInterrupt:
        print("\n\n👋 Workflow interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        print("💡 Verifica que tienes instaladas todas las dependencias:")
        print("   pip install -r requirements.txt")


def ejemplo_workflow_automatico():
    """🤖 Ejemplo de workflow automático completo."""
    print("\n🤖 EJECUTANDO WORKFLOW AUTOMÁTICO...")
    
    workflow = SmartDentalWorkflowManager()
    
    # Ejecutar workflow completo sin intervención
    workflow._run_complete_workflow()
    
    print("\n✅ Workflow automático completado")
    print("📂 Revisa 'Dist/dental_ai' para los resultados")


def ejemplo_analisis_rapido():
    """⚡ Ejemplo de análisis rápido de datasets."""
    print("\n⚡ ANÁLISIS RÁPIDO DE DATASETS...")
    
    workflow = SmartDentalWorkflowManager()
    
    # Solo escanear y mostrar resumen
    workflow._scan_and_analyze()
    workflow._show_categories_menu()
    
    print("\n📊 Análisis rápido completado")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Dental AI Workflow Manager")
    parser.add_argument("--mode", choices=["interactive", "auto", "analysis"], 
                       default="interactive", help="Modo de ejecución")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        main()
    elif args.mode == "auto":
        ejemplo_workflow_automatico()
    elif args.mode == "analysis":
        ejemplo_analisis_rapido()
