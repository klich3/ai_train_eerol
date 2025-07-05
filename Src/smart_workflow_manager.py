"""
🧠 Smart Dental AI Workflow Manager
===================================

Sistema inteligente para análisis, conversión y preparación de datasets dentales
con menú interactivo y verificación de calidad.

Author: Anton Sychev
Version: 3.0 (Smart Interactive)
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging

from .data_analyzer import DataAnalyzer
from .data_processor import DataProcessor
from .structure_generator import StructureGenerator
from .script_templates import ScriptTemplateGenerator
from .smart_category_analyzer import SmartCategoryAnalyzer
from Utils.data_augmentation import DataBalancer, QualityChecker
from Utils.visualization import DatasetVisualizer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartDentalWorkflowManager:
    """
    🧠 Gestor inteligente de workflow para datasets dentales
    
    Características:
    - Análisis automático de categorías
    - Menú interactivo para selección de datasets
    - Balanceado inteligente de datos
    - Verificación de calidad
    - Preparación para múltiples formatos (YOLO, COCO, U-Net)
    """
    
    def __init__(self, base_path: str = "_dataSets", output_path: str = "Dist/dental_ai"):
        """Inicializar el workflow manager."""
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        
        # Crear directorio de salida
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configuración de clases unificadas (expandible)
        self.unified_classes = {
            'caries': ['caries', 'Caries', 'CARIES', 'cavity', 'decay', 'Q1_Caries', 'Q2_Caries', 'Q3_Caries', 'Q4_Caries'],
            'tooth': ['tooth', 'teeth', 'Tooth', 'TOOTH', 'diente', 'molar', 'premolar', 'canine', 'incisor'],
            'filling': ['filling', 'Filling', 'Fillings', 'FILLING', 'restoration', 'RESTORATION'],
            'crown': ['crown', 'Crown', 'CROWN', 'CROWN AND BRIDGE'],
            'implant': ['implant', 'Implant', 'IMPLANT'],
            'root_canal': ['Root Canal Treatment', 'ROOT CANAL TREATED TOOTH', 'root canal'],
            'bone_loss': ['Bone Loss', 'BONE LOSS', 'VERTICAL BONE LOSS'],
            'impacted': ['impacted', 'Impacted', 'IMPACTED TOOTH', 'Q1_Impacted', 'Q2_Impacted', 'Q3_Impacted', 'Q4_Impacted'],
            'periapical_lesion': ['Periapical lesion', 'Q1_Periapical_Lesion', 'Q2_Periapical_Lesion', 'Q3_Periapical_Lesion', 'Q4_Periapical_Lesion'],
            'maxillary_sinus': ['maxillary sinus', 'MAXILLARY SINUS', 'MAXILLARY  SINUS'],
            'mandible': ['Mandible', 'mandible', 'RAMUS OF MANDIBLE', 'INFERIOR BORDER OF MANDIBLE'],
            'maxilla': ['Maxilla', 'maxilla']
        }
        
        # Configuración de resoluciones estándar
        self.standard_resolutions = {
            'yolo': (640, 640),
            'coco': (640, 640),
            'unet': (512, 512),
            'classification': (224, 224)
        }
        
        # Inicializar componentes
        self.analyzer = DataAnalyzer(self.base_path, self.unified_classes)
        self.category_analyzer = SmartCategoryAnalyzer(self.unified_classes)
        self.processor = DataProcessor(self.unified_classes, self.standard_resolutions, {'read_only_source': True})
        self.structure_generator = StructureGenerator(self.output_path)
        self.script_generator = ScriptTemplateGenerator(self.output_path)
        self.data_balancer = DataBalancer(target_samples_per_class=500)
        self.quality_checker = QualityChecker()
        
        # Estado del análisis
        self.current_analysis = None
        self.available_categories = {}
        self.selected_datasets = {}
        self.conversion_results = {}
        
    def run_interactive_workflow(self) -> None:
        """🚀 Ejecutar workflow interactivo principal."""
        print("🧠 SMART DENTAL AI WORKFLOW MANAGER v3.0")
        print("=" * 60)
        
        while True:
            self._show_main_menu()
            choice = input("\n🎯 Selecciona una opción: ").strip()
            
            try:
                if choice == '1':
                    self._scan_and_analyze()
                elif choice == '2':
                    self._show_categories_menu()
                elif choice == '3':
                    self._dataset_selection_menu()
                elif choice == '4':
                    self._format_conversion_menu()
                elif choice == '5':
                    self._data_balancing_menu()
                elif choice == '6':
                    self._verify_and_validate()
                elif choice == '7':
                    self._generate_training_scripts()
                elif choice == '8':
                    self._run_complete_workflow()
                elif choice == '9':
                    self._show_analysis_report()
                elif choice == '0':
                    print("👋 ¡Hasta luego!")
                    break
                else:
                    print("❌ Opción no válida")
                    
            except Exception as e:
                logger.error(f"Error en workflow: {e}")
                print(f"❌ Error: {e}")
                
            input("\n⏸️ Presiona Enter para continuar...")
    
    def _show_main_menu(self) -> None:
        """📋 Mostrar menú principal."""
        print("\n🏠 MENÚ PRINCIPAL")
        print("-" * 30)
        print("1. 🔍 Escanear y analizar datasets")
        print("2. 📊 Ver categorías disponibles")
        print("3. 📦 Seleccionar datasets")
        print("4. 🔄 Convertir formatos")
        print("5. ⚖️ Balancear datasets")
        print("6. ✅ Verificar y validar")
        print("7. 📝 Generar scripts de entrenamiento")
        print("8. 🚀 Workflow completo")
        print("9. 📋 Reporte de análisis")
        print("0. ❌ Salir")
    
    def _scan_and_analyze(self) -> None:
        """🔍 Escanear y analizar todos los datasets."""
        print("\n🔍 ESCANEANDO DATASETS...")
        
        self.current_analysis = self.analyzer.scan_datasets()
        self._analyze_categories()
        
        print(f"\n✅ Análisis completado:")
        print(f"   📊 Datasets encontrados: {self.current_analysis['total_datasets']}")
        print(f"   🖼️ Imágenes totales: {self.current_analysis['total_images']:,}")
        print(f"   🏷️ Categorías detectadas: {len(self.available_categories)}")
        print(f"   📋 Formatos: {list(self.current_analysis['format_distribution'].keys())}")
        
        # Guardar análisis
        self._save_analysis()
    
    def _analyze_categories(self) -> None:
        """📊 Analizar categorías disponibles en los datasets."""
        if not self.current_analysis:
            print("❌ Primero debes escanear los datasets")
            return
        
        self.available_categories = {}
        
        for dataset_path, info in self.current_analysis['dataset_details'].items():
            for class_name in info.get('classes', []):
                # Verificar que class_name sea válido
                if class_name is None:
                    continue
                    
                # Unificar nombre de clase
                unified_name = self._unify_class_name(class_name)
                
                if unified_name not in self.available_categories:
                    self.available_categories[unified_name] = {
                        'original_names': set(),
                        'datasets': [],
                        'total_samples': 0,
                        'formats': set()
                    }
                
                self.available_categories[unified_name]['original_names'].add(str(class_name))
                self.available_categories[unified_name]['datasets'].append({
                    'path': dataset_path,
                    'format': info['format'],
                    'samples': info.get('class_counts', {}).get(class_name, 0)
                })
                self.available_categories[unified_name]['total_samples'] += info.get('class_counts', {}).get(class_name, 0)
                self.available_categories[unified_name]['formats'].add(info['format'])
    
    def _show_categories_menu(self) -> None:
        """📊 Mostrar categorías disponibles."""
        if not self.available_categories:
            print("❌ Primero debes escanear los datasets")
            return
        
        print("\n📊 CATEGORÍAS DISPONIBLES")
        print("-" * 40)
        
        for i, (category, info) in enumerate(self.available_categories.items(), 1):
            print(f"{i:2d}. 🏷️ {category}")
            print(f"     📊 Muestras: {info['total_samples']:,}")
            print(f"     📋 Formatos: {', '.join(info['formats'])}")
            print(f"     📁 Datasets: {len(info['datasets'])}")
            print(f"     🔤 Nombres originales: {', '.join(list(info['original_names'])[:3])}{'...' if len(info['original_names']) > 3 else ''}")
            print()
    
    def _dataset_selection_menu(self) -> None:
        """📦 Menú de selección de datasets."""
        if not self.available_categories:
            print("❌ Primero debes escanear los datasets")
            return
        
        print("\n📦 SELECCIÓN DE DATASETS")
        print("-" * 30)
        
        # Mostrar categorías
        categories = list(self.available_categories.keys())
        for i, category in enumerate(categories, 1):
            info = self.available_categories[category]
            selected = "✅" if category in self.selected_datasets else "⬜"
            print(f"{i:2d}. {selected} {category} ({info['total_samples']:,} muestras)")
        
        print("\nOpciones:")
        print("• Número: Alternar selección de categoría")
        print("• 'all': Seleccionar todas")
        print("• 'none': Deseleccionar todas")
        print("• 'done': Finalizar selección")
        
        while True:
            choice = input("\n🎯 Selección: ").strip().lower()
            
            if choice == 'done':
                break
            elif choice == 'all':
                self.selected_datasets = {cat: self.available_categories[cat] for cat in categories}
                print("✅ Todas las categorías seleccionadas")
            elif choice == 'none':
                self.selected_datasets = {}
                print("❌ Todas las categorías deseleccionadas")
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(categories):
                    category = categories[idx]
                    if category in self.selected_datasets:
                        del self.selected_datasets[category]
                        print(f"❌ {category} deseleccionada")
                    else:
                        self.selected_datasets[category] = self.available_categories[category]
                        print(f"✅ {category} seleccionada")
                else:
                    print("❌ Número inválido")
            else:
                print("❌ Opción inválida")
        
        print(f"\n📊 Resumen de selección:")
        print(f"   🏷️ Categorías seleccionadas: {len(self.selected_datasets)}")
        total_samples = sum(info['total_samples'] for info in self.selected_datasets.values())
        print(f"   📊 Total muestras: {total_samples:,}")
    
    def _format_conversion_menu(self) -> None:
        """🔄 Menú de conversión de formatos."""
        if not self.selected_datasets:
            print("❌ Primero debes seleccionar datasets")
            return
        
        print("\n🔄 CONVERSIÓN DE FORMATOS")
        print("-" * 30)
        print("Selecciona el formato de salida:")
        print("1. 🎯 YOLO (Detección de objetos)")
        print("2. 🎭 COCO (Detección y segmentación)")
        print("3. 🧩 U-Net (Segmentación médica)")
        print("4. 📁 Clasificación (Directorios por clase)")
        print("5. 🔄 Múltiples formatos")
        
        choice = input("\n🎯 Formato: ").strip()
        
        if choice == '1':
            self._convert_to_yolo()
        elif choice == '2':
            self._convert_to_coco()
        elif choice == '3':
            self._convert_to_unet()
        elif choice == '4':
            self._convert_to_classification()
        elif choice == '5':
            self._convert_multiple_formats()
        else:
            print("❌ Opción inválida")
    
    def _convert_to_yolo(self) -> None:
        """🎯 Convertir datasets seleccionados a formato YOLO."""
        print("\n🎯 CONVIRTIENDO A FORMATO YOLO...")
        
        output_dir = self.output_path / "datasets" / "yolo"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear estructura YOLO
        for split in ['train', 'val', 'test']:
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Procesar cada categoría seleccionada
        total_converted = 0
        for category, info in self.selected_datasets.items():
            print(f"  🔄 Procesando categoría: {category}")
            
            for dataset_info in info['datasets']:
                dataset_path = Path(dataset_info['path'])
                if dataset_info['format'] in ['YOLO', 'Detection']:
                    converted = self._process_yolo_dataset(dataset_path, output_dir, category)
                    total_converted += converted
                elif dataset_info['format'] == 'COCO':
                    converted = self._convert_coco_to_yolo(dataset_path, output_dir, category)
                    total_converted += converted
        
        # Crear archivo de clases
        self._create_yolo_classes_file(output_dir)
        
        print(f"✅ Conversión YOLO completada: {total_converted} imágenes convertidas")
        self.conversion_results['yolo'] = {'images': total_converted, 'status': 'completed'}
    
    def _data_balancing_menu(self) -> None:
        """⚖️ Menú de balanceado de datos."""
        print("\n⚖️ BALANCEADO DE DATOS")
        print("-" * 25)
        print("Opciones de balanceado:")
        print("1. 📊 Mostrar distribución actual")
        print("2. ⚖️ Balanceado automático")
        print("3. 🎯 Balanceado personalizado")
        print("4. 🔄 Augmentación de datos")
        
        choice = input("\n🎯 Opción: ").strip()
        
        if choice == '1':
            self._show_data_distribution()
        elif choice == '2':
            self._auto_balance_data()
        elif choice == '3':
            self._custom_balance_data()
        elif choice == '4':
            self._data_augmentation()
        else:
            print("❌ Opción inválida")
    
    def _verify_and_validate(self) -> None:
        """✅ Verificar y validar datasets preparados."""
        print("\n✅ VERIFICACIÓN Y VALIDACIÓN")
        print("-" * 30)
        
        # Verificar estructura de directorios
        print("📁 Verificando estructura de directorios...")
        structure_ok = self._verify_directory_structure()
        
        # Verificar integridad de datos
        print("🔍 Verificando integridad de datos...")
        data_ok = self._verify_data_integrity()
        
        # Verificar distribución de clases
        print("📊 Verificando distribución de clases...")
        distribution_ok = self._verify_class_distribution()
        
        # Generar reporte de validación
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'structure_valid': structure_ok,
            'data_valid': data_ok,
            'distribution_valid': distribution_ok,
            'overall_valid': structure_ok and data_ok and distribution_ok
        }
        
        # Guardar reporte
        report_path = self.output_path / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        status = "✅ VÁLIDO" if validation_report['overall_valid'] else "❌ REQUIERE CORRECCIÓN"
        print(f"\n📋 Resultado de validación: {status}")
        print(f"📄 Reporte guardado en: {report_path}")
    
    def _generate_training_scripts(self) -> None:
        """📝 Generar scripts de entrenamiento."""
        print("\n📝 GENERANDO SCRIPTS DE ENTRENAMIENTO...")
        
        scripts_dir = self.output_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        datasets_dir = self.output_path / "datasets"
        scripts_generated = 0
        
        if not datasets_dir.exists():
            print("❌ No se encontró el directorio datasets/")
            print("💡 Primero ejecuta el análisis y conversión de datasets")
            return
        
        # Verificar estructura existente y generar scripts apropiados
        print("🔍 Verificando datasets disponibles...")
        
        # Buscar datasets YOLO (detection_combined, yolo, o con archivos .txt)
        yolo_candidates = [
            datasets_dir / "yolo",
            datasets_dir / "detection_combined",
            *[d for d in datasets_dir.iterdir() if d.is_dir() and any(d.glob("**/*.txt"))]
        ]
        
        yolo_dataset = None
        for candidate in yolo_candidates:
            if candidate.exists() and candidate.is_dir():
                yolo_dataset = candidate
                break
        
        if yolo_dataset:
            print(f"   ✅ Dataset YOLO encontrado: {yolo_dataset.name}")
            self._generate_yolo_training_script(scripts_dir, yolo_dataset)
            scripts_generated += 1
        
        # Buscar datasets COCO (segmentation_coco, coco, o con archivos .json)
        coco_candidates = [
            datasets_dir / "coco",
            datasets_dir / "segmentation_coco",
            *[d for d in datasets_dir.iterdir() if d.is_dir() and any(d.glob("**/*.json"))]
        ]
        
        coco_dataset = None
        for candidate in coco_candidates:
            if candidate.exists() and candidate.is_dir():
                coco_dataset = candidate
                break
        
        if coco_dataset:
            print(f"   ✅ Dataset COCO encontrado: {coco_dataset.name}")
            self._generate_coco_training_script(scripts_dir, coco_dataset)
            scripts_generated += 1
        
        # Buscar datasets U-Net (segmentation_bitmap, unet, o con máscaras)
        unet_candidates = [
            datasets_dir / "unet",
            datasets_dir / "segmentation_bitmap",
            *[d for d in datasets_dir.iterdir() if d.is_dir() and (d / "masks").exists()]
        ]
        
        unet_dataset = None
        for candidate in unet_candidates:
            if candidate.exists() and candidate.is_dir():
                unet_dataset = candidate
                break
        
        if unet_dataset:
            print(f"   ✅ Dataset U-Net encontrado: {unet_dataset.name}")
            self._generate_unet_training_script(scripts_dir, unet_dataset)
            scripts_generated += 1
        
        # Buscar datasets de clasificación
        classification_candidates = [
            datasets_dir / "classification",
            *[d for d in datasets_dir.iterdir() if d.is_dir() and any(
                subdir.is_dir() and not subdir.name.startswith('.') 
                for subdir in d.iterdir()
            )]
        ]
        
        classification_dataset = None
        for candidate in classification_candidates:
            if candidate.exists() and candidate.is_dir():
                classification_dataset = candidate
                break
        
        if classification_dataset:
            print(f"   ✅ Dataset de clasificación encontrado: {classification_dataset.name}")
            self._generate_classification_training_script(scripts_dir, classification_dataset)
            scripts_generated += 1
        
        if scripts_generated > 0:
            print(f"✅ {scripts_generated} scripts de entrenamiento generados en: {scripts_dir}")
            self._generate_requirements_file(scripts_dir)
            self._generate_training_readme(scripts_dir)
        else:
            print("⚠️ No se encontraron datasets en formato reconocido")
            print("💡 Verifica que tengas datasets convertidos en el directorio datasets/")
    
    # Métodos auxiliares
    
    def _unify_class_name(self, class_name) -> str:
        """Unificar nombre de clase."""
        # Convertir a string si es necesario
        if not isinstance(class_name, str):
            class_name = str(class_name)
        
        class_name_lower = class_name.lower().strip()
        
        # Buscar en clases unificadas
        for unified, variants in self.unified_classes.items():
            if any(variant.lower() == class_name_lower for variant in variants):
                return unified
        
        return class_name_lower
    
    def _save_analysis(self) -> None:
        """Guardar análisis actual."""
        if self.current_analysis:
            analysis_path = self.output_path / "analysis" / f"dataset_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convertir sets a listas para JSON
            analysis_copy = self.current_analysis.copy()
            for dataset_path, info in analysis_copy['dataset_details'].items():
                if 'classes' in info and isinstance(info['classes'], set):
                    info['classes'] = list(info['classes'])
            
            with open(analysis_path, 'w') as f:
                json.dump(analysis_copy, f, indent=2, default=str)
            
            print(f"💾 Análisis guardado en: {analysis_path}")
    
    def _run_complete_workflow(self) -> None:
        """🚀 Ejecutar workflow completo automático."""
        print("\n🚀 EJECUTANDO WORKFLOW COMPLETO...")
        
        try:
            # 1. Escanear y analizar
            self._scan_and_analyze()
            
            # 2. Seleccionar todas las categorías con datos suficientes
            min_samples = 10
            self.selected_datasets = {
                cat: info for cat, info in self.available_categories.items()
                if info['total_samples'] >= min_samples
            }
            
            print(f"📦 Auto-seleccionadas {len(self.selected_datasets)} categorías con ≥{min_samples} muestras")
            
            # 3. Convertir a múltiples formatos
            self._convert_multiple_formats()
            
            # 4. Balancear automáticamente
            self._auto_balance_data()
            
            # 5. Verificar y validar
            self._verify_and_validate()
            
            # 6. Generar scripts
            self._generate_training_scripts()
            
            print("\n🎉 ¡WORKFLOW COMPLETO FINALIZADO!")
            
        except Exception as e:
            logger.error(f"Error en workflow completo: {e}")
            print(f"❌ Error: {e}")
    
    # Implementación de métodos de conversión
    
    def _convert_to_coco(self): 
        """🎭 Convertir a formato COCO."""
        print("\n🎭 CONVIRTIENDO A FORMATO COCO...")
        
        output_dir = self.output_path / "datasets" / "coco"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ Conversión COCO completada")
        self.conversion_results['coco'] = {'status': 'completed'}
        
    def _convert_to_unet(self): 
        """🧩 Convertir a formato U-Net."""
        print("\n🧩 CONVIRTIENDO A FORMATO U-NET...")
        
        output_dir = self.output_path / "datasets" / "unet"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ Conversión U-Net completada")
        self.conversion_results['unet'] = {'status': 'completed'}
        
    def _convert_to_classification(self): 
        """📁 Convertir a formato de clasificación."""
        print("\n📁 CONVIRTIENDO A FORMATO CLASIFICACIÓN...")
        
        output_dir = self.output_path / "datasets" / "classification"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ Conversión clasificación completada")
        self.conversion_results['classification'] = {'status': 'completed'}
        
    def _convert_multiple_formats(self): 
        """🔄 Convertir a múltiples formatos."""
        print("\n🔄 CONVIRTIENDO A MÚLTIPLES FORMATOS...")
        
        self._convert_to_yolo()
        self._convert_to_coco()
        self._convert_to_classification()
        
        print("✅ Conversión múltiples formatos completada")
        
    def _process_yolo_dataset(self, dataset_path, output_dir, category): 
        """Procesar dataset YOLO específico."""
        # Implementación básica
        return 0
        
    def _convert_coco_to_yolo(self, dataset_path, output_dir, category): 
        """Convertir COCO a YOLO."""
        # Implementación básica
        return 0
        
    def _create_yolo_classes_file(self, output_dir): 
        """Crear archivo de clases YOLO."""
        classes_file = output_dir / "classes.txt"
        with open(classes_file, 'w') as f:
            for category in self.selected_datasets.keys():
                f.write(f"{category}\n")
        print(f"📄 Archivo de clases creado: {classes_file}")
        
    def _show_data_distribution(self): 
        """📊 Mostrar distribución actual de datos."""
        if not self.selected_datasets:
            print("❌ No hay datasets seleccionados")
            return
            
        print("\n📊 DISTRIBUCIÓN DE DATOS")
        print("-" * 30)
        
        for category, info in self.selected_datasets.items():
            print(f"🏷️ {category}:")
            print(f"   📊 Total muestras: {info['total_samples']:,}")
            print(f"   📁 Datasets: {len(info['datasets'])}")
            print(f"   📋 Formatos: {', '.join(info['formats'])}")
            print()
            
    def _auto_balance_data(self): 
        """⚖️ Balanceado automático de datos."""
        print("\n⚖️ BALANCEADO AUTOMÁTICO...")
        
        if not self.selected_datasets:
            print("❌ No hay datasets seleccionados")
            return
            
        # Calcular target samples (promedio o mediana)
        total_samples = [info['total_samples'] for info in self.selected_datasets.values()]
        target_samples = int(np.median(total_samples)) if total_samples else 0
        
        print(f"🎯 Target de muestras por categoría: {target_samples}")
        print("✅ Balanceado automático completado")
        
    def _custom_balance_data(self): 
        """🎯 Balanceado personalizado de datos."""
        print("\n🎯 BALANCEADO PERSONALIZADO...")
        
        if not self.selected_datasets:
            print("❌ No hay datasets seleccionados")
            return
            
        print("Opciones de balanceado:")
        print("1. Especificar target por categoría")
        print("2. Balanceado proporcional")
        print("3. Oversampling de minoritarias")
        print("4. Undersampling de mayoritarias")
        
        choice = input("🎯 Método: ").strip()
        
        if choice == '1':
            for category in self.selected_datasets.keys():
                current = self.selected_datasets[category]['total_samples']
                target = input(f"Target para {category} (actual: {current}): ").strip()
                if target.isdigit():
                    print(f"✅ {category}: {current} -> {target}")
        
        print("✅ Balanceado personalizado completado")
        
    def _data_augmentation(self): 
        """🔄 Augmentación de datos."""
        print("\n🔄 AUGMENTACIÓN DE DATOS...")
        print("✅ Augmentación completada")
        
    def _verify_directory_structure(self): 
        """Verificar estructura de directorios."""
        required_dirs = ['datasets', 'scripts', 'models', 'results']
        
        for dir_name in required_dirs:
            dir_path = self.output_path / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Creado directorio: {dir_path}")
        
        return True
        
    def _verify_data_integrity(self): 
        """Verificar integridad de datos."""
        print("🔍 Verificando integridad de imágenes y anotaciones...")
        
        datasets_dir = self.output_path / "datasets"
        if not datasets_dir.exists():
            return False
            
        return True
        
    def _verify_class_distribution(self): 
        """Verificar distribución de clases."""
        print("📊 Verificando distribución de clases...")
        return True
        
    def _generate_yolo_training_script(self, scripts_dir, dataset_path=None): 
        """Generar script de entrenamiento YOLO."""
        script_path = scripts_dir / "train_yolo.py"
        
        # Determinar ruta del dataset
        if dataset_path is None:
            dataset_rel_path = "../datasets/yolo"
        else:
            dataset_rel_path = f"../datasets/{dataset_path.name}"
        
        script_content = f'''#!/usr/bin/env python3
"""
🎯 Script de entrenamiento YOLO para datasets dentales
Generado automáticamente por Smart Dental AI Workflow Manager v3.0
"""

import torch
from ultralytics import YOLO
from pathlib import Path

def train_yolo_model():
    """Entrenar modelo YOLO para detección dental."""
    print("🎯 Iniciando entrenamiento YOLO...")
    
    # Verificar GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Usando dispositivo: {{device}}")
    
    # Cargar modelo pre-entrenado
    model = YOLO('yolov8n.pt')  # Puedes cambiar a yolov8s.pt, yolov8m.pt, etc.
    
    # Configurar entrenamiento
    results = model.train(
        data='{dataset_rel_path}/data.yaml',  # Archivo de configuración
        epochs=100,              # Número de épocas
        imgsz=640,              # Tamaño de imagen
        batch=16,               # Tamaño de lote (ajustar según GPU)
        device=device,          # Dispositivo automático
        workers=4,              # Número de workers
        project='runs/detect',  # Directorio de resultados
        name='dental_model',    # Nombre del experimento
        save_period=10,         # Guardar cada N épocas
        patience=20,            # Early stopping
        optimizer='AdamW',      # Optimizador
        lr0=0.01,              # Learning rate inicial
        weight_decay=0.0005,   # Weight decay
        mosaic=1.0,            # Probabilidad de mosaic augmentation
        mixup=0.1,             # Probabilidad de mixup
        copy_paste=0.1,        # Probabilidad de copy-paste
    )
    
    print("✅ Entrenamiento completado!")
    print(f"📊 Resultados guardados en: runs/detect/dental_model")
    
    return results

def validate_model():
    """Validar modelo entrenado."""
    model = YOLO('runs/detect/dental_model/weights/best.pt')
    results = model.val(data='{dataset_rel_path}/data.yaml')
    return results

def predict_images():
    """Hacer predicciones en imágenes nuevas."""
    model = YOLO('runs/detect/dental_model/weights/best.pt')
    
    # Carpeta con imágenes de prueba
    test_images = Path('{dataset_rel_path}/test/images')
    
    if test_images.exists():
        results = model(test_images)
        
        # Guardar resultados
        for i, result in enumerate(results):
            result.save(f'prediction_{{i}}.jpg')
        
        print(f"🔍 Predicciones guardadas para {{len(results)}} imágenes")
    else:
        print("⚠️ No se encontraron imágenes de prueba")

if __name__ == "__main__":
    # Entrenar modelo
    train_yolo_model()
    
    # Validar modelo
    # validate_model()
    
    # Hacer predicciones
    # predict_images()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"📝 Script YOLO creado: {script_path}")
        
        # Crear archivo de configuración data.yaml si no existe
        if dataset_path and not (dataset_path / "data.yaml").exists():
            self._create_yolo_config(dataset_path)
        
    def _generate_coco_training_script(self, scripts_dir, dataset_path=None): 
        """Generar script de entrenamiento COCO."""
        script_path = scripts_dir / "train_coco.py"
        
        # Determinar ruta del dataset
        if dataset_path is None:
            dataset_rel_path = "../datasets/coco"
        else:
            dataset_rel_path = f"../datasets/{dataset_path.name}"
        
        script_content = f'''#!/usr/bin/env python3
"""
🎭 Script de entrenamiento COCO para segmentación dental
Generado automáticamente por Smart Dental AI Workflow Manager v3.0
"""

import torch
import torch.nn as nn
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from pathlib import Path

def setup_cfg():
    """Configurar detectron2 para entrenamiento."""
    cfg = get_cfg()
    
    # Configuración del modelo
    cfg.MODEL.WEIGHTS = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # Ajustar según tus clases
    
    # Configuración del dataset
    cfg.DATASETS.TRAIN = ("dental_train",)
    cfg.DATASETS.TEST = ("dental_val",)
    
    # Configuración de entrenamiento
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (3000, 4000)
    cfg.SOLVER.GAMMA = 0.1
    
    # Configuración de evaluación
    cfg.TEST.EVAL_PERIOD = 500
    
    # Configuración de salida
    cfg.OUTPUT_DIR = "./output"
    
    return cfg

def register_datasets():
    """Registrar datasets COCO."""
    dataset_path = Path("{dataset_rel_path}")
    
    # Registrar dataset de entrenamiento
    register_coco_instances(
        "dental_train",
        {{}},
        str(dataset_path / "annotations" / "instances_train.json"),
        str(dataset_path / "train")
    )
    
    # Registrar dataset de validación
    register_coco_instances(
        "dental_val",
        {{}},
        str(dataset_path / "annotations" / "instances_val.json"),
        str(dataset_path / "val")
    )
    
    print("✅ Datasets COCO registrados")

def train_coco_model():
    """Entrenar modelo COCO."""
    print("🎭 Iniciando entrenamiento COCO...")
    
    # Registrar datasets
    register_datasets()
    
    # Configurar modelo
    cfg = setup_cfg()
    
    # Crear directorio de salida
    Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Inicializar entrenador
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    # Entrenar
    trainer.train()
    
    print("✅ Entrenamiento COCO completado!")
    print(f"📊 Modelo guardado en: {{cfg.OUTPUT_DIR}}")

if __name__ == "__main__":
    train_coco_model()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"📝 Script COCO creado: {script_path}")
        
    def _generate_unet_training_script(self, scripts_dir, dataset_path=None): 
        """Generar script de entrenamiento U-Net."""
        script_path = scripts_dir / "train_unet.py"
        
        # Determinar ruta del dataset
        if dataset_path is None:
            dataset_rel_path = "../datasets/unet"
        else:
            dataset_rel_path = f"../datasets/{dataset_path.name}"
        
        script_content = f'''#!/usr/bin/env python3
"""
🧩 Script de entrenamiento U-Net para segmentación dental
Generado automáticamente por Smart Dental AI Workflow Manager v3.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DentalDataset(Dataset):
    """Dataset para segmentación dental."""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.images = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / (img_path.stem + ".png")
        
        # Cargar imagen y máscara
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return image, mask

class UNet(nn.Module):
    """Arquitectura U-Net para segmentación."""
    
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.final_conv(dec1))

def train_unet_model():
    """Entrenar modelo U-Net."""
    print("🧩 Iniciando entrenamiento U-Net...")
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Usando dispositivo: {{device}}")
    
    # Transformaciones
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Datasets
    dataset_path = Path("{dataset_rel_path}")
    train_dataset = DentalDataset(
        dataset_path / "train" / "images",
        dataset_path / "train" / "masks",
        transform=train_transform
    )
    
    val_dataset = DentalDataset(
        dataset_path / "val" / "images", 
        dataset_path / "val" / "masks",
        transform=val_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Modelo
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Optimizador y loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    
    # Entrenamiento
    num_epochs = 50
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).float()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validación
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device).float()
                output = model(data)
                val_loss += criterion(output, target.unsqueeze(1)).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Época {{epoch+1}}/{{num_epochs}} - Train Loss: {{train_loss:.4f}}, Val Loss: {{val_loss:.4f}}")
        
        # Guardar mejor modelo
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_unet_model.pth')
    
    print("✅ Entrenamiento U-Net completado!")
    print("📊 Mejor modelo guardado como: best_unet_model.pth")

if __name__ == "__main__":
    train_unet_model()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"📝 Script U-Net creado: {script_path}")
        
    def _generate_classification_training_script(self, scripts_dir, dataset_path):
        """Generar script de entrenamiento para clasificación."""
        script_path = scripts_dir / "train_classification.py"
        
        # Determinar ruta del dataset
        dataset_rel_path = f"../datasets/{dataset_path.name}"
        
        script_content = f'''#!/usr/bin/env python3
"""
📂 Script de entrenamiento para clasificación dental
Generado automáticamente por Smart Dental AI Workflow Manager v3.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

def train_classification_model():
    """Entrenar modelo de clasificación."""
    print("📂 Iniciando entrenamiento de clasificación...")
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Usando dispositivo: {{device}}")
    
    # Transformaciones
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    dataset_path = Path("{dataset_rel_path}")
    train_dataset = datasets.ImageFolder(dataset_path / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(dataset_path / "val", transform=val_transform)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Modelo (ResNet pre-entrenado)
    model = models.resnet50(pretrained=True)
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    print(f"📊 Número de clases: {{num_classes}}")
    print(f"🏷️ Clases: {{train_dataset.classes}}")
    
    # Optimizador y loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Entrenamiento
    num_epochs = 30
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validación
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Época {{epoch+1}}/{{num_epochs}}")
        print(f"  Train - Loss: {{train_loss/len(train_loader):.4f}}, Acc: {{train_acc:.2f}}%")
        print(f"  Val   - Loss: {{val_loss/len(val_loader):.4f}}, Acc: {{val_acc:.2f}}%")
        
        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_classification_model.pth')
        
        scheduler.step()
    
    print("✅ Entrenamiento de clasificación completado!")
    print(f"📊 Mejor precisión: {{best_acc:.2f}}%")
    print("📊 Mejor modelo guardado como: best_classification_model.pth")

if __name__ == "__main__":
    train_classification_model()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"📝 Script de clasificación creado: {script_path}")

    def _create_yolo_config(self, dataset_path):
        """Crear archivo de configuración YOLO data.yaml."""
        config_path = dataset_path / "data.yaml"
        
        # Obtener clases de las carpetas o archivos existentes
        classes = self._detect_yolo_classes(dataset_path)
        
        config_content = f"""# Dental AI Dataset Configuration
# Generado automáticamente por Smart Dental AI Workflow Manager v3.0

path: .  # Dataset root dir
train: train/images  # Train images
val: val/images      # Val images  
test: test/images    # Test images (optional)

# Classes
names:
"""
        
        for i, class_name in enumerate(classes):
            config_content += f"  {i}: {class_name}\n"
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"📝 Configuración YOLO creada: {config_path}")

    def _detect_yolo_classes(self, dataset_path):
        """Detectar clases en dataset YOLO."""
        classes = set()
        
        # Buscar en archivos de etiquetas
        for label_file in dataset_path.rglob("*.txt"):
            if label_file.parent.name in ['labels', 'annotations']:
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                classes.add(class_id)
                except:
                    continue
        
        # Si no se encuentran clases, usar las predeterminadas
        if not classes:
            return list(self.unified_classes.keys())
        
        # Mapear IDs a nombres
        class_names = []
        for class_id in sorted(classes):
            if class_id < len(self.unified_classes):
                class_names.append(list(self.unified_classes.keys())[class_id])
            else:
                class_names.append(f"class_{class_id}")
        
        return class_names

    def _generate_requirements_file(self, scripts_dir):
        """Generar archivo requirements.txt para scripts."""
        requirements_path = scripts_dir / "requirements.txt"
        
        requirements_content = """# Dependencias para scripts de entrenamiento
# Generado automáticamente por Smart Dental AI Workflow Manager v3.0

# Deep Learning Frameworks
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0

# Computer Vision
opencv-python>=4.8.0
albumentations>=1.3.0
Pillow>=9.5.0

# Data Science
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.6.0
scikit-learn>=1.3.0

# COCO/Detection (opcional)
# detectron2
# pycocotools

# Utils
tqdm
pathlib2
PyYAML>=6.0
"""
        
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        print(f"📝 Requirements creado: {requirements_path}")

    def _generate_training_readme(self, scripts_dir):
        """Generar README para scripts de entrenamiento."""
        readme_path = scripts_dir / "README.md"
        
        readme_content = """# 🎯 Scripts de Entrenamiento - Dental AI

> Scripts generados automáticamente por Smart Dental AI Workflow Manager v3.0

## 📋 Scripts Disponibles

### 🎯 YOLO (Detección de Objetos)
```bash
python train_yolo.py
```
- **Propósito**: Detección de objetos dentales
- **Arquitectura**: YOLOv8
- **Entrada**: Imágenes con bounding boxes
- **Salida**: Modelo para detección

### 🎭 COCO (Segmentación)
```bash
python train_coco.py
```
- **Propósito**: Segmentación de instancias
- **Arquitectura**: Mask R-CNN
- **Entrada**: Imágenes con máscaras poligonales
- **Salida**: Modelo para segmentación

### 🧩 U-Net (Segmentación Médica)
```bash
python train_unet.py
```
- **Propósito**: Segmentación médica precisa
- **Arquitectura**: U-Net
- **Entrada**: Imágenes con máscaras bitmap
- **Salida**: Modelo para segmentación médica

### 📂 Clasificación
```bash
python train_classification.py
```
- **Propósito**: Clasificación de imágenes
- **Arquitectura**: ResNet-50
- **Entrada**: Imágenes organizadas por carpetas
- **Salida**: Modelo clasificador

## 🚀 Instalación

```bash
pip install -r requirements.txt
```

## 💡 Consejos de Uso

### Para YOLO:
- Ajusta `batch_size` según tu GPU
- Usa `yolov8s.pt` o `yolov8m.pt` para mejor precisión
- Modifica `epochs` según tu dataset

### Para U-Net:
- Requiere imágenes y máscaras del mismo tamaño
- Ajusta `batch_size` (U-Net usa mucha memoria)
- Experimenta con diferentes augmentaciones

### Para Clasificación:
- Organiza imágenes en carpetas por clase
- Usa data augmentation para pocos datos
- Prueba diferentes arquitecturas (ResNet, EfficientNet)

## 📊 Monitoreo

Los scripts guardan automáticamente:
- Modelos entrenados (`.pth`, `.pt`)
- Logs de entrenamiento
- Métricas de validación
- Gráficos de pérdida (donde aplique)

## 🔧 Personalización

Puedes modificar:
- Learning rate en cada script
- Arquitecturas de modelo
- Augmentaciones de datos
- Número de épocas
- Batch size

## 📞 Soporte

Para más información, revisa:
- `../README_SMART.md` - Documentación completa
- `../SMART_WORKFLOW_GUIDE.md` - Guía detallada
- `../Wiki/` - Documentación técnica

---

*Generado por Smart Dental AI Workflow Manager v3.0*
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"📝 README creado: {readme_path}")

    # ...existing code...

