"""
🎛️ Main Workflow Manager Module
Módulo principal que orquesta todo el flujo de trabajo
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from .data_analyzer import DataAnalyzer
from .data_processor import DataProcessor
from .structure_generator import StructureGenerator
from .script_templates import ScriptGenerator


class DentalDataWorkflowManager:
    """
    🎛️ Gestor principal de flujo de trabajo para datasets dentales
    Orquesta el análisis, procesamiento y preparación de datos
    """
    
    def __init__(self, base_path: str = None, output_path: str = None):
        # Configurar rutas
        self.base_path = Path(base_path) if base_path else Path("_dataSets")
        
        # Cambiar a Dist/dental_ai como salida final
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = Path("Dist") / "dental_ai"
        
        # Crear directorio de salida si no existe
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configuración de clases unificadas
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
            'coco': (1024, 1024),
            'unet': (512, 512)
        }
        
        # Configuración de seguridad
        self.safety_config = {
            'backup_enabled': True,
            'read_only_source': True,
            'verify_copy': True,
            'preserve_original_structure': True
        }
        
        # Configuración del workflow
        self.workflow_config = {
            'train_ratio': 0.7,
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            'min_samples_per_class': 10,
            'max_augmentation_factor': 5,
            'class_balance_threshold': 0.1
        }
        
        # Estructura dental-ai
        self.dental_ai_structure = {
            'datasets': {
                'detection_combined': 'YOLO format fusionados',
                'segmentation_coco': 'COCO format unificado', 
                'segmentation_bitmap': 'Máscaras para U-Net',
                'classification': 'Clasificación por carpetas'
            },
            'models': {
                'yolo_detect': 'Modelos YOLO detección',
                'yolo_segment': 'Modelos YOLO segmentación',
                'unet_teeth': 'Modelos U-Net dientes',
                'cnn_classifier': 'Clasificadores CNN'
            },
            'training': {
                'scripts': 'Scripts de entrenamiento',
                'configs': 'Configuraciones de entrenamiento',
                'logs': 'Logs de entrenamiento'
            },
            'api': {
                'main.py': 'API principal',
                'models': 'Modelos para la API',
                'utils': 'Utilidades'
            },
            'docs': 'Documentación'
        }
        
        # Inicializar módulos
        self.analyzer = DataAnalyzer(self.base_path, self.unified_classes)
        self.processor = DataProcessor(self.unified_classes, self.standard_resolutions, self.safety_config)
        self.structure_generator = StructureGenerator(self.dental_ai_structure)
        self.script_generator = ScriptGenerator()
        
        # Logging
        self.log_entries = []
    
    def log_message(self, message: str, file_path: str = None):
        """Función de logging con timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        self.log_entries.append(formatted_message)
        
        if file_path:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(formatted_message + "\n")
    
    def scan_and_analyze_datasets(self) -> Dict[str, Any]:
        """🔍 Escanea y analiza todos los datasets disponibles."""
        self.log_message("🔍 Iniciando escaneo y análisis de datasets...")
        
        # Realizar análisis
        analysis = self.analyzer.scan_datasets()
        
        # Generar reportes
        self.analyzer.generate_analysis_report(analysis, self.output_path)
        self.analyzer.create_summary_table(analysis, self.output_path)
        self.analyzer.create_visualizations(analysis, self.output_path)
        
        self.log_message(f"✅ Análisis completado. Datasets encontrados: {analysis['total_datasets']}")
        return analysis
    
    def create_dental_ai_structure(self):
        """🏗️ Crea la estructura completa de dental-ai."""
        self.log_message(f"🏗️ Creando estructura dental-ai en: {self.output_path}")
        
        self.structure_generator.create_structure(self.output_path)
        self.structure_generator.create_documentation(self.output_path)
        
        self.log_message("✅ Estructura dental-ai creada exitosamente")
    
    def create_training_scripts(self):
        """📝 Genera scripts de entrenamiento para todos los formatos."""
        self.log_message("📝 Generando scripts de entrenamiento...")
        
        training_path = self.output_path / "training"
        
        # Generar scripts para cada tipo de modelo
        self.script_generator.create_yolo_training_script(training_path)
        self.script_generator.create_unet_training_script(training_path)
        self.script_generator.create_classification_script(training_path)
        
        self.log_message("✅ Scripts de entrenamiento generados")
    
    def create_api_template(self):
        """🌐 Crea template de API para inferencia."""
        self.log_message("🌐 Creando template de API...")
        
        api_path = self.output_path / "api"
        self.script_generator.create_api_template(api_path)
        
        self.log_message("✅ Template de API creado")
    
    def merge_yolo_datasets(self, dataset_paths: List[str] = None) -> Dict[str, Any]:
        """🔄 Fusiona datasets YOLO."""
        if dataset_paths is None:
            # Buscar automáticamente datasets YOLO
            analysis = self.analyzer.scan_datasets()
            dataset_paths = analysis.get('yolo_datasets', [])
        
        if not dataset_paths:
            self.log_message("⚠️ No se encontraron datasets YOLO para fusionar")
            return {}
        
        self.log_message(f"🔄 Fusionando {len(dataset_paths)} datasets YOLO...")
        
        stats = self.processor.merge_yolo_datasets(dataset_paths, self.output_path)
        
        self.log_message(f"✅ Fusión YOLO completada. Imágenes procesadas: {stats['total_images']}")
        return stats
    
    def merge_coco_datasets(self, dataset_paths: List[str] = None) -> Dict[str, Any]:
        """🔄 Fusiona datasets COCO."""
        if dataset_paths is None:
            analysis = self.analyzer.scan_datasets()
            dataset_paths = analysis.get('coco_datasets', [])
        
        if not dataset_paths:
            self.log_message("⚠️ No se encontraron datasets COCO para fusionar")
            return {}
        
        self.log_message(f"🔄 Fusionando {len(dataset_paths)} datasets COCO...")
        
        stats = self.processor.merge_coco_datasets(dataset_paths, self.output_path)
        
        self.log_message(f"✅ Fusión COCO completada. Imágenes procesadas: {stats['total_images']}")
        return stats
    
    def create_classification_dataset(self, dataset_paths: List[str] = None) -> Dict[str, Any]:
        """📁 Crea dataset de clasificación."""
        if dataset_paths is None:
            analysis = self.analyzer.scan_datasets()
            dataset_paths = analysis.get('pure_image_datasets', [])
        
        if not dataset_paths:
            self.log_message("⚠️ No se encontraron datasets de imágenes para clasificación")
            return {}
        
        self.log_message(f"📁 Creando dataset de clasificación...")
        
        stats = self.processor.create_classification_dataset(dataset_paths, self.output_path)
        
        self.log_message(f"✅ Dataset de clasificación creado. Imágenes procesadas: {stats['total_images']}")
        return stats
    
    def run_complete_workflow(self):
        """🚀 Ejecuta el workflow completo."""
        self.log_message("🚀 INICIANDO WORKFLOW COMPLETO...")
        
        try:
            # 1. Crear estructura
            self.create_dental_ai_structure()
            
            # 2. Analizar datasets
            analysis = self.scan_and_analyze_datasets()
            
            # 3. Procesar datasets
            yolo_stats = self.merge_yolo_datasets()
            coco_stats = self.merge_coco_datasets()
            classification_stats = self.create_classification_dataset()
            
            # 4. Generar scripts de entrenamiento
            self.create_training_scripts()
            
            # 5. Crear API template
            self.create_api_template()
            
            # 6. Generar reporte final
            final_report = {
                'workflow_completed': True,
                'timestamp': datetime.now().isoformat(),
                'output_path': str(self.output_path),
                'analysis': analysis,
                'processing_stats': {
                    'yolo': yolo_stats,
                    'coco': coco_stats,
                    'classification': classification_stats
                },
                'log_entries': self.log_entries
            }
            
            # Guardar reporte final
            with open(self.output_path / 'workflow_report.json', 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
            
            self.log_message("🎉 WORKFLOW COMPLETADO EXITOSAMENTE!")
            print(f"\n🎯 RESUMEN FINAL:")
            print(f"   📂 Salida: {self.output_path}")
            print(f"   📊 Datasets analizados: {analysis.get('total_datasets', 0)}")
            print(f"   🔄 YOLO procesadas: {yolo_stats.get('total_images', 0)}")
            print(f"   🔄 COCO procesadas: {coco_stats.get('total_images', 0)}")
            print(f"   📁 Clasificación procesadas: {classification_stats.get('total_images', 0)}")
            print(f"   📝 Scripts de entrenamiento: ✅")
            print(f"   🌐 API template: ✅")
            
        except Exception as e:
            self.log_message(f"❌ Error en workflow: {e}")
            raise
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """📊 Obtiene estadísticas completas de los datasets."""
        analysis_file = self.output_path / "dental_dataset_analysis.json"
        
        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self.scan_and_analyze_datasets()
    
    def validate_output_structure(self) -> bool:
        """✅ Valida que la estructura de salida esté completa."""
        required_dirs = [
            'datasets', 'models', 'training', 'api', 'docs'
        ]
        
        for dir_name in required_dirs:
            if not (self.output_path / dir_name).exists():
                return False
        
        return True
