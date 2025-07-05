#!/usr/bin/env python3
"""
🎯 Escáner Específico de Dataset
================================

Analiza una carpeta específica para determinar su tipo y estructura.
Útil para datasets muy grandes o para análisis detallado.

Author: Anton Sychev
Created: 2025-07-05
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent))
from auto_organize_datasets import SmartDatasetOrganizer


class SpecificDatasetScanner:
    """🎯 Escáner para un dataset específico"""
    
    def __init__(self):
        self.organizer = SmartDatasetOrganizer()
        self.supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def analyze_dataset(self, dataset_path: str) -> Dict:
        """🔍 Analiza un dataset específico en detalle"""
        path = Path(dataset_path)
        
        if not path.exists():
            return {"error": f"La ruta {dataset_path} no existe"}
        
        if not path.is_dir():
            return {"error": f"La ruta {dataset_path} no es una carpeta"}
        
        print(f"🔍 ANALIZANDO DATASET: {path.name}")
        print("="*50)
        
        # Análisis básico
        dataset_type, info = self.organizer.detect_dataset_type(path)
        
        # Análisis detallado adicional
        detailed_info = self._detailed_analysis(path)
        
        # Combinar información
        result = {
            "path": str(path),
            "name": path.name,
            "type": dataset_type,
            "basic_info": info,
            "detailed_info": detailed_info,
            "recommendations": self._get_recommendations(dataset_type, info, detailed_info)
        }
        
        return result
    
    def _detailed_analysis(self, path: Path) -> Dict:
        """📊 Análisis detallado del dataset"""
        print(f"\n📊 ANÁLISIS DETALLADO...")
        
        # Estructura de archivos
        all_files = list(path.rglob("*"))
        directories = [f for f in all_files if f.is_dir()]
        files = [f for f in all_files if f.is_file()]
        
        # Extensiones de archivos
        extensions = {}
        for file in files:
            ext = file.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        # Distribución por subdirectorios
        subdir_distribution = {}
        for directory in directories:
            if directory != path:  # Excluir el directorio raíz
                relative_path = directory.relative_to(path)
                image_count = len([f for f in directory.rglob("*") 
                                 if f.suffix.lower() in self.supported_image_extensions])
                if image_count > 0:
                    subdir_distribution[str(relative_path)] = image_count
        
        # Tamaños de archivos
        file_sizes = []
        total_size = 0
        for file in files:
            try:
                size = file.stat().st_size
                file_sizes.append(size)
                total_size += size
            except:
                continue
        
        avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
        max_file_size = max(file_sizes) if file_sizes else 0
        min_file_size = min(file_sizes) if file_sizes else 0
        
        # Análisis de imágenes
        image_analysis = self._analyze_images(path)
        
        detailed = {
            "total_directories": len(directories),
            "total_files": len(files),
            "extensions": extensions,
            "subdir_distribution": subdir_distribution,
            "size_analysis": {
                "total_size_mb": total_size / (1024 * 1024),
                "avg_file_size_kb": avg_file_size / 1024,
                "max_file_size_mb": max_file_size / (1024 * 1024),
                "min_file_size_bytes": min_file_size
            },
            "image_analysis": image_analysis
        }
        
        # Imprimir información detallada
        print(f"   📁 Directorios: {detailed['total_directories']}")
        print(f"   📄 Archivos: {detailed['total_files']}")
        print(f"   💾 Tamaño total: {detailed['size_analysis']['total_size_mb']:.1f} MB")
        print(f"   📊 Extensiones más comunes:")
        
        sorted_extensions = sorted(extensions.items(), key=lambda x: x[1], reverse=True)
        for ext, count in sorted_extensions[:5]:
            print(f"      {ext or '(sin ext)'}: {count} archivos")
        
        if subdir_distribution:
            print(f"   📂 Distribución por subcarpetas:")
            for subdir, count in list(subdir_distribution.items())[:5]:
                print(f"      {subdir}: {count} imágenes")
        
        return detailed
    
    def _analyze_images(self, path: Path) -> Dict:
        """🖼️ Análisis específico de imágenes"""
        image_files = []
        for ext in self.supported_image_extensions:
            image_files.extend(path.rglob(f"*{ext}"))
        
        if not image_files:
            return {"error": "No se encontraron imágenes"}
        
        # Analizar una muestra de imágenes para obtener dimensiones
        sample_size = min(20, len(image_files))
        dimensions = []
        
        try:
            import cv2
            for img_file in image_files[:sample_size]:
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        h, w = img.shape[:2]
                        dimensions.append((w, h))
                except:
                    continue
        except ImportError:
            # Si OpenCV no está disponible, usar PIL
            try:
                from PIL import Image
                for img_file in image_files[:sample_size]:
                    try:
                        with Image.open(img_file) as img:
                            dimensions.append(img.size)
                    except:
                        continue
            except ImportError:
                return {"error": "No se puede analizar imágenes (falta OpenCV o PIL)"}
        
        if dimensions:
            widths = [d[0] for d in dimensions]
            heights = [d[1] for d in dimensions]
            
            analysis = {
                "total_images": len(image_files),
                "sample_analyzed": len(dimensions),
                "avg_width": sum(widths) / len(widths),
                "avg_height": sum(heights) / len(heights),
                "max_width": max(widths),
                "max_height": max(heights),
                "min_width": min(widths),
                "min_height": min(heights),
                "common_resolutions": self._find_common_resolutions(dimensions)
            }
            
            print(f"   🖼️ Análisis de imágenes ({analysis['sample_analyzed']} muestras):")
            print(f"      Resolución promedio: {analysis['avg_width']:.0f}x{analysis['avg_height']:.0f}")
            print(f"      Rango: {analysis['min_width']}x{analysis['min_height']} - {analysis['max_width']}x{analysis['max_height']}")
            
            return analysis
        
        return {"error": "No se pudieron analizar las dimensiones"}
    
    def _find_common_resolutions(self, dimensions: List[Tuple[int, int]]) -> Dict:
        """🔍 Encuentra resoluciones comunes"""
        resolution_count = {}
        for w, h in dimensions:
            res = f"{w}x{h}"
            resolution_count[res] = resolution_count.get(res, 0) + 1
        
        # Ordenar por frecuencia
        sorted_resolutions = sorted(resolution_count.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_resolutions[:5])  # Top 5
    
    def _get_recommendations(self, dataset_type: str, basic_info: Dict, detailed_info: Dict) -> List[str]:
        """💡 Genera recomendaciones para el dataset"""
        recommendations = []
        
        if dataset_type == "YOLO":
            recommendations.append("✅ Dataset YOLO listo para entrenar")
            recommendations.append("💡 Verificar que todas las clases estén en classes.txt")
            
        elif dataset_type == "COCO":
            recommendations.append("✅ Dataset COCO detectado")
            recommendations.append("💡 Validar estructura JSON antes del entrenamiento")
            
        elif dataset_type == "PASCAL_VOC":
            recommendations.append("🔄 Convertir a YOLO para mejor compatibilidad")
            recommendations.append("💡 Usar el convertidor automático")
            
        elif dataset_type == "PURE_IMAGES":
            recommendations.append("📝 Dataset sin anotaciones")
            recommendations.append("💡 Considerar anotar manualmente o usar herramientas automáticas")
            
        elif dataset_type == "CLASSIFICATION":
            recommendations.append("📊 Dataset de clasificación por carpetas")
            recommendations.append("💡 Verificar balance entre clases")
            
        elif dataset_type == "UNET":
            recommendations.append("🎭 Dataset de segmentación U-Net")
            recommendations.append("💡 Verificar correspondencia imagen-máscara")
            
        # Recomendaciones basadas en tamaño
        if basic_info.get('images', 0) < 100:
            recommendations.append("⚠️ Dataset pequeño - considerar data augmentation")
        elif basic_info.get('images', 0) > 10000:
            recommendations.append("📈 Dataset grande - considerar dividir en train/val/test")
        
        # Recomendaciones basadas en tamaño de archivo
        total_size = detailed_info.get('size_analysis', {}).get('total_size_mb', 0)
        if total_size > 1000:  # > 1GB
            recommendations.append("💾 Dataset pesado - considerar optimización de imágenes")
        
        return recommendations
    
    def save_analysis_report(self, analysis: Dict, output_path: str = None):
        """💾 Guarda el reporte de análisis"""
        if output_path is None:
            output_path = f"dataset_analysis_{analysis['name']}.json"
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Reporte guardado en: {output_path}")
    
    def print_summary(self, analysis: Dict):
        """📋 Imprime resumen del análisis"""
        if "error" in analysis:
            print(f"❌ Error: {analysis['error']}")
            return
        
        print(f"\n📋 RESUMEN DEL ANÁLISIS")
        print("="*30)
        print(f"📁 Dataset: {analysis['name']}")
        print(f"🔍 Tipo: {analysis['type']}")
        print(f"🖼️ Imágenes: {analysis['basic_info']['images']}")
        print(f"💾 Tamaño: {analysis['basic_info']['size_mb']:.1f} MB")
        
        print(f"\n💡 RECOMENDACIONES:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")


def main():
    """🚀 Función principal"""
    parser = argparse.ArgumentParser(description="🎯 Escáner específico de dataset")
    parser.add_argument("path", help="Ruta del dataset a analizar")
    parser.add_argument("--save", "-s", help="Guardar reporte en archivo JSON")
    parser.add_argument("--detailed", "-d", action="store_true", help="Mostrar análisis detallado")
    
    args = parser.parse_args()
    
    scanner = SpecificDatasetScanner()
    
    # Analizar dataset
    analysis = scanner.analyze_dataset(args.path)
    
    # Mostrar resumen
    scanner.print_summary(analysis)
    
    # Guardar reporte si se solicita
    if args.save:
        scanner.save_analysis_report(analysis, args.save)
    
    print(f"\n✅ Análisis completado!")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Modo interactivo si no hay argumentos
        print("🎯 ESCÁNER ESPECÍFICO DE DATASET")
        print("="*40)
        
        dataset_path = input("📁 Ingresa la ruta del dataset: ").strip()
        
        if not dataset_path:
            print("❌ Ruta vacía")
            sys.exit(1)
        
        scanner = SpecificDatasetScanner()
        analysis = scanner.analyze_dataset(dataset_path)
        scanner.print_summary(analysis)
        
        # Preguntar si guardar reporte
        save_response = input("\n💾 ¿Guardar reporte en JSON? (s/N): ").strip().lower()
        if save_response in ['s', 'si', 'sí', 'yes', 'y']:
            filename = f"analysis_{Path(dataset_path).name}.json"
            scanner.save_analysis_report(analysis, filename)
    else:
        main()
