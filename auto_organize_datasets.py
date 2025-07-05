#!/usr/bin/env python3
"""
🤖 Auto-Organizador Inteligente de Datasets Dentales
====================================================

Escanea carpetas sueltas en _dataSets, detecta su tipo automáticamente,
convierte al formato estándar y las mueve a la carpeta correspondiente.

Author: Anton Sychev
Created: 2025-07-05
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))


class SmartDatasetOrganizer:
    """🤖 Organizador inteligente de datasets"""
    
    def __init__(self, base_path: str = "_dataSets", min_images: int = 5):
        self.base_path = Path(base_path)
        self.organized_folders = {"_YOLO", "_COCO", "_UNET", "_pure images and masks", "Archivo", "Dataset"}
        self.supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.min_images = min_images  # Mínimo de imágenes para considerar un dataset válido
        
        # Contadores para estadísticas
        self.stats = {
            'carpetas_analizadas': 0,
            'subcarpetas_encontradas': 0,
            'datasets_detectados': 0,
            'datasets_convertidos': 0,
            'datasets_movidos': 0,
            'datasets_muy_pequeños': 0,
            'errores': 0
        }
    
    def scan_unorganized_folders(self) -> List[Path]:
        """🔍 Escanea recursivamente carpetas que no están organizadas."""
        print("🔍 ESCANEANDO CARPETAS NO ORGANIZADAS...")
        
        unorganized = []
        
        if not self.base_path.exists():
            print(f"❌ El directorio {self.base_path} no existe")
            return unorganized
        
        def scan_recursive(current_path: Path, depth: int = 0, max_depth: int = 3):
            """Escanea recursivamente hasta una profundidad máxima"""
            if depth > max_depth:
                return
                
            for item in current_path.iterdir():
                if (item.is_dir() and 
                    not item.name.startswith('.') and
                    item.name not in self.organized_folders):
                    
                    # Si estamos en el nivel base, evitar carpetas organizadas
                    if depth == 0 and item.name in self.organized_folders:
                        continue
                    
                    # Contar imágenes en esta carpeta (recursivamente)
                    image_count = self._count_images_recursive(item)
                    
                    if image_count >= self.min_images:
                        unorganized.append(item)
                        print(f"   📁 Encontrada: {item.relative_to(self.base_path)} ({image_count} imágenes)")
                    elif image_count > 0:
                        print(f"   � Muy pequeña: {item.relative_to(self.base_path)} ({image_count} imágenes)")
                        self.stats['datasets_muy_pequeños'] += 1
                    
                    # Continuar escaneando subcarpetas si no tiene suficientes imágenes
                    if image_count < self.min_images:
                        scan_recursive(item, depth + 1, max_depth)
                    
                    self.stats['subcarpetas_encontradas'] += 1
        
        # Iniciar escaneo recursivo
        scan_recursive(self.base_path)
        
        print(f"📊 Total carpetas válidas: {len(unorganized)}")
        print(f"📊 Carpetas muy pequeñas: {self.stats['datasets_muy_pequeños']}")
        return unorganized
    
    def _count_images_recursive(self, folder_path: Path) -> int:
        """Cuenta imágenes recursivamente en una carpeta"""
        count = 0
        try:
            for ext in self.supported_image_extensions:
                count += len(list(folder_path.glob(f"**/*{ext}")))
        except Exception:
            pass
        return count
    
    def detect_dataset_type(self, folder_path: Path) -> Tuple[str, Dict]:
        """🔍 Detecta automáticamente el tipo de dataset."""
        print(f"\n🔍 Analizando: {folder_path.relative_to(self.base_path)}")
        
        # Obtener archivos de la carpeta recursivamente
        all_files = list(folder_path.rglob("*"))
        image_files = [f for f in all_files if f.suffix.lower() in self.supported_image_extensions]
        txt_files = [f for f in all_files if f.suffix.lower() == '.txt']
        json_files = [f for f in all_files if f.suffix.lower() == '.json']
        xml_files = [f for f in all_files if f.suffix.lower() == '.xml']
        
        info = {
            'total_files': len(all_files),
            'images': len(image_files),
            'txt_files': len(txt_files),
            'json_files': len(json_files),
            'xml_files': len(xml_files),
            'folder_structure': self._analyze_folder_structure(folder_path),
            'size_mb': self._calculate_folder_size(folder_path)
        }
        
        print(f"   📊 Imágenes: {info['images']}, TXT: {info['txt_files']}, JSON: {info['json_files']}, XML: {info['xml_files']}")
        print(f"   💾 Tamaño: {info['size_mb']:.1f} MB")
        
        # Validar que tenga suficientes imágenes
        if info['images'] < self.min_images:
            print(f"   ⚠️ Muy pocas imágenes ({info['images']} < {self.min_images})")
            return "TOO_SMALL", info
        
        # 1. DETECTAR YOLO (mayor prioridad si hay archivos .txt)
        if self._is_yolo_format(folder_path, image_files, txt_files):
            return "YOLO", info
        
        # 2. DETECTAR COCO
        if self._is_coco_format(folder_path, image_files, json_files):
            return "COCO", info
        
        # 3. DETECTAR U-NET (máscaras)
        if self._is_unet_format(folder_path, image_files):
            return "UNET", info
        
        # 4. DETECTAR PASCAL VOC
        if self._is_pascal_voc_format(folder_path, image_files, xml_files):
            return "PASCAL_VOC", info
        
        # 5. DETECTAR CLASIFICACIÓN (estructura de carpetas por clase)
        if self._is_classification_format(folder_path, image_files):
            return "CLASSIFICATION", info
        
        # 6. IMÁGENES PURAS (sin anotaciones pero con suficientes imágenes)
        if len(image_files) >= self.min_images:
            return "PURE_IMAGES", info
        
        return "UNKNOWN", info
    
    def _calculate_folder_size(self, folder_path: Path) -> float:
        """Calcula el tamaño de la carpeta en MB"""
        total_size = 0
        try:
            for file_path in folder_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception:
            pass
        return total_size / (1024 * 1024)  # Convert to MB
        
        # 5. DETECTAR PASCAL VOC
        if self._is_pascal_voc_format(folder_path, image_files, xml_files):
            return "PASCAL_VOC", info
        
        # 6. IMÁGENES PURAS (sin anotaciones)
        if len(image_files) > 0:
            return "PURE_IMAGES", info
        
        return "UNKNOWN", info
    
    def _is_yolo_format(self, folder_path: Path, image_files: List[Path], txt_files: List[Path]) -> bool:
        """Detecta formato YOLO con validación mejorada"""
        if len(image_files) == 0 or len(txt_files) == 0:
            return False
        
        # Verificar que haya correspondencia entre imágenes y txt files
        image_stems = {img.stem for img in image_files}
        txt_stems = {txt.stem for txt in txt_files}
        
        # Al menos 30% de coincidencias
        matching_ratio = len(image_stems & txt_stems) / len(image_stems)
        if matching_ratio < 0.3:
            return False
        
        # Buscar archivos .txt con coordenadas YOLO válidas
        yolo_annotations = 0
        sample_size = min(10, len(txt_files))
        
        for txt_file in txt_files[:sample_size]:
            try:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                    valid_lines = 0
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # class x y w h
                            try:
                                # Verificar formato YOLO (valores entre 0 y 1)
                                class_id = int(parts[0])
                                coords = [float(p) for p in parts[1:5]]
                                if (class_id >= 0 and 
                                    all(0 <= c <= 1 for c in coords) and
                                    coords[2] > 0 and coords[3] > 0):  # width y height > 0
                                    valid_lines += 1
                            except ValueError:
                                continue
                    
                    # Si al menos 50% de las líneas son válidas
                    if len(lines) > 0 and valid_lines / len(lines) >= 0.5:
                        yolo_annotations += 1
                        
            except Exception:
                continue
        
        detection_ratio = yolo_annotations / sample_size if sample_size > 0 else 0
        is_yolo = detection_ratio >= 0.4  # 40% de archivos válidos
        
        if is_yolo:
            print(f"   ✅ YOLO detectado: {yolo_annotations}/{sample_size} archivos válidos")
        
        return is_yolo
    
    def _is_coco_format(self, folder_path: Path, image_files: List[Path], json_files: List[Path]) -> bool:
        """Detecta formato COCO con validación mejorada"""
        if len(json_files) == 0:
            return False
        
        # Buscar archivos JSON con estructura COCO
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if (isinstance(data, dict) and 
                        'images' in data and 
                        'annotations' in data and 
                        'categories' in data):
                        
                        # Validar estructura básica
                        images = data.get('images', [])
                        annotations = data.get('annotations', [])
                        categories = data.get('categories', [])
                        
                        if (len(images) > 0 and 
                            len(annotations) > 0 and 
                            len(categories) > 0):
                            print(f"   ✅ COCO detectado: {len(images)} imágenes, {len(annotations)} anotaciones")
                            return True
            except Exception:
                continue
        
        return False
    
    def _is_unet_format(self, folder_path: Path, image_files: List[Path]) -> bool:
        """Detecta formato U-Net (imágenes + máscaras) con validación mejorada"""
        if len(image_files) == 0:
            return False
            
        # Buscar patrones típicos de U-Net
        mask_keywords = {'mask', 'label', 'gt', 'ground_truth', 'seg', 'segmentation', 'annotation'}
        
        # Verificar estructura de carpetas
        subdirs = [d.name.lower() for d in folder_path.rglob("*") if d.is_dir()]
        has_mask_folder = any(keyword in subdir for subdir in subdirs for keyword in mask_keywords)
        
        if has_mask_folder:
            print(f"   ✅ U-Net detectado: carpeta de máscaras encontrada")
            return True
        
        # Verificar si hay imágenes que parecen máscaras por nombre
        potential_masks = 0
        sample_size = min(50, len(image_files))
        
        for img_file in image_files[:sample_size]:
            if any(keyword in img_file.name.lower() for keyword in mask_keywords):
                potential_masks += 1
        
        ratio = potential_masks / sample_size if sample_size > 0 else 0
        is_unet = ratio > 0.2  # 20% son máscaras
        
        if is_unet:
            print(f"   ✅ U-Net detectado: {potential_masks}/{sample_size} archivos parecen máscaras")
        
        return is_unet
    
    def _is_classification_format(self, folder_path: Path, image_files: List[Path]) -> bool:
        """Detecta formato de clasificación (carpetas por clase) con validación mejorada"""
        if len(image_files) == 0:
            return False
            
        # Verificar estructura: carpetas con nombres de clases
        class_folders = [d for d in folder_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if len(class_folders) < 2:
            return False
        
        # Verificar que las subcarpetas contengan imágenes
        images_in_subfolders = 0
        total_images_in_subfolders = 0
        valid_class_folders = 0
        
        for class_folder in class_folders:
            class_images = []
            for ext in self.supported_image_extensions:
                class_images.extend(class_folder.glob(f"*{ext}"))
            
            if len(class_images) >= 3:  # Al menos 3 imágenes por clase
                valid_class_folders += 1
                total_images_in_subfolders += len(class_images)
        
        # Si la mayoría de imágenes están en subcarpetas válidas, es clasificación
        if len(image_files) > 0:
            ratio = total_images_in_subfolders / len(image_files)
            is_classification = ratio > 0.7 and valid_class_folders >= 2
            
            if is_classification:
                print(f"   ✅ Clasificación detectada: {valid_class_folders} clases, {total_images_in_subfolders} imágenes")
            
            return is_classification
        
        return False
    
    def _is_pascal_voc_format(self, folder_path: Path, image_files: List[Path], xml_files: List[Path]) -> bool:
        """Detecta formato Pascal VOC con validación mejorada"""
        if len(xml_files) == 0 or len(image_files) == 0:
            return False
        
        # Verificar correspondencia entre imágenes y XML
        image_stems = {img.stem for img in image_files}
        xml_stems = {xml.stem for xml in xml_files}
        matching_ratio = len(image_stems & xml_stems) / len(image_stems)
        
        if matching_ratio < 0.3:
            return False
        
        # Verificar estructura XML de Pascal VOC
        voc_annotations = 0
        sample_size = min(10, len(xml_files))
        
        for xml_file in xml_files[:sample_size]:
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                if (root.tag == 'annotation' and
                    root.find('filename') is not None and
                    root.find('object') is not None and
                    root.find('size') is not None):
                    voc_annotations += 1
            except Exception:
                continue
        
        detection_ratio = voc_annotations / sample_size if sample_size > 0 else 0
        is_voc = detection_ratio >= 0.4
        
        if is_voc:
            print(f"   ✅ Pascal VOC detectado: {voc_annotations}/{sample_size} archivos válidos")
        
        return is_voc
    
    def _analyze_folder_structure(self, folder_path: Path) -> Dict:
        """Analiza la estructura de carpetas"""
        structure = {
            'subdirs': [],
            'common_patterns': [],
            'depth': 0
        }
        
        subdirs = [d.name for d in folder_path.iterdir() if d.is_dir()]
        structure['subdirs'] = subdirs
        
        # Detectar patrones comunes
        common_patterns = {
            'train_val_test': any(name in subdirs for name in ['train', 'val', 'test']),
            'images_labels': any(name in subdirs for name in ['images', 'labels', 'annotations']),
            'class_based': len(subdirs) > 1 and all(len(name) < 20 for name in subdirs)
        }
        
        structure['common_patterns'] = [k for k, v in common_patterns.items() if v]
        
        return structure
    
    def convert_and_organize_dataset(self, folder_path: Path, dataset_type: str, info: Dict) -> bool:
        """🔄 Convierte y organiza el dataset según su tipo."""
        print(f"\n🔄 PROCESANDO: {folder_path.name} (Tipo: {dataset_type})")
        
        try:
            # Determinar carpeta destino
            if dataset_type == "YOLO":
                target_folder = self.base_path / "_YOLO"
            elif dataset_type == "COCO":
                target_folder = self.base_path / "_COCO"
            elif dataset_type == "UNET":
                target_folder = self.base_path / "_UNET"
            elif dataset_type == "CLASSIFICATION":
                target_folder = self.base_path / "_pure images and masks"
            elif dataset_type == "PASCAL_VOC":
                # Convertir Pascal VOC a YOLO y mover a _YOLO
                return self._convert_pascal_voc_to_yolo(folder_path, info)
            elif dataset_type == "PURE_IMAGES":
                target_folder = self.base_path / "_pure images and masks"
            else:
                print(f"⚠️ Tipo {dataset_type} no soportado para conversión automática")
                return False
            
            # Crear carpeta destino si no existe
            target_folder.mkdir(exist_ok=True)
            destination = target_folder / folder_path.name
            
            # Verificar si ya existe
            if destination.exists():
                print(f"⚠️ Ya existe: {destination}")
                response = input("¿Sobrescribir? (s/N): ").strip().lower()
                if response not in ['s', 'si', 'sí', 'yes', 'y']:
                    return False
                shutil.rmtree(destination)
            
            # Mover la carpeta
            shutil.move(str(folder_path), str(destination))
            print(f"✅ Movido a: {destination}")
            
            self.stats['datasets_movidos'] += 1
            return True
            
        except Exception as e:
            print(f"❌ Error procesando {folder_path.name}: {e}")
            self.stats['errores'] += 1
            return False
    
    def _convert_pascal_voc_to_yolo(self, folder_path: Path, info: Dict) -> bool:
        """Convierte Pascal VOC a formato YOLO"""
        print(f"🔄 Convirtiendo Pascal VOC a YOLO...")
        
        try:
            import xml.etree.ElementTree as ET
            
            # Crear carpeta temporal para la conversión
            temp_folder = folder_path.parent / f"{folder_path.name}_yolo_converted"
            temp_folder.mkdir(exist_ok=True)
            
            # Buscar archivos XML e imágenes
            xml_files = list(folder_path.glob("**/*.xml"))
            image_files = []
            for ext in self.supported_image_extensions:
                image_files.extend(folder_path.glob(f"**/*{ext}"))
            
            converted_count = 0
            
            for xml_file in xml_files:
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    # Obtener información de la imagen
                    filename = root.find('filename').text
                    size = root.find('size')
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
                    
                    # Encontrar imagen correspondiente
                    image_path = None
                    for img in image_files:
                        if img.name == filename or img.stem == Path(filename).stem:
                            image_path = img
                            break
                    
                    if not image_path:
                        continue
                    
                    # Copiar imagen
                    shutil.copy2(image_path, temp_folder / image_path.name)
                    
                    # Convertir anotaciones a YOLO
                    yolo_annotations = []
                    for obj in root.findall('object'):
                        class_name = obj.find('name').text
                        bbox = obj.find('bndbox')
                        
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)
                        
                        # Convertir a formato YOLO
                        x_center = (xmin + xmax) / 2.0 / width
                        y_center = (ymin + ymax) / 2.0 / height
                        bbox_width = (xmax - xmin) / width
                        bbox_height = (ymax - ymin) / height
                        
                        # Usar 0 como class_id por defecto
                        yolo_annotations.append(f"0 {x_center} {y_center} {bbox_width} {bbox_height}")
                    
                    # Guardar archivo YOLO
                    txt_filename = Path(filename).stem + '.txt'
                    with open(temp_folder / txt_filename, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    
                    converted_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Error convirtiendo {xml_file.name}: {e}")
                    continue
            
            if converted_count > 0:
                # Mover a _YOLO
                target_folder = self.base_path / "_YOLO"
                target_folder.mkdir(exist_ok=True)
                destination = target_folder / f"{folder_path.name}_converted"
                
                if destination.exists():
                    shutil.rmtree(destination)
                
                shutil.move(str(temp_folder), str(destination))
                
                # Eliminar carpeta original
                shutil.rmtree(folder_path)
                
                print(f"✅ Convertido Pascal VOC → YOLO: {converted_count} anotaciones")
                print(f"✅ Guardado en: {destination}")
                
                self.stats['datasets_convertidos'] += 1
                self.stats['datasets_movidos'] += 1
                return True
            else:
                # Limpiar carpeta temporal
                shutil.rmtree(temp_folder)
                return False
                
        except Exception as e:
            print(f"❌ Error en conversión Pascal VOC: {e}")
            return False
    
    def run_auto_organization(self, dry_run: bool = False) -> Dict:
        """🚀 Ejecuta la organización automática."""
        print("🚀 INICIANDO AUTO-ORGANIZACIÓN DE DATASETS")
        print("="*50)
        
        if dry_run:
            print("🔍 MODO DRY-RUN: Solo análisis, sin mover archivos")
        
        # Escanear carpetas no organizadas
        unorganized_folders = self.scan_unorganized_folders()
        
        if not unorganized_folders:
            print("✅ No hay carpetas para organizar")
            return self.stats
        
        print(f"\n📋 PLAN DE ORGANIZACIÓN:")
        print("-" * 30)
        
        organization_plan = []
        
        for folder in unorganized_folders:
            self.stats['carpetas_analizadas'] += 1
            
            dataset_type, info = self.detect_dataset_type(folder)
            
            if dataset_type not in ["UNKNOWN", "TOO_SMALL"]:
                self.stats['datasets_detectados'] += 1
                
                organization_plan.append({
                    'folder': folder,
                    'type': dataset_type,
                    'info': info
                })
                
                print(f"📁 {folder.relative_to(self.base_path)}")
                print(f"   🔍 Tipo detectado: {dataset_type}")
                print(f"   📊 Imágenes: {info['images']} | Tamaño: {info['size_mb']:.1f} MB")
                
                if dataset_type == "YOLO":
                    print(f"   📁 Destino: _YOLO/")
                elif dataset_type == "COCO":
                    print(f"   📁 Destino: _COCO/")
                elif dataset_type == "UNET":
                    print(f"   📁 Destino: _UNET/")
                elif dataset_type == "CLASSIFICATION":
                    print(f"   📁 Destino: _pure images and masks/")
                elif dataset_type == "PASCAL_VOC":
                    print(f"   🔄 Convertir a YOLO → _YOLO/")
                elif dataset_type == "PURE_IMAGES":
                    print(f"   📁 Destino: _pure images and masks/")
                print()
            elif dataset_type == "TOO_SMALL":
                print(f"📁 {folder.relative_to(self.base_path)}")
                print(f"   ⚠️ Muy pequeño: {info['images']} imágenes < {self.min_images}")
                print()
            else:
                print(f"📁 {folder.relative_to(self.base_path)}")
                print(f"   ❓ Tipo no reconocido")
                print(f"   📊 Imágenes: {info['images']}, Otros: {info['total_files'] - info['images']}")
                print()
        
        if not organization_plan:
            print("⚠️ No se detectaron datasets válidos para organizar")
            return self.stats
        
        # Confirmar ejecución
        if not dry_run:
            print(f"\n🎯 RESUMEN:")
            print(f"   📁 Carpetas a procesar: {len(organization_plan)}")
            print(f"   🔄 Conversiones necesarias: {sum(1 for p in organization_plan if p['type'] == 'PASCAL_VOC')}")
            
            response = input(f"\n¿Proceder con la organización? (s/N): ").strip().lower()
            if response not in ['s', 'si', 'sí', 'yes', 'y']:
                print("❌ Operación cancelada")
                return self.stats
        
        # Ejecutar organización
        if not dry_run:
            print(f"\n🔄 EJECUTANDO ORGANIZACIÓN...")
            print("-" * 30)
            
            for plan in organization_plan:
                success = self.convert_and_organize_dataset(
                    plan['folder'], 
                    plan['type'], 
                    plan['info']
                )
                
                if success:
                    print(f"✅ {plan['folder'].name} organizado correctamente")
                else:
                    print(f"❌ Error organizando {plan['folder'].name}")
        
        return self.stats
    
    def print_final_stats(self):
        """📊 Imprime estadísticas finales."""
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print("="*40)
        print(f"📁 Carpetas analizadas: {self.stats['carpetas_analizadas']}")
        print(f"� Subcarpetas encontradas: {self.stats['subcarpetas_encontradas']}")
        print(f"�🔍 Datasets detectados: {self.stats['datasets_detectados']}")
        print(f"� Datasets muy pequeños: {self.stats['datasets_muy_pequeños']}")
        print(f"�🔄 Datasets convertidos: {self.stats['datasets_convertidos']}")
        print(f"📦 Datasets movidos: {self.stats['datasets_movidos']}")
        print(f"❌ Errores: {self.stats['errores']}")
        
        if self.stats['datasets_detectados'] > 0:
            success_rate = (self.stats['datasets_movidos'] / self.stats['datasets_detectados']) * 100
            print(f"✅ Tasa de éxito: {success_rate:.1f}%")
        
        print(f"\n💡 Parámetros utilizados:")
        print(f"   🖼️ Mínimo de imágenes por dataset: {self.min_images}")
        print(f"   📁 Carpetas organizadas excluidas: {', '.join(self.organized_folders)}")


def main():
    """🚀 Función principal"""
    print("🤖 AUTO-ORGANIZADOR INTELIGENTE DE DATASETS DENTALES")
    print("="*60)
    print()
    print("Esta herramienta:")
    print("• 🔍 Escanea recursivamente carpetas en _dataSets")
    print("• 🤖 Detecta automáticamente el tipo de dataset")
    print("• 🔄 Convierte formatos cuando es necesario")
    print("• 📦 Mueve a la carpeta correspondiente")
    print("• 📊 Proporciona estadísticas detalladas")
    print()
    
    # Configuración
    min_images = 5
    config_response = input(f"¿Cambiar mínimo de imágenes por dataset? (actual: {min_images}): ").strip()
    if config_response and config_response.isdigit():
        min_images = int(config_response)
        print(f"✅ Usando mínimo: {min_images} imágenes")
    
    organizer = SmartDatasetOrganizer(min_images=min_images)
    
    # Preguntar si hacer dry-run primero
    dry_run_response = input("\n¿Hacer análisis previo sin mover archivos? (S/n): ").strip().lower()
    dry_run = dry_run_response not in ['n', 'no', 'nope']
    
    # Ejecutar organización
    stats = organizer.run_auto_organization(dry_run=dry_run)
    
    # Mostrar estadísticas
    organizer.print_final_stats()
    
    # Si fue dry-run, preguntar si ejecutar realmente
    if dry_run and stats['datasets_detectados'] > 0:
        print(f"\n💡 El análisis encontró {stats['datasets_detectados']} datasets organizables.")
        execute_response = input("¿Ejecutar la organización real? (s/N): ").strip().lower()
        
        if execute_response in ['s', 'si', 'sí', 'yes', 'y']:
            print(f"\n🔄 EJECUTANDO ORGANIZACIÓN REAL...")
            organizer.stats = {k: 0 for k in organizer.stats.keys()}  # Reset stats
            organizer.run_auto_organization(dry_run=False)
            organizer.print_final_stats()


if __name__ == "__main__":
    main()
