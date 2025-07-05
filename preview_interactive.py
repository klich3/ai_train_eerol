#!/usr/bin/env python3
"""
🔍 DENTAL DATASET PREVIEW - VERSIÓN INTERACTIVA
===============================================

Herramienta interactiva para previsualizar datasets dentales
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

from dataset_preview_tool import DatasetPreviewTool

def mostrar_menu_principal():
    """Mostrar menú principal de previsualización."""
    menu = """
    🔍 HERRAMIENTA DE PREVISUALIZACIÓN DE DATASETS
    =============================================
    
    1. 📸 Previsualizar imagen individual
    2. 📁 Previsualizar múltiples imágenes de una carpeta
    3. 🔄 Convertir y previsualizar formato
    4. 📊 Analizar dataset completo
    5. ❓ Ayuda sobre formatos
    0. 🚪 Salir
    """
    print(menu)

def previsualizar_imagen_individual():
    """Previsualizar una imagen individual."""
    print("\n📸 PREVISUALIZACIÓN DE IMAGEN INDIVIDUAL")
    print("="*45)
    
    # Solicitar rutas
    image_path = input("📸 Ruta de la imagen: ").strip()
    if not image_path:
        print("❌ Ruta de imagen requerida")
        return
    
    annotation_path = input("📋 Ruta de anotaciones: ").strip()
    if not annotation_path:
        print("❌ Ruta de anotaciones requerida")
        return
    
    # Detectar formato
    formato = detectar_formato(annotation_path)
    print(f"🔍 Formato detectado: {formato}")
    
    # Confirmar formato
    confirmar = input(f"¿Es correcto el formato {formato}? (s/N): ").strip().lower()
    if confirmar not in ['s', 'si', 'sí', 'yes', 'y']:
        formato = seleccionar_formato()
    
    # Archivo de clases opcional
    classes_file = input("📋 Archivo de clases (opcional): ").strip() or None
    
    # Guardar resultado
    save_output = input("💾 Guardar resultado (opcional): ").strip() or None
    
    # Crear herramienta y previsualizar
    tool = DatasetPreviewTool()
    try:
        tool.preview_dataset(
            image_path=image_path,
            annotation_path=annotation_path,
            format_type=formato,
            classes_file=classes_file,
            save_output=save_output
        )
    except Exception as e:
        print(f"❌ Error: {e}")

def previsualizar_carpeta():
    """Previsualizar múltiples imágenes de una carpeta."""
    print("\n📁 PREVISUALIZACIÓN DE CARPETA")
    print("="*35)
    
    carpeta_imagenes = input("📁 Carpeta de imágenes: ").strip()
    if not carpeta_imagenes:
        print("❌ Carpeta de imágenes requerida")
        return
    
    carpeta_anotaciones = input("📋 Carpeta de anotaciones: ").strip()
    if not carpeta_anotaciones:
        print("❌ Carpeta de anotaciones requerida")
        return
    
    formato = seleccionar_formato()
    
    # Buscar archivos
    img_folder = Path(carpeta_imagenes)
    ann_folder = Path(carpeta_anotaciones)
    
    if not img_folder.exists():
        print(f"❌ Carpeta de imágenes no existe: {carpeta_imagenes}")
        return
    
    if not ann_folder.exists():
        print(f"❌ Carpeta de anotaciones no existe: {carpeta_anotaciones}")
        return
    
    # Buscar imágenes
    extensiones_img = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    imagenes = []
    for ext in extensiones_img:
        imagenes.extend(img_folder.glob(f'*{ext}'))
        imagenes.extend(img_folder.glob(f'*{ext.upper()}'))
    
    print(f"📊 Imágenes encontradas: {len(imagenes)}")
    
    if not imagenes:
        print("❌ No se encontraron imágenes")
        return
    
    # Mostrar lista de imágenes
    print(f"\n📋 IMÁGENES ENCONTRADAS:")
    for i, img in enumerate(imagenes[:10], 1):  # Mostrar primeras 10
        print(f"   {i}. {img.name}")
    
    if len(imagenes) > 10:
        print(f"   ... y {len(imagenes) - 10} más")
    
    # Seleccionar imagen
    try:
        seleccion = input(f"\nSelecciona imagen (1-{len(imagenes)}) o 'todas' para batch: ").strip()
        
        if seleccion.lower() in ['todas', 'all', 'batch']:
            previsualizar_batch(imagenes, ann_folder, formato)
        else:
            idx = int(seleccion) - 1
            if 0 <= idx < len(imagenes):
                img_seleccionada = imagenes[idx]
                ann_path = encontrar_anotacion(img_seleccionada, ann_folder, formato)
                
                if ann_path:
                    tool = DatasetPreviewTool()
                    tool.preview_dataset(
                        image_path=str(img_seleccionada),
                        annotation_path=str(ann_path),
                        format_type=formato
                    )
                else:
                    print(f"❌ No se encontró anotación para {img_seleccionada.name}")
            else:
                print("❌ Selección inválida")
    
    except ValueError:
        print("❌ Entrada inválida")

def previsualizar_batch(imagenes, ann_folder, formato):
    """Previsualizar múltiples imágenes en batch."""
    print(f"\n🔄 PROCESANDO {len(imagenes)} IMÁGENES...")
    
    tool = DatasetPreviewTool()
    procesadas = 0
    errores = 0
    
    for img_path in imagenes:
        try:
            ann_path = encontrar_anotacion(img_path, ann_folder, formato)
            if ann_path:
                print(f"📸 Procesando: {img_path.name}")
                tool.preview_dataset(
                    image_path=str(img_path),
                    annotation_path=str(ann_path),
                    format_type=formato,
                    save_output=f"preview_{img_path.stem}_annotated.jpg"
                )
                procesadas += 1
            else:
                print(f"⚠️ Sin anotación: {img_path.name}")
        except Exception as e:
            print(f"❌ Error en {img_path.name}: {e}")
            errores += 1
    
    print(f"\n✅ Procesamiento completado:")
    print(f"   📊 Procesadas: {procesadas}")
    print(f"   ❌ Errores: {errores}")

def encontrar_anotacion(img_path, ann_folder, formato):
    """Encontrar archivo de anotación correspondiente."""
    img_stem = img_path.stem
    
    if formato == 'yolo':
        ann_path = ann_folder / f"{img_stem}.txt"
    elif formato == 'coco':
        # Para COCO, buscar annotations.json o similar
        possible_files = ['annotations.json', 'instances.json', f"{img_stem}.json"]
        for filename in possible_files:
            ann_path = ann_folder / filename
            if ann_path.exists():
                return ann_path
        return None
    elif formato == 'csv':
        ann_path = ann_folder / f"{img_stem}.csv"
        # También buscar un CSV general
        if not ann_path.exists():
            ann_path = ann_folder / "annotations.csv"
    else:  # json
        ann_path = ann_folder / f"{img_stem}.json"
    
    return ann_path if ann_path.exists() else None

def detectar_formato(annotation_path):
    """Detectar formato automáticamente."""
    ext = Path(annotation_path).suffix.lower()
    if ext == '.txt':
        return 'yolo'
    elif ext == '.json':
        return 'coco'
    elif ext == '.csv':
        return 'csv'
    else:
        return 'yolo'

def seleccionar_formato():
    """Seleccionar formato manualmente."""
    formatos = {
        '1': 'yolo',
        '2': 'coco', 
        '3': 'csv',
        '4': 'json'
    }
    
    print(f"\n📊 SELECCIONAR FORMATO:")
    print("1. YOLO (.txt)")
    print("2. COCO (.json)")
    print("3. CSV (.csv)")
    print("4. JSON personalizado (.json)")
    
    while True:
        seleccion = input("Selecciona formato (1-4): ").strip()
        if seleccion in formatos:
            return formatos[seleccion]
        print("❌ Selección inválida")

def mostrar_ayuda_formatos():
    """Mostrar ayuda sobre formatos soportados."""
    ayuda = """
    📚 FORMATOS SOPORTADOS
    =====================
    
    🎯 YOLO (.txt):
    Formato: class_id x_center y_center width height
    Ejemplo: 0 0.5 0.5 0.3 0.4
    - Coordenadas normalizadas (0-1)
    - Una línea por objeto
    
    🎯 COCO (.json):
    Formato JSON estándar de COCO
    - Contiene: images, annotations, categories
    - Bbox formato: [x, y, width, height]
    - Coordenadas en píxeles
    
    🎯 CSV (.csv):
    Columnas: filename, class, xmin, ymin, xmax, ymax
    - Coordenadas en píxeles
    - Una fila por objeto
    
    🎯 JSON Personalizado (.json):
    Formato: {"imagen.jpg": [{"class": "...", "bbox": [...]}]}
    - Flexible para formatos personalizados
    
    💡 CONSEJOS:
    • La herramienta detecta automáticamente el formato por extensión
    • Puedes sobrescribir la detección automática
    • Para COCO, asegúrate de que el archivo contenga la imagen especificada
    """
    print(ayuda)

def main():
    """Función principal interactiva."""
    print("🔍 HERRAMIENTA DE PREVISUALIZACIÓN DE DATASETS DENTALES")
    print("="*60)
    print("Visualiza anotaciones en formatos YOLO, COCO, CSV y JSON")
    print()
    
    while True:
        mostrar_menu_principal()
        choice = input("🎯 Selecciona una opción: ").strip()
        
        try:
            if choice == '1':
                previsualizar_imagen_individual()
            elif choice == '2':
                previsualizar_carpeta()
            elif choice == '3':
                print("🚧 Función de conversión en desarrollo")
            elif choice == '4':
                print("🚧 Análisis de dataset completo en desarrollo")
            elif choice == '5':
                mostrar_ayuda_formatos()
            elif choice == '0':
                print("👋 ¡Hasta luego!")
                break
            else:
                print("❌ Opción inválida")
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️ Operación interrumpida")
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
        
        if choice != '0':
            input("\n📋 Presiona Enter para continuar...")

if __name__ == "__main__":
    main()
