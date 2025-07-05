#!/usr/bin/env python3
"""
🧪 DEMO DE LA HERRAMIENTA DE PREVISUALIZACIÓN
=============================================

Este script demuestra cómo usar la herramienta de previsualización
con datos de ejemplo.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def crear_imagen_ejemplo():
    """Crear una imagen de ejemplo."""
    # Crear imagen de 640x640 simulando una radiografía dental
    img = np.ones((640, 640, 3), dtype=np.uint8) * 40  # Fondo gris oscuro
    
    # Simular una mandíbula
    cv2.ellipse(img, (320, 500), (250, 100), 0, 0, 180, (200, 200, 200), -1)
    
    # Simular dientes
    teeth_positions = [
        (200, 480), (240, 475), (280, 470), (320, 468), 
        (360, 470), (400, 475), (440, 480)
    ]
    
    for i, (x, y) in enumerate(teeth_positions):
        # Diente normal
        cv2.rectangle(img, (x-15, y-25), (x+15, y+10), (255, 255, 255), -1)
        
        # Agregar algunos problemas dentales
        if i == 2:  # Caries en el tercer diente
            cv2.circle(img, (x, y-10), 8, (50, 50, 50), -1)
        elif i == 5:  # Empaste en el sexto diente
            cv2.rectangle(img, (x-8, y-15), (x+8, y-5), (180, 180, 180), -1)
    
    return img

def crear_anotaciones_yolo():
    """Crear anotaciones YOLO de ejemplo."""
    # Formato YOLO: class_id x_center y_center width height (normalizadas)
    anotaciones = [
        "0 0.3125 0.75 0.046875 0.0546875",  # Diente en (200, 480)
        "1 0.4375 0.728125 0.025 0.025",     # Caries en (280, 470)  
        "0 0.4375 0.742188 0.046875 0.0546875", # Diente en (280, 470)
        "0 0.5 0.73125 0.046875 0.0546875",     # Diente en (320, 468)
        "0 0.5625 0.734375 0.046875 0.0546875", # Diente en (360, 470)
        "2 0.625 0.728125 0.025 0.015625",      # Empaste en (400, 475)
        "0 0.625 0.742188 0.046875 0.0546875",  # Diente en (400, 475)
        "0 0.6875 0.75 0.046875 0.0546875"      # Diente en (440, 480)
    ]
    return "\n".join(anotaciones)

def crear_archivo_clases():
    """Crear archivo de clases dentales."""
    clases = [
        "tooth",      # 0
        "caries",     # 1  
        "filling",    # 2
        "crown",      # 3
        "implant",    # 4
        "root_canal", # 5
        "bone_loss",  # 6
        "impacted"    # 7
    ]
    return "\n".join(clases)

def crear_datos_ejemplo():
    """Crear dataset de ejemplo para la demo."""
    print("🧪 CREANDO DATOS DE EJEMPLO PARA DEMO")
    print("="*40)
    
    # Crear carpeta de ejemplo
    demo_folder = Path("demo_dataset")
    demo_folder.mkdir(exist_ok=True)
    
    images_folder = demo_folder / "images"
    labels_folder = demo_folder / "labels"
    images_folder.mkdir(exist_ok=True)
    labels_folder.mkdir(exist_ok=True)
    
    # Crear imagen de ejemplo
    print("📸 Creando imagen de ejemplo...")
    img_ejemplo = crear_imagen_ejemplo()
    img_path = images_folder / "dental_xray_001.jpg"
    cv2.imwrite(str(img_path), cv2.cvtColor(img_ejemplo, cv2.COLOR_RGB2BGR))
    
    # Crear anotaciones YOLO
    print("📋 Creando anotaciones YOLO...")
    anotaciones = crear_anotaciones_yolo()
    txt_path = labels_folder / "dental_xray_001.txt"
    with open(txt_path, 'w') as f:
        f.write(anotaciones)
    
    # Crear archivo de clases
    print("📝 Creando archivo de clases...")
    clases = crear_archivo_clases()
    classes_path = demo_folder / "classes.txt"
    with open(classes_path, 'w') as f:
        f.write(clases)
    
    print(f"✅ Datos de ejemplo creados en: {demo_folder}")
    print(f"   📸 Imagen: {img_path}")
    print(f"   📋 Anotaciones: {txt_path}")
    print(f"   📝 Clases: {classes_path}")
    
    return img_path, txt_path, classes_path

def demo_herramienta_previewt():
    """Demostrar la herramienta de previsualización."""
    print("🔍 DEMO DE HERRAMIENTA DE PREVISUALIZACIÓN")
    print("="*50)
    print()
    
    # Crear datos de ejemplo
    img_path, txt_path, classes_path = crear_datos_ejemplo()
    
    print(f"\n🚀 EJECUTANDO PREVISUALIZACIÓN...")
    
    try:
        from dataset_preview_tool import DatasetPreviewTool
        
        # Crear herramienta
        tool = DatasetPreviewTool()
        
        # Previsualizar
        tool.preview_dataset(
            image_path=str(img_path),
            annotation_path=str(txt_path),
            format_type='yolo',
            classes_file=str(classes_path),
            save_output="demo_preview_result.jpg"
        )
        
        print(f"\n✅ Demo completada exitosamente!")
        print(f"💾 Resultado guardado en: demo_preview_result.jpg")
        
    except Exception as e:
        print(f"❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n💡 PRÓXIMOS PASOS:")
    print(f"1. Revisa la imagen generada con anotaciones")
    print(f"2. Usa la herramienta con tus propios datos:")
    print(f"   python dataset_preview_tool.py imagen.jpg anotaciones.txt")
    print(f"3. Modo interactivo:")
    print(f"   python preview_interactive.py")

def mostrar_comandos_ejemplo():
    """Mostrar ejemplos de comandos."""
    print("📚 EJEMPLOS DE USO DE LA HERRAMIENTA")
    print("="*40)
    print()
    
    print("🎯 LÍNEA DE COMANDOS:")
    print("# YOLO formato")
    print("python dataset_preview_tool.py imagen.jpg anotaciones.txt --format yolo")
    print()
    print("# COCO formato")
    print("python dataset_preview_tool.py imagen.jpg annotations.json --format coco")
    print()
    print("# Con archivo de clases")
    print("python dataset_preview_tool.py imagen.jpg anotaciones.txt --classes classes.txt")
    print()
    print("# Guardar resultado")
    print("python dataset_preview_tool.py imagen.jpg anotaciones.txt --output resultado.jpg")
    print()
    
    print("🎯 MODO INTERACTIVO:")
    print("python preview_interactive.py")
    print()
    
    print("🎯 DESDE MAIN.PY:")
    print("python main.py")
    print("# Selecciona opción 15")

def main():
    """Función principal de la demo."""
    print("🧪 DEMO COMPLETA DE PREVISUALIZACIÓN DE DATASETS")
    print("="*55)
    print()
    
    opciones = """
    ¿Qué te gustaría hacer?
    
    1. 🧪 Ejecutar demo con datos de ejemplo
    2. 📚 Ver ejemplos de comandos
    3. 🔍 Información sobre formatos soportados
    4. 🚀 Abrir herramienta interactiva
    0. 🚪 Salir
    """
    
    while True:
        print(opciones)
        choice = input("🎯 Selecciona una opción: ").strip()
        
        if choice == '1':
            demo_herramienta_previewt()
        elif choice == '2':
            mostrar_comandos_ejemplo()
        elif choice == '3':
            from preview_interactive import mostrar_ayuda_formatos
            mostrar_ayuda_formatos()
        elif choice == '4':
            import subprocess
            subprocess.run([sys.executable, 'preview_interactive.py'])
        elif choice == '0':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida")
        
        if choice != '0':
            input("\n📋 Presiona Enter para continuar...")

if __name__ == "__main__":
    main()
