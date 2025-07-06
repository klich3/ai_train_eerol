#!/usr/bin/env python3
"""
🎨 Visual Model Demo
Demostración visual para modelos YOLO dentales
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics no está instalado. Instala con: pip install ultralytics")

class VisualModelDemo:
    """Demo visual para modelos dentales."""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.classes = None
        
        # Colores para cada clase
        self.colors = sns.color_palette("husl", 20)
        
    def load_model(self):
        """Cargar modelo YOLO."""
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics no está disponible")
            
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            
        print(f"🔄 Cargando modelo: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.classes = self.model.names
        print(f"✅ Modelo cargado - Clases: {list(self.classes.values())}")
        
    def predict_and_visualize(self, image_path, output_path=None):
        """Predecir y visualizar resultados."""
        if not self.model:
            self.load_model()
            
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Realizar predicción
        results = self.model(str(image_path))
        
        # Crear visualización
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Imagen original
        axes[0].imshow(image_rgb)
        axes[0].set_title("Imagen Original", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Imagen con predicciones
        axes[1].imshow(image_rgb)
        axes[1].set_title("Predicciones del Modelo", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        detections_info = []
        
        # Procesar resultados
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Extraer información
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf.cpu().numpy())
                    cls_id = int(box.cls.cpu().numpy())
                    cls_name = self.classes.get(cls_id, f"class_{cls_id}")
                    
                    # Color para esta clase
                    color = self.colors[cls_id % len(self.colors)]
                    
                    # Dibujar caja
                    rect = Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=3, 
                        edgecolor=color, 
                        facecolor='none'
                    )
                    axes[1].add_patch(rect)
                    
                    # Etiqueta
                    label = f"{cls_name}: {conf:.2f}"
                    axes[1].text(
                        x1, y1-10, label,
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                        color='white'
                    )
                    
                    detections_info.append({
                        'class': cls_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # Añadir información de detecciones
        info_text = f"Detecciones: {len(detections_info)}\n"
        for i, det in enumerate(detections_info[:5]):  # Máximo 5
            info_text += f"{i+1}. {det['class']}: {det['confidence']:.3f}\n"
        if len(detections_info) > 5:
            info_text += f"... y {len(detections_info)-5} más"
            
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Guardar si se especifica
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualización guardada: {output_path}")
        
        plt.show()
        
        return detections_info
    
    def create_detection_summary(self, detections_list, output_path=None):
        """Crear resumen de detecciones."""
        # Agrupar por clase
        class_counts = {}
        class_confidences = {}
        
        for detections in detections_list:
            for det in detections:
                cls_name = det['class']
                conf = det['confidence']
                
                if cls_name not in class_counts:
                    class_counts[cls_name] = 0
                    class_confidences[cls_name] = []
                    
                class_counts[cls_name] += 1
                class_confidences[cls_name].append(conf)
        
        # Crear gráfico
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfico de barras - conteos
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        axes[0,0].bar(classes, counts, color=sns.color_palette("viridis", len(classes)))
        axes[0,0].set_title("Detecciones por Clase", fontweight='bold')
        axes[0,0].set_ylabel("Número de Detecciones")
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Gráfico de confianza promedio
        avg_confidences = [np.mean(class_confidences[cls]) for cls in classes]
        axes[0,1].bar(classes, avg_confidences, color=sns.color_palette("plasma", len(classes)))
        axes[0,1].set_title("Confianza Promedio por Clase", fontweight='bold')
        axes[0,1].set_ylabel("Confianza Promedio")
        axes[0,1].set_ylim(0, 1)
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Distribución de confianzas
        all_confidences = []
        for confs in class_confidences.values():
            all_confidences.extend(confs)
            
        axes[1,0].hist(all_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,0].set_title("Distribución de Confianzas", fontweight='bold')
        axes[1,0].set_xlabel("Confianza")
        axes[1,0].set_ylabel("Frecuencia")
        
        # Tabla de estadísticas
        axes[1,1].axis('off')
        stats_text = "📊 ESTADÍSTICAS\n\n"
        stats_text += f"Total detecciones: {sum(counts)}\n"
        stats_text += f"Clases detectadas: {len(classes)}\n"
        stats_text += f"Confianza promedio: {np.mean(all_confidences):.3f}\n\n"
        
        stats_text += "Por clase:\n"
        for cls in classes:
            count = class_counts[cls]
            avg_conf = np.mean(class_confidences[cls])
            stats_text += f"• {cls}: {count} ({avg_conf:.3f})\n"
            
        axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes,
                      fontsize=11, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Resumen guardado: {output_path}")
            
        plt.show()

def demo_model(model_path, images_dir, max_images=5):
    """Ejecutar demo del modelo."""
    print("🎨 INICIANDO DEMO VISUAL")
    print("=======================")
    
    if not YOLO_AVAILABLE:
        print("❌ Error: ultralytics no está instalado")
        print("💡 Instala con: pip install ultralytics matplotlib seaborn")
        return
    
    # Inicializar demo
    demo = VisualModelDemo(model_path)
    
    try:
        demo.load_model()
    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        return
    
    # Buscar imágenes
    images_dir = Path(images_dir)
    if not images_dir.exists():
        print(f"❌ Directorio de imágenes no encontrado: {images_dir}")
        return
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    image_files = image_files[:max_images]
    
    if not image_files:
        print(f"❌ No se encontraron imágenes en: {images_dir}")
        return
    
    print(f"🖼️ Procesando {len(image_files)} imágenes...")
    
    # Crear directorio de salida
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    
    all_detections = []
    
    # Procesar cada imagen
    for i, img_path in enumerate(image_files):
        print(f"\n📸 Procesando {i+1}/{len(image_files)}: {img_path.name}")
        
        try:
            output_path = output_dir / f"demo_{i+1:02d}_{img_path.stem}.png"
            detections = demo.predict_and_visualize(img_path, output_path)
            all_detections.append(detections)
            
        except Exception as e:
            print(f"❌ Error procesando {img_path.name}: {e}")
    
    # Crear resumen
    if all_detections:
        print(f"\n📊 Creando resumen de resultados...")
        summary_path = output_dir / "detection_summary.png"
        demo.create_detection_summary(all_detections, summary_path)
        
        print(f"\n🎉 Demo completado!")
        print(f"📂 Resultados en: {output_dir}")

if __name__ == "__main__":
    # Configuración por defecto
    base_dir = Path(__file__).parent
    
    # Buscar modelo
    model_paths = [
        base_dir / "models" / "yolo_detect" / "best.pt",
        base_dir / "datasets" / "detection_combined" / "logs" / "training_*/weights/best.pt",
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break
        # Buscar con glob para patrones
        matches = list(path.parent.parent.glob(path.name)) if path.parent.parent.exists() else []
        if matches:
            model_path = matches[0]
            break
    
    if not model_path:
        print("❌ No se encontró modelo entrenado")
        print("💡 Entrena un modelo primero o especifica la ruta manualmente")
        sys.exit(1)
    
    # Buscar imágenes
    images_dir = base_dir / "datasets" / "detection_combined" / "val" / "images"
    if not images_dir.exists():
        images_dir = base_dir / "datasets" / "detection_combined" / "test" / "images"
    
    if not images_dir.exists():
        print("❌ No se encontraron imágenes de prueba")
        sys.exit(1)
    
    # Ejecutar demo
    demo_model(model_path, images_dir, max_images=3)
