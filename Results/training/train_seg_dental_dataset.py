#!/usr/bin/env python3
# 🦷 Entrenamiento de segmentación para dental_dataset

import os
import sys
import torch

# Verificar que detectron2 esté instalado
try:
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2 import model_zoo
    from detectron2.utils.logger import setup_logger
    DETECTRON2_AVAILABLE = True
except ImportError as e:
    print("❌ Error: Detectron2 no está instalado")
    print("💡 Para instalar detectron2, ejecuta:")
    print("   python install_detectron2.py")
    print("   o manualmente:")
    print("   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/index.html")
    print(f"📝 Error específico: {e}")
    DETECTRON2_AVAILABLE = False

def setup_config():
    """Configuración para entrenamiento de segmentación."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Configuración del dataset
    cfg.DATASETS.TRAIN = ("dental_dataset_train",)
    cfg.DATASETS.TEST = ("dental_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Configuración de entrenamiento
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 3000
    cfg.SOLVER.STEPS = (2000,)
    cfg.SOLVER.GAMMA = 0.1
    
    # Configuración del modelo
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("dental_dataset_train").thing_classes)
    
    # Directorio de salida
    cfg.OUTPUT_DIR = f"./logs/dental_dataset_segmentation"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def register_datasets():
    """Registra los datasets COCO."""
    # Usar rutas relativas desde el directorio de ejecución
    dataset_path = "../datasets/segmentation_coco"
    
    print("📁 Dataset COCO: " + dataset_path)
    
    if not os.path.exists(dataset_path):
        print("❌ Error: No se encontró el dataset: " + dataset_path)
        print("💡 Ejecute desde training/ y asegúrese de que existe:")
        print("   ../datasets/segmentation_coco/")
        return
    
    # Verificar estructura del dataset
    if os.path.exists(os.path.join(dataset_path, "annotations")):
        # Estructura: dataset/annotations/, dataset/images/
        register_coco_instances(
            "dental_dataset_train",
            {},
            os.path.join(dataset_path, "annotations", "instances_train.json"),
            os.path.join(dataset_path, "images")
        )
        register_coco_instances(
            "dental_dataset_val",
            {},
            os.path.join(dataset_path, "annotations", "instances_val.json"),
            os.path.join(dataset_path, "images")
        )
    else:
        # Estructura: dataset/train/, dataset/val/, dataset/test/
        for split in ["train", "val", "test"]:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                register_coco_instances(
                    "dental_dataset_" + split,
                    {},
                    os.path.join(split_path, "annotations.json"),
                    split_path
                )

def main():
    if not DETECTRON2_AVAILABLE:
        print("🚫 No se puede entrenar segmentación sin detectron2")
        print("💡 Instala detectron2 primero con: python install_detectron2.py")
        sys.exit(1)
        
    setup_logger()
    print("🦷 Iniciando entrenamiento de segmentación para dental_dataset")
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"🚀 Usando GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ Usando CPU - el entrenamiento será más lento")
    
    # Registrar datasets
    register_datasets()
    
    # Configurar entrenamiento
    cfg = setup_config()
    
    # Crear trainer y entrenar
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print("✅ Entrenamiento completado")
    print(f"📁 Modelo guardado en: {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
