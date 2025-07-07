# 🔧 EEROL - Universal Dataset Management Tool

**EEROL** is a universal tool for computer vision dataset management. It allows scanning, analyzing, converting, splitting, and training models with datasets in various formats.

## ✨ Features

- 🔍 **Automatic scanning** of datasets in any directory
- 📊 **Detailed analysis** of structure, formats, and categories
- 🔄 **Conversion** between formats (YOLO ↔ COCO ↔ Pascal VOC)
- ✂️ **Custom splitting** with configurable proportions (train/val/test)
- 🚀 **Automatic generation** of training scripts
- 👁️ **Annotation preview** on images
- 🧹 **Automatic cleanup** of unnecessary files
- 🎯 **Multi-format support** for YOLO, COCO, PyTorch, TensorFlow, U-Net

## 🚀 Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Make executable** (on Linux/macOS):
   ```bash
   chmod +x eerol.py
   ```

## 📋 Usage

### Interactive Mode

```bash
python eerol.py
```

### Command Line

#### Scan datasets

```bash
python eerol.py scan --path /path/to/datasets
python eerol.py scan  # Uses current directory or HOME
```

#### Convert format

```bash
python eerol.py convert --input-path /path/dataset --format yolo --name my_dataset
python eerol.py convert --input-path /path/dataset --format coco
```

#### Preview annotations

```bash
python eerol.py preview --image image.jpg --annotation annotation.txt --format yolo
python eerol.py preview --image image.jpg --annotation annotation.xml --format pascal_voc
```

#### Split dataset

```bash
python eerol.py split --input-path /path/dataset --train-ratio 0.7 --val-ratio 0.3
python eerol.py split --input-path /path/dataset --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2
```

#### List training datasets

```bash
python eerol.py list
```

#### Train model

```bash
python eerol.py train --dataset my_dataset
python eerol.py train  # Interactive selection
```

#### Clean files

```bash
python eerol.py clean
```

## 📁 Output Structure

EEROL generates datasets in the `Train/` folder with the following structure:

```
Train/
├── my_dataset_yolo/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── test/           # Optional
│   │   ├── images/
│   │   └── labels/
│   ├── data.yaml       # YOLO configuration
│   ├── split_info.yaml # Split information
│   └── train.py        # Training script
```

## 🎯 Supported Formats

### Input (Automatic Detection)

- **YOLO**: `.txt` + `data.yaml`
- **COCO**: `.json` with standard structure
- **Pascal VOC**: `.xml` with annotations

### Output (Conversion)

- **YOLO**: Standard structure with `data.yaml`
- **COCO**: JSON with images and annotations
- **Pascal VOC**: Individual XML per image

## 🚀 Training Scripts

EEROL automatically generates optimized training scripts:

- **YOLOv8**: Using ultralytics
- **COCO**: Base for Detectron2/MMDetection
- **PyTorch**: Customizable template
- **TensorFlow**: Base for TF Object Detection API
- **U-Net**: For semantic segmentation

## 🔧 Configuration

EEROL automatically creates:

- `~/.eerol/config.yaml`: Global configuration
- `Train/`: Generated datasets directory
- `Results/`: Results directory
- `Backups/`: Backup directory

## 📊 Complete Usage Example

1. **Scan** existing datasets:

   ```bash
   python eerol.py scan --path ~/datasets
   ```

2. **Convert** to YOLO:

   ```bash
   python eerol.py convert --input-path ~/datasets/my_coco_dataset --format yolo --name converted_yolo
   ```

3. **Split** with custom proportions:

   ```bash
   python eerol.py split --input-path Train/converted_yolo --train-ratio 0.8 --val-ratio 0.2 --name final_dataset
   ```

4. **Train** the model:
   ```bash
   python eerol.py train --dataset final_dataset
   ```

## 🛠️ Customization

### Add New Formats

Edit `eerol/dataset_converter.py` to add new conversion formats.

### Customize Training Scripts

Modify `eerol/script_generator.py` to add new frameworks or customize parameters.

### Add New Validations

Extend `eerol/utils.py` to add specific format validations.

## 🧹 Cleanup

EEROL can automatically clean:

- `__pycache__` files
- Temporary files
- Obsolete project files
- Framework caches

## ⚠️ Important Notes

- **Backup**: EEROL always preserves original datasets
- **Dependencies**: ML frameworks are installed on demand
- **Memory**: For large datasets, consider using SSD
- **GPU**: Scripts automatically detect GPU availability

## 🤝 Contributions

This is a refactored project from a tool specific to dental datasets, now converted into a universal tool. Contributions are welcome.

## 📄 License

Open-source project. See the license file for more details.

---

**EEROL makes computer vision dataset management simple and efficient!** 🚀
