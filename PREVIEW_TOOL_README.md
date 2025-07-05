# 🔍 Herramienta de Previsualización de Datasets Dentales

Una herramienta completa para visualizar y validar anotaciones en datasets dentales que soporta múltiples formatos.

## ✨ Características

- **Múltiples formatos**: YOLO, COCO, CSV, JSON
- **Detección automática** de formato por extensión
- **Previsualización interactiva** con menús fáciles de usar
- **Procesamiento por lotes** para múltiples imágenes
- **Soporte para clases dentales** específicas
- **Guardado de resultados** con anotaciones visuales
- **Integración completa** con el sistema principal

## 🚀 Formas de Uso

### 1. Línea de Comandos (Rápido)

```bash
# Uso básico
python dataset_preview_tool.py imagen.jpg anotaciones.txt

# Especificar formato
python dataset_preview_tool.py imagen.jpg anotaciones.txt --format yolo

# Con archivo de clases
python dataset_preview_tool.py imagen.jpg anotaciones.txt --classes classes.txt

# Guardar resultado
python dataset_preview_tool.py imagen.jpg anotaciones.txt --output resultado.jpg
```

### 2. Modo Interactivo (Recomendado)

```bash
python preview_interactive.py
```

### 3. Desde el Sistema Principal

```bash
python main.py
# Selecciona opción 15: Previsualizar anotaciones de datasets
```

### 4. Demo con Datos de Ejemplo

```bash
python demo_preview_tool.py
```

## 📊 Formatos Soportados

### 🎯 YOLO (.txt)
```
# Formato: class_id x_center y_center width height
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```
- Coordenadas normalizadas (0-1)
- Una línea por objeto
- Ideal para detección de objetos

### 🎯 COCO (.json)
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 50, 75],
      "segmentation": [...]
    }
  ],
  "categories": [...]
}
```
- Formato estándar de COCO
- Soporte para segmentación
- Coordenadas en píxeles

### 🎯 CSV (.csv)
```csv
filename,class,xmin,ymin,xmax,ymax
imagen.jpg,caries,100,100,150,175
imagen.jpg,tooth,200,150,250,225
```
- Formato tabular simple
- Una fila por objeto
- Coordenadas en píxeles

### 🎯 JSON Personalizado (.json)
```json
{
  "imagen.jpg": [
    {
      "class": "caries",
      "bbox": [100, 100, 50, 75]
    }
  ]
}
```
- Formato flexible
- Personalizable para proyectos específicos

## 🦷 Clases Dentales Soportadas

La herramienta reconoce automáticamente estas clases dentales comunes:

- `caries` / `cavity` - Caries dental
- `tooth` / `teeth` - Dientes
- `filling` / `restoration` - Empastes
- `crown` - Coronas
- `implant` - Implantes
- `root_canal` - Endodoncia
- `bone_loss` - Pérdida ósea
- `impacted` - Dientes impactados
- `periapical_lesion` - Lesiones periapicales
- `maxillary_sinus` - Seno maxilar
- `mandible` - Mandíbula
- `maxilla` - Maxilar

## 🎨 Ejemplos Visuales

### Antes (Imagen Original)
- Imagen de rayos X dental sin anotaciones

### Después (Con Anotaciones)
- Rectángulos de colores alrededor de objetos detectados
- Etiquetas de clase claramente visibles
- Diferentes colores para cada clase
- Información estadística mostrada

## 📋 Opciones de Línea de Comandos

```bash
python dataset_preview_tool.py imagen anotaciones [opciones]

Argumentos requeridos:
  imagen        Ruta a la imagen
  anotaciones   Ruta al archivo de anotaciones

Opciones:
  --format, -f  {yolo,coco,csv,json,auto}  Formato de anotaciones (default: auto)
  --classes, -c                            Archivo de clases (opcional)
  --output, -o                             Guardar imagen anotada
  --help, -h                               Mostrar ayuda
```

## 🔄 Modo Interactivo - Opciones

1. **📸 Previsualizar imagen individual**
   - Seleccionar imagen y anotaciones
   - Detectar formato automáticamente
   - Configurar opciones avanzadas

2. **📁 Previsualizar múltiples imágenes**
   - Procesar carpeta completa
   - Procesamiento por lotes
   - Generar múltiples resultados

3. **📊 Análizar dataset completo** *(En desarrollo)*
   - Estadísticas de todo el dataset
   - Distribución de clases
   - Métricas de calidad

## 🛠️ Instalación y Dependencias

```bash
# Instalar dependencias requeridas
pip install opencv-python matplotlib numpy pandas

# Para COCO específicamente
pip install pycocotools

# Ejecutar herramienta
python dataset_preview_tool.py --help
```

## 💡 Consejos de Uso

### ✅ Mejores Prácticas

1. **Verificar rutas**: Asegúrate de que las rutas de imagen y anotaciones sean correctas
2. **Formato correcto**: Verifica que el formato de anotaciones coincida con el especificado
3. **Archivo de clases**: Usa archivo de clases para mejores etiquetas
4. **Coordenadas**: Verifica que las coordenadas estén en el rango correcto

### ⚠️ Problemas Comunes

1. **Imagen no encontrada**: Verifica la ruta de la imagen
2. **Anotaciones vacías**: El archivo puede no tener el formato correcto
3. **Coordenadas fuera de rango**: Las coordenadas pueden estar mal normalizadas
4. **Formato no detectado**: Especifica el formato manualmente

### 🔧 Solución de Problemas

```bash
# Verificar que la imagen se carga correctamente
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Verificar archivo de anotaciones
head -5 anotaciones.txt

# Ejecutar con detección automática
python dataset_preview_tool.py imagen.jpg anotaciones.txt --format auto
```

## 🧪 Ejecutar Demo

Para probar la herramienta con datos de ejemplo:

```bash
# Demo completa con datos generados
python demo_preview_tool.py

# Esto creará:
# - demo_dataset/images/dental_xray_001.jpg (imagen ejemplo)
# - demo_dataset/labels/dental_xray_001.txt (anotaciones YOLO)
# - demo_dataset/classes.txt (archivo de clases)
# - demo_preview_result.jpg (resultado con anotaciones)
```

## 🔗 Integración con Sistema Principal

La herramienta está completamente integrada con el sistema principal:

1. **Acceso desde menú**: Opción 15 en `main.py`
2. **Detección automática**: Busca datasets existentes
3. **Formatos compatibles**: Funciona con todos los formatos del sistema
4. **Resultados persistentes**: Guarda resultados en estructura del proyecto

## 📈 Casos de Uso

### 🔍 Validación de Datos
- Verificar que las anotaciones están correctas
- Detectar errores en coordenadas
- Validar clases asignadas

### 🎯 Control de Calidad
- Revisar datasets antes del entrenamiento
- Verificar consistencia en anotaciones
- Detectar objetos mal anotados

### 📊 Análisis Visual
- Entender distribución de clases
- Visualizar patrones en datos
- Preparar reportes visuales

### 🔄 Depuración
- Diagnosticar problemas en datasets
- Verificar conversiones de formato
- Comparar diferentes versiones

---

## 📞 Soporte

Si encuentras problemas o tienes sugerencias:

1. Verifica que todas las dependencias están instaladas
2. Ejecuta la demo para verificar funcionamiento básico
3. Revisa la documentación de formatos soportados
4. Usa el modo interactivo para configuración guiada

¡La herramienta está diseñada para ser intuitiva y robusta! 🚀
