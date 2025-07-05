#!/usr/bin/env python3
"""
🧹 Script para limpiar conflictos en la estructura dental-ai
"""

import sys
from pathlib import Path

# Agregar ruta de módulos
sys.path.append(str(Path(__file__).parent / "Src"))

def limpiar_conflictos():
    """🧹 Limpia archivos que entran en conflicto con directorios."""
    output_path = Path("Dist/dental_ai")
    
    print("🧹 LIMPIANDO CONFLICTOS EN ESTRUCTURA DENTAL-AI...")
    print(f"📂 Directorio: {output_path}")
    
    archivos_conflictivos = []
    
    # Buscar archivos que puedan entrar en conflicto
    if output_path.exists():
        for item in output_path.rglob("*"):
            if item.is_file():
                # Verificar si el nombre del archivo podría ser un directorio
                parent_dir = item.parent
                potential_conflict = parent_dir / item.stem  # Sin extensión
                
                # Casos específicos conocidos
                if item.name in ['models', 'utils', 'scripts', 'configs']:
                    archivos_conflictivos.append(item)
                    print(f"⚠️ Archivo conflictivo encontrado: {item}")
    
    if archivos_conflictivos:
        print(f"\n📋 Archivos conflictivos encontrados: {len(archivos_conflictivos)}")
        for archivo in archivos_conflictivos:
            print(f"   • {archivo}")
        
        respuesta = input("\n❓ ¿Deseas eliminar estos archivos conflictivos? (s/N): ").strip().lower()
        
        if respuesta in ['s', 'si', 'sí', 'yes', 'y']:
            for archivo in archivos_conflictivos:
                try:
                    archivo.unlink()
                    print(f"✅ Eliminado: {archivo}")
                except Exception as e:
                    print(f"❌ Error al eliminar {archivo}: {e}")
            print(f"\n🎉 Limpieza completada!")
        else:
            print("❌ Limpieza cancelada")
    else:
        print("✅ No se encontraron conflictos")

def verificar_estructura():
    """🔍 Verifica la estructura actual."""
    output_path = Path("Dist/dental_ai")
    
    print(f"\n🔍 ESTRUCTURA ACTUAL:")
    print(f"="*40)
    
    if not output_path.exists():
        print("❌ El directorio Dist/dental_ai no existe")
        return
    
    for item in sorted(output_path.iterdir()):
        if item.is_dir():
            print(f"📁 {item.name}/")
            # Mostrar subdirectorios
            try:
                subdirs = [d for d in item.iterdir() if d.is_dir()]
                for subdir in sorted(subdirs):
                    print(f"   📁 {subdir.name}/")
            except PermissionError:
                print(f"   ⚠️ Sin permisos para leer")
        else:
            print(f"📄 {item.name}")

def main():
    """🎯 Función principal."""
    print("🧹 LIMPIADOR DE CONFLICTOS DENTAL-AI")
    print("="*40)
    
    verificar_estructura()
    print()
    limpiar_conflictos()
    print()
    verificar_estructura()

if __name__ == "__main__":
    main()
