#!/usr/bin/env python3
"""
🛠️ INSTALADOR DE DETECTRON2 para Dental AI
===========================================

Script para instalar detectron2 de forma segura según el sistema.
"""

import subprocess
import sys
import platform
import torch

def check_torch_version():
    """Verifica la versión de PyTorch."""
    print(f"🔍 PyTorch versión: {torch.__version__}")
    print(f"🔍 CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔍 CUDA versión: {torch.version.cuda}")
    return torch.__version__, torch.cuda.is_available()

def install_detectron2():
    """Instala detectron2 según el sistema."""
    torch_version, cuda_available = check_torch_version()
    
    print("\n🚀 Instalando detectron2...")
    
    try:
        if cuda_available:
            # Instalar versión con CUDA
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "detectron2", "-f", 
                "https://dl.fbaipublicfiles.com/detectron2/wheels/index.html"
            ]
        else:
            # Instalar versión CPU
            cmd = [
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/facebookresearch/detectron2.git"
            ]
        
        print(f"💻 Ejecutando: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        print("✅ Detectron2 instalado exitosamente!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando detectron2: {e}")
        print("\n📝 Instalación manual:")
        print("1. Para CUDA: pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/index.html")
        print("2. Para CPU: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        return False
        
    return True

def test_detectron2():
    """Prueba que detectron2 funcione."""
    try:
        import detectron2
        from detectron2.utils.logger import setup_logger
        setup_logger()
        print(f"✅ Detectron2 versión: {detectron2.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Error importando detectron2: {e}")
        return False

def main():
    print("🦷 INSTALADOR DE DETECTRON2 - Dental AI")
    print("=" * 50)
    
    # Verificar PyTorch
    check_torch_version()
    
    # Intentar importar detectron2
    try:
        import detectron2
        print(f"✅ Detectron2 ya está instalado: {detectron2.__version__}")
        return
    except ImportError:
        print("⚠️ Detectron2 no está instalado")
    
    # Instalar detectron2
    if install_detectron2():
        # Probar instalación
        test_detectron2()
    else:
        print("❌ Falló la instalación automática")
        print("💡 Consulta la documentación oficial: https://detectron2.readthedocs.io/")

if __name__ == "__main__":
    main()
