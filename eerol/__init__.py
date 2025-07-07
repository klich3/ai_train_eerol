"""
🔧 EEROL - Universal Dataset Management Tool
============================================

Universal toolkit for computer vision dataset management.
"""

__version__ = "3.0"
__author__ = "Universal Tool"

from .dataset_scanner import DatasetScanner
from .dataset_converter import DatasetConverter
from .dataset_preview import DatasetPreview
from .dataset_splitter import DatasetSplitter
from .script_generator import ScriptGenerator
from .utils import EerolUtils

__all__ = [
    'DatasetScanner',
    'DatasetConverter', 
    'DatasetPreview',
    'DatasetSplitter',
    'ScriptGenerator',
    'EerolUtils'
]
