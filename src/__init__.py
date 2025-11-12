"""
FEWS: Famine Early Warning System
"""

from .fews_system import FEWSSystem
from .ipc_parser import IPCParser, RegionRiskAssessment
from .document_processor import DocumentProcessor

__all__ = [
    'FEWSSystem',
    'IPCParser',
    'RegionRiskAssessment',
    'DocumentProcessor',
]

__version__ = '1.0.0'

