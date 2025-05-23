"""
Domain-Specific Evaluation Modules for OpenEvals
===============================================

This package contains specialized evaluation modules for various scientific domains:
- Biology: Protein structure, genomics, drug discovery
- Physics: Fusion energy, materials science, plasma physics
- Virology: Outbreak modeling, viral genomics
- Security: AI safety, robustness testing
- Multi-modal: Image, audio, video evaluations

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .biology import *
from .physics import *
from .security import *
from .multimodal import *

__all__ = [
    # Biology
    'ProteinStructureEvaluator',
    'GenomicsEvaluator', 
    'DrugDiscoveryEvaluator',
    'ViralGenomicsEvaluator',
    
    # Physics
    'FusionPhysicsEvaluator',
    'MaterialsScienceEvaluator',
    'PlasmaPhysicsEvaluator',
    
    # Security
    'AdversarialRobustnessEvaluator',
    'AISecurityEvaluator',
    'BiasDetectionEvaluator',
    
    # Multi-modal
    'MultiModalEvaluator',
    'ImageEvaluator',
    'AudioEvaluator',
    'VideoEvaluator'
] 