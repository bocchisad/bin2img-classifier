"""bin2img — Binary Visualizer & Malware Family Classifier."""

from bin2img.api import create_app
from bin2img.constants import __version__
from bin2img.features import FEATURE_NAMES, FeatureExtractor, FeatureVector
from bin2img.model import UNKNOWN_LABEL, MalwareFamilyClassifier, PredictionResult
from bin2img.parser import BinaryMetadata, BinaryParser, SectionInfo, calculate_entropy
from bin2img.visualizer import BinaryVisualizer, VisualizationResult, resolve_image_width

__all__ = [
    "FEATURE_NAMES",
    "BinaryMetadata",
    "BinaryParser",
    "BinaryVisualizer",
    "FeatureExtractor",
    "FeatureVector",
    "UNKNOWN_LABEL",
    "MalwareFamilyClassifier",
    "PredictionResult",
    "SectionInfo",
    "VisualizationResult",
    "__version__",
    "calculate_entropy",
    "create_app",
    "resolve_image_width",
]
