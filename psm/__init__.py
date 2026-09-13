"""PCFC electrode microstructure analysis pipeline."""
import warnings

# scikit-image 0.26 renamed remove_small_objects' `min_size` kwarg. We keep
# `min_size` (its semantics are what we want) and silence the rename notice so
# output stays clean across versions. requirements.txt pins <0.26 anyway.
warnings.filterwarnings(
    "ignore", message=".*min_size.*", category=FutureWarning
)

__version__ = "0.1.0"

from . import segmentation, psd, dream3d, tortuosity, boundaries  # noqa: E402,F401
from . import batch, stats  # noqa: E402,F401