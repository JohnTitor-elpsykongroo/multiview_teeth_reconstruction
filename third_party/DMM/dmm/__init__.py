"""Static dual-arch interfaces implemented in the DMM source tree."""
from pathlib import Path
import sys

DMM_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_VERSION = "independent_arch_v1"
_dependencies = str(DMM_ROOT / "third_party")
if _dependencies not in sys.path:
    sys.path.insert(0, _dependencies)

CONTRACT_ID = "dual_arch_static_semantic"
VERSION = "1.0.0"
PROFILE = "coupled_component_dmm"
MODEL_UNIT_MM = 50.0
TEETH = {
    "upper": tuple(range(11, 18)) + tuple(range(21, 28)),
    "lower": tuple(range(31, 38)) + tuple(range(41, 48)),
}
CHANNEL_FDI = TEETH["upper"] + TEETH["lower"]
