"""Stage two: RGB observations, separate from the frozen DMM implementation."""

VERSION = "1.0.0"
FDI_IDS = tuple(q * 10 + t for q in range(1, 5) for t in range(1, 9))
CLASS_IDS = (0,) + FDI_IDS
