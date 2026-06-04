from .finger_tapping import FingerTappingPipeline
from .hand_open_close import HandOpenClosePipeline
from .hand_up_down import HandUpDownPipeline
from .both_still import BothStillPipeline
from .batch import BatchInferencePipeline

__all__ = [
    "FingerTappingPipeline",
    "HandOpenClosePipeline",
    "HandUpDownPipeline",
    "BothStillPipeline",
    "BatchInferencePipeline",
]
