from .ResultMerge import mergebypoly, mergebyrec
from .dota_evaluation_task1 import voc_eval as dota_eval_task1
from .dota_evaluation_task2 import voc_eval as dota_eval_task2


__all__ = [
    "mergebypoly", "mergebyrec", "dota_eval_task1", "dota_eval_task2"
]
