# All trainer classes should be imported here. Script main.py loads trainer
# from here. 

from .trainer_default import TrainerDefault
from .trainer_debug.trainer_debug_visualization import Trainer_DebugVisualization
from .trainer_debug.trainer_debug_v1 import Trainer_Debug_V1
from .trainer_resnet50_fc10.trainer_resnet50_fc10_overfit_v1 import Trainer_ResNet50_FC10_Overfit_v1
from .trainer_resnet50_fc10.trainer_resnet50_fc10_overfit_v2 import Trainer_ResNet50_FC10_Overfit_v2