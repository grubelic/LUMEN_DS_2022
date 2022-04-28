# All trainer classes should be imported here. Script main.py loads trainer
# from here. 

from .trainer_default import TrainerDefault
from .trainer_debug.trainer_debug_visualization import Trainer_DebugVisualization
from .trainer_debug.trainer_debug_v1 import Trainer_Debug_V1
from .trainer_resnet50_fc10.trainer_resnet50_fc10_overfit_v1 import Trainer_ResNet50_FC10_Overfit_v1
from .trainer_resnet50_fc10.trainer_resnet50_fc10_overfit_v2 import Trainer_ResNet50_FC10_Overfit_v2
from .trainer_resnet50_fc10.trainer_resnet50_fc10_overfit_v3 import Trainer_ResNet50_FC10_Overfit_v3
from .trainer_resnet50_fc10.trainer_resnet50_fc10_overfit_v4 import Trainer_ResNet50_FC10_Overfit_v4
from .trainer_resnet50_fc10.trainer_resnet50_fc10_overfit_v5 import Trainer_ResNet50_FC10_Overfit_v5
from .trainer_resnet50_fc10.trainer_resnet50_fc10_overfit_v6 import Trainer_ResNet50_FC10_Overfit_v6

from .trainer_resnet50_fc3.trainer_resnet50_fc3_overfit_v1 import Trainer_ResNet50_FC3_Overfit_v1
from .trainer_resnet50_fc3.trainer_resnet50_fc3_overfit_v2 import Trainer_ResNet50_FC3_Overfit_v2
from .trainer_resnet50_fc3.trainer_resnet50_fc3_overfit_v3 import Trainer_ResNet50_FC3_Overfit_v3
from .trainer_resnet50_fc3.trainer_resnet50_fc3_overfit_v4 import Trainer_ResNet50_FC3_Overfit_v4
from .trainer_resnet50_fc3.trainer_resnet50_fc3_overfit_v5 import Trainer_ResNet50_FC3_Overfit_v5
from .trainer_resnet50_fc3.trainer_resnet50_fc3_overfit_v6 import Trainer_ResNet50_FC3_Overfit_v6
from .trainer_resnet50_fc3.trainer_resnet50_fc3_v1 import Trainer_ResNet50_FC3_v1

from .trainer_resnet50_fc5.trainer_resnet50_fc5_overfit_v1 import Trainer_ResNet50_FC5_Overfit_v1
from .trainer_resnet50_fc5.trainer_resnet50_fc5_v1 import Trainer_ResNet50_FC5_v1
