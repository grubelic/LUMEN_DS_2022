from architecture.train import TrainerDefault
from architecture.model.model_resnet50_fc3 import Net
from torchvision import transforms
from torch.optim import SGD
import torch
from torch.optim.lr_scheduler import StepLR

class Trainer_ResNet50_FC3_Overfit_v5(TrainerDefault):
    model = Net()
    # Training parameters
    learning_rate = 0.01
    epoch_num = 100
    batch_size = 2
    batches_per_step = 8

    num_workers = 2

    # Training moderators
    optimizer = SGD
    optimizer_kwds = {
        'momentum': 0.0,
        'weight_decay': 0
    }

    scheduler = StepLR
    scheduler_kwds = {
        'step_size': 20,
        'gamma': 0.3
    }
    dataset_train_kwds = {
        'transform': transforms.Compose([
            #transforms.RandomResizedCrop((224, 224), scale=(0.25, 1.00)),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'image_directions': ['N', 'E', 'S', 'W']
    }
    dataset_val_kwds = {
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'image_directions': ['N', 'E', 'S', 'W']
    }

    def initialize(self):
        super().initialize()
        self.model.freeze_part(self.model.feature_extractor)

    def training_epoch(self):
        self.optimizer.zero_grad()
        self.model.train()
        self.model.feature_extractor.eval()
        print('Feature extractor set to eval mode.')

        loss_running = torch.tensor(0., device=self.device[0])

        for batch_ind, batch in enumerate(self.dataloader_train):
            input = {
                'image_N': batch['image_N'].to(self.device[0]),
                'image_E': batch['image_E'].to(self.device[0]),
                'image_S': batch['image_S'].to(self.device[0]),
                'image_W': batch['image_W'].to(self.device[0])
            }
            
            output_gt = batch['output'].to(self.device[0])
            output_train = self.model(input)
            loss_batch = self.criterion(output_train, output_gt)
            loss_batch /= self.batches_per_step*self.batch_size
            loss_batch.backward()
            loss_running += loss_batch.clone().detach()

            if (batch_ind+1)%self.batches_per_step == 0:
                # loss_running /= (self.batch_size*self.batches_per_step)
                self.losses_train.append(float(loss_running))
                if self.RUN_MODE == 'd':
                    print(f'--Train loss: {loss_running}')
                loss_running = torch.tensor(0., device=self.device[0])

                self.optimizer.step()
                self.optimizer.zero_grad()

        self.scheduler.step()

