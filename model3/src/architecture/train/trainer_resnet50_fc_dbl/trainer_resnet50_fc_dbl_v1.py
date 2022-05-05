from matplotlib.lines import Line2D
from architecture.train import TrainerDefault
from architecture.model.model_resnet50_fc_dbl import Net
from torchvision import transforms
from torch.optim import SGD
import torch
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import LinearLR
import os
import os.path as osp

class Trainer_ResNet50_FC_dbl_v1(TrainerDefault):
    model = Net()
    # Training parameters
    learning_rate = 0.01
    epoch_num = 30
    batch_size = 8
    batches_per_step = 1

    num_workers = 2

    # Training moderators
    optimizer = SGD
    optimizer_kwds = {
        'momentum': 0.2,
        'weight_decay': 0
    }

    warmup_epochs = 1
    warmup_scheduler = LinearLR
    warmup_scheduler_kwds = {
        'start_factor': 1e-9,
        'end_factor': 1.
    }

    regular_scheduler = StepLR
    regular_scheduler_kwds = {
        'step_size': 7,
        'gamma': 0.3
    }
    dataset_train_kwds = {
        'transform': transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.25, 1.00)),
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
        self.scheduler = self.warmup_scheduler(
            self.optimizer, 
            total_iters=self.warmup_epochs * \
                int(len(self.dataset_train) \
                    / (self.batches_per_step*self.batch_size)),
            **self.warmup_scheduler_kwds)

    def train(self):
        assert self.model is not None, "TrainerDefault cannot be used on it's" \
            " own. Custom Trainer should either inherit TrainerDefault and" \
            " specify Trainer.model or should be defined from scratch."
        
        self.initialize()
        
        self.model.to(self.device[0])

        self.output_path_parameters = osp.join(self.output_dir, 'parameters')
        self.output_path_losses_train = osp.join(self.output_dir, 
                                                 'losses_train.csv')
        self.output_path_losses_val = osp.join(self.output_dir, 
                                               'losses_val.csv')
        self.output_path_losses_img = osp.join(self.output_dir, 
                                               'losses_img.png')

        print(self.model)

        self.log_message('Initial validation.')
        self.validation_epoch(epoch=0)

        for epoch in range(self.epoch_num):
            if epoch == self.warmup_epochs:
                self.scheduler = self.regular_scheduler(
                    self.optimizer, **self.regular_scheduler_kwds)

            self.log_message(f'Training. Epoch {epoch+1}. '
                f'lr={self.scheduler.get_last_lr()}')
            self.training_epoch(epoch)
            
            self.log_message(f'Validation. Epoch {epoch+1}.')
            self.validation_epoch(epoch=epoch+1)

            self.handle_saving(epoch)
        
        print(f'Training finished. All outputs saved to {self.output_dir}.')

    def training_epoch(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()
        self.model.feature_extractor.eval()
        if self.RUN_MODE == 'd':
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
                    print(f'--LR:', self.scheduler.get_last_lr())
                    print(f'--Grad norm / num_prms:')
                    print(f'---Feature extractor:',
                        self.model.get_grad_norm(
                            self.model.feature_extractor))
                    print(f'---Fully connected per image:',
                        self.model.get_grad_norm(
                            self.model.fully_connected_layers_per_image))
                    print(f'---Fully connected final:',
                        self.model.get_grad_norm(
                            self.model.fully_connected_layers_final))
                loss_running = torch.tensor(0., device=self.device[0])

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if epoch < self.warmup_epochs:
                    self.scheduler.step()

        if epoch >= self.warmup_epochs:
            self.scheduler.step()

