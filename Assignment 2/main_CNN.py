"""
Author: adopted from Ricardo Eugenio González Valenzuela by Oliver Gurney-Champion | Spring 2023
Date modified: March 2023

Skin cancer is the most common cancer globally, with melanoma being the most deadly form. Dermoscopy is a skin imaging
modality that has demonstrated improvement for diagnosis of skin cancer compared to unaided visual inspection. However,
clinicians should receive adequate training for those improvements to be realized. In order to make expertise more
widely available, the International Skin Imaging Collaboration (ISIC) has developed the ISIC Archive, an international
repository of dermoscopic images, for both the purposes of clinical training, and for supporting technical research
toward automated algorithmic analysis by hosting the ISIC Challenges.

The dataset is already pre-processed for you in batches of 128x256x256x3 (128 images, of size 256x256 rgb)
"""

#loading libraries

import os
import numpy as np
import argparse
import glob

import torch
import torchmetrics
import torch.nn.functional as F
from torchvision import transforms
from sys import platform
from Data_loader import Scan_Dataset, Scan_DataModule, Random_Rotate
from visualization import show_data, show_data_logger
from CNNs import SimpleConvNet
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

#start interactieve sessie om wandb.login te runnen
wandb.login()

#seeding for reproducible results
SEED=42
#torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
np.random.RandomState(SEED)

if platform == "linux" or platform == "linux2":
    data_dir = '/gpfs/work5/0/prjs1312/data/classification'
else:
    #set data location on your local computer. Data can be downloaded from:
    # https://surfdrive.surf.nl/files/index.php/s/QWJUE37bHojMVKQ
    # PW: deeplearningformedicalimaging
    data_dir = 'C:\scratch\Surf\Documents\Onderwijs\DeepLearning_MedicalImaging\opgaven\opgave 2\AI-Course_StudentChallenge\data\classification'

print('data is loaded from ' + data_dir)
# view data
nn_set = 'train' # ['train', 'val', 'test']
index = 0

# study the effect of augmentation here!
dataset = Scan_Dataset(os.path.join(data_dir, nn_set))
show_data(dataset,index,n_images_display=5)
train_transforms = transforms.Compose([Random_Rotate(0.1), transforms.ToTensor()])
dataset = Scan_Dataset(os.path.join(data_dir, nn_set),transform = train_transforms)
show_data(dataset,index,n_images_display=5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)

models = {'custom_convnet': SimpleConvNet}

optimizers = {'adam': torch.optim.Adam,
              'sgd': torch.optim.SGD}

metrics = {'acc': torchmetrics.Accuracy('binary').to(device),
           'f1': torchmetrics.F1Score('binary').to(device),
           'precision': torchmetrics.Precision('binary').to(device),
           'recall': torchmetrics.Recall('binary').to(device)}


class Classifier(pl.LightningModule):
    def __init__(self, *args):
        super().__init__()
        self.save_hyperparameters()
        self.counter=0

        # defining model
        self.model_name = config['model_name']
        assert self.model_name in models, f'Model name "{self.model_name}" is not available. List of available names: {list(models.keys())}'
        self.model = models[self.model_name]().to(device)

        # assigning optimizer values
        self.optimizer_name = config['optimizer_name']
        self.lr = config['optimizer_lr']
        self.plot = False

    def step(self, batch, nn_set):
        X, y = batch
        X, y = X.float().to(device), y.to(device)
        y_hat = self.model(X).squeeze(1)
        y_prob = torch.sigmoid(y_hat)
        self.y_prob=y_prob

        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        self.log(f'{nn_set}_loss', loss, on_step=False, on_epoch=True)

        for metric_name, metric_fn in metrics.items():
            score = metric_fn(y_prob, y)
            self.log(f'{nn_set}_{metric_name}', score, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, 'train')
        if batch_idx == 0:
            fig = show_data_logger(batch,0,self.y_prob,n_images_display = 5, message = 'train_example')
            self.logger.log_image("train_example",[fig],step=self.counter)
        batch_dictionary = {'loss': loss}
        self.log_dict(batch_dictionary)
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, 'val')
        if batch_idx == 0:
            fig = show_data_logger(batch,0,self.y_prob,n_images_display = 5, message = 'test-example')
            self.logger.log_image("test_example",[fig],step=self.counter)
            self.counter = self.counter+1
        batch_dictionary = {'loss': loss}
        self.log_dict(batch_dictionary)
        return batch_dictionary

    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        epoch_dictionary = {'loss': avg_loss, 'log': {'loss': avg_loss}}
        return epoch_dictionary




    def test_step(self, batch, batch_idx):
        self.step(batch, 'test')

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        assert self.optimizer_name in optimizers, f'Optimizer name "{self.optimizer_name}" is not available. List of available names: {list(models.keys())}'
        return optimizers[self.optimizer_name](self.parameters(), lr=self.lr)


def run(config):
    logger = WandbLogger(name=config['experiment_name'], project='ISIC')
    data = Scan_DataModule(config)
    classifier = Classifier(config)
    logger.watch(classifier)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=config['checkpoint_folder_save'], monitor='val_f1', mode = 'max')
    trainer = pl.Trainer(accelerator=device, max_epochs=config['max_epochs'],
                         logger=logger, callbacks=[checkpoint_callback],
                         default_root_dir=config['bin'],
                         log_every_n_steps=1)
    trainer.fit(classifier, data)
    # load best model
    PATH = glob.glob(os.path.join(config['checkpoint_folder_save'], '*'))[0]
    model = Classifier.load_from_checkpoint(PATH)
    model.eval()

    # make test dataloader
    test_data = Scan_DataModule(config)

    # test model
    trainer = pl.Trainer()
    trainer.test(model, dataloaders=test_data, verbose=True)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    # Optimizer hyperparameters
    parser.add_argument('--optimizer_lr', default=0.1, type=float, nargs='+',
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Minibatch size')
    parser.add_argument('--model_name', default='custom_convnet', type=str,
                        help='defines model to use')
    parser.add_argument('--optimizer_name', default='sgd', type=str,
                        help='optimizer options: adam and sgd (default)')
    # Other hyperparameters
    parser.add_argument('--max_epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--experiment_name', default='test1', type=str,
                        help='name of experiment')
    parser.add_argument('--checkpoint_folder_path', default=False, type=str,
                        help='path of experiment to load')
    parser.add_argument('--checkpoint_folder_save', default=None, type=str,
                        help='path of experiment')
    args = parser.parse_args()
    config = vars(args)

    config.update({
        'train_data_dir': os.path.join(data_dir, 'train'),
        'val_data_dir': os.path.join(data_dir, 'val'),
        'test_data_dir': os.path.join(data_dir, 'test'),
        'bin': 'models/'})

    run(config)
    # Feel free to add any additional functions, such as plotting of the loss curve here
