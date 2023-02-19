from typing import List, Tuple, Dict
import os
import time
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import Tensor

class Trainer:
    def __init__(self,
        model,
        dataloaders,
        loss_func,
        metric,
        optimizer,
        scheduler,
        **kwargs,
    ):
        self.model = model
        self.dataloaders = dataloaders
        self.loss_func = loss_func
        self.metric = metric
        self.optimizer = optimizer(self.model.parameters(), lr=kwargs["learning_rate"])
        self.scheduler = scheduler(self.optimizer, factor=kwargs["reduce_lr_factor"], patience=kwargs["patience"], verbose=True)
        self.train_losses = []
        self.val_losses = []
        self.train_scores = []
        self.val_scores = []
        self.best_score = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())
    
    @staticmethod
    def print_epoch_start(epoch: int, n_epochs: int) -> None:
        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)

    def set_model_state(self, phase: str) -> None:
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
                
    def optimize_loss(self, inputs: Tensor, targets: Tensor, phase: int):
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(phase == 'train'):
            preds = self.model(inputs)
            loss = self.loss_func(preds, targets)

            if phase == 'train':
                loss.backward()
                self.optimizer.step()
         
        return preds, loss
       
    def get_mean_loss(self, loss, phase):
        return loss/len(self.dataloaders[phase].dataset)
    
    def get_mean_score(self, score, phase):
        return score/len(self.dataloaders[phase].dataset)
    
    def update_loss_and_score(self, loss, score, phase):
         if phase == "train":
             self.train_losses.append(loss)
             self.train_scores.append(score)
         else:
             self.val_losses.append(loss)
             self.val_scores.append(score)
         
    def train(self, n_epochs):
        start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        for epoch in range(n_epochs):
            self.print_epoch_start(epoch, n_epochs)

            for phase in ['train', 'val']:
                self.set_model_state(phase)
    
                running_loss = 0.0
                running_score = 0
    
                with tqdm(total=len(dataloaders[phase]), colour="#00FF00") as pbar:
                    for batch_idx, (inputs, targets) in enumerate(dataloaders[phase]):
                        pbar.set_description(f"Processing batch {batch_idx}")
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        
                        preds, loss = self.optimize_loss(inputs, targets, phase)
                        running_loss += loss.item() * inputs.size(0)
                        running_score += self.metric(preds, targets).item()*inputs.size(0)
                        
                        pbar.update(1)
    
                epoch_loss = self.get_mean_loss(running_loss, phase)
                epoch_score = self.get_mean_score(running_score, phase)
                self.update_loss_and_score(epoch_loss, epoch_score, phase)
                if phase == 'train':
                    self.scheduler.step(running_loss)
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Score: {epoch_score:.4f}')

                if phase == 'val' and epoch_score > self.best_score:
                    self.best_score = epoch_score
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
            print()
            #try:
                # if self.val_scores[-1] < self.val_scores[-10]:
                    # print(f"Early stopping at epoch {epoch + 1}")
                    # break
            # except IndexError:
                # pass
    
        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val score: {self.best_score:4f}')
    
        self.model.load_state_dict(self.best_model_wts)

    def plot_loss(self):
        fig, ax = plt.subplots(2, figsize=(7, 12))
        x = [i for i in range(1, len(self.train_losses) + 1)]
        for i, criterion in enumerate(["Loss", "Dice Score"]):
            if criterion == "Loss":
                ax[i].plot(x, self.train_losses)
                ax[i].plot(x, self.val_losses)
            else:
                ax[i].plot(x, self.train_scores)
                ax[i].plot(x, self.val_scores)
            ax[i].set_title(f"{criterion} Curves")
            ax[i].set_xlabel('Epochs')
            ax[i].set_ylabel(f'{criterion}')
            ax[i].legend(["Training", "Validation"])
        plt.savefig("losses", bbox_inches="tight")
        plt.show()

    def save(self, path):
        assert os.path.splitext(path)[-1] == ".pt", f"Invalid extension {os.path.splitext(path)[-1]:!r}"
        torch.save(self.model, path)
        print(f"Saved best weights to {path}")
