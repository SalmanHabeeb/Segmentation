from data import RiverSegDataset
import os
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Metric
from torchvision.models import resnet50, ResNet50_Weights

def get_pretrained_backbone():
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    for name, layer in backbone.named_children():
       if name != "fc":
           layer.requires_grad = False
    backbone.fc = nn.Identity()
    return backbone

def get_dataset(train_paths, val_paths, img_tfms, mask_tfms) -> Dataset:
    dataset = {}
    dataset["train"] = RiverSegDataset(train_paths["IMAGE_PATH"], train_paths["MASK_PATH"], img_tfs, mask_tfs, validation_split=0)
    dataset["val"] = RiverSegDataset(val_paths["IMAGE_PATH"], val_paths["MASK_PATH"], img_tfs, mask_tfs, training=False, validation_split=1)
    return dataset

def get_dataloaders(dataset):
    dataloaders = {}
    dataloaders["train"] = DataLoader(dataset["train"], shuffle=True, batch_size=64)
    dataloaders["val"] = DataLoader(dataset["val"], shuffle=True, batch_size=64)
    return dataloaders

class DiceScore(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("dice_score", default=torch.tensor(0), dist_reduce_fx=None)
    
    def update(self, preds: Tensor, targets: Tensor) -> None:
        # preds, targets = self._input_format(preds, target)
        preds, targets = torch.flatten(preds), torch.flatten(targets)
        assert preds.shape == targets.shape
        self.intersect = (preds*targets).sum()
        self.union = (torch.abs(preds) + torch.abs(targets)).sum()
    
    def compute(self) -> Tensor:
        return 2*self.intersect/self.union
