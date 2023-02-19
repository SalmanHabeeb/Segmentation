
import os
from typing import List, Tuple, Dict
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import ToTensor, Resize, Normalize, Grayscale

class RiverSegDataset(Dataset):
    def __init__(self, image_path: str, mask_path: str, image_transform=None, mask_transform=None, training: bool =True, validation_split: float=0.2, seed: int=1,):
        super().__init__()
        self.X_path = image_path
        self.y_path = mask_path
        self.X_transform = image_transform
        self.y_transform = mask_transform

        np.random.seed(seed)
        all_X, all_y = self.retrieve_Xy(self.X_path, self.y_path)
        mask = np.random.uniform(0, 1, len(all_X))
        mask = mask > validation_split
        
        self.X = np.array(all_X)
        self.y = np.array(all_y)

        self.X = self.X[mask] if training else self.X[~mask]
        self.y = self.y[mask] if training else self.y[~mask]

        assert len(self.X) > 0, "Image directory is empty or split is not done properly"
        assert len(self.X) == len(self.y), f"Images and targets have different no. of files {len(self.X)=}, {len(self.y)=}"

        print(f"Found {len(self)} files in each of image and target directory.")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        X = read_image(self.X[idx], ImageReadMode.RGB).float()
        # X = ToTensor()(X)
        if self.X_transform:
            X = self.X_transform(X)
        y = read_image(self.X[idx], ImageReadMode.RGB).float()
        # y = ToTensor()(y)
        if self.y_transform:
            y = self.y_transform(y)
        return X, y

    @classmethod
    def get_path(self, dir:str, file_:str) -> str:
        return os.path.join(dir, file_)

    @classmethod
    def retrieve_Xy(self, path1:str, path2:str) -> Tuple[List[str]]:
        files_path1 = []
        files_path2 = []
        counter = 0
        for file_ in os.listdir(path1):
            if file_.split(os.extsep)[0] + ".png" in os.listdir(path2):
                files_path1.append(self.get_path(path1, file_))
                files_path2.append(self.get_path(path2, file_.split(os.extsep)[0] + ".png"))
            else:
                counter += 1
                print(file_)
        print(counter, "uncommon files in image and target data.")
        return files_path1, files_path2

class Transform:
    def __init__(self, resize, normalize):
        self.resize = resize
        self.normalize = normalize
    def __call__(self, x: Tensor) -> Tensor:
        if self.resize:
            result = Resize(self.resize)(x)
        if self.normalize:
            result = Normalize([0., 0., 0.,], [255., 255., 255.])(result)
        return result

class ImageTransform(Transform):
    def __init__(self, resize=None, normalize=True):
        super().__init__(resize, normalize)

class MaskTransform(Transform):
    def __init__(self, resize=None, normalize=True, convert2gray=False):
        super().__init__(resize, normalize)
        self.convert2gray = convert2gray
    def __call__(self, x: Tensor) -> Tensor:
        if self.resize:
            result = Resize(self.resize)(x)
        if self.convert2gray:
            result = Grayscale(num_output_channels=1)(result)
        if self.normalize:
            result = Normalize([0.,], [255.,])(result)
        return result
