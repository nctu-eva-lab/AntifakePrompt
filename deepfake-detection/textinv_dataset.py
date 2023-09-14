from PIL import Image
import pandas as pd
import numpy as np

from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

class TextInvDataset(Dataset):
    def __init__(self, csv, vis_processor=None, txt_processor=None):
        
        self.path_and_labels = pd.read_csv(csv, index_col="img_path")
        self.vis_processor = vis_processor
        self.txt_processor = txt_processor

    def __len__(self):
        
        return len(list(self.path_and_labels.index))

    def __getitem__(self, index):

        image_path = list(self.path_and_labels.index)[index]
        image = Image.open(image_path).convert("RGB")
        if self.vis_processor:
            image = self.vis_processors(image).unsqueeze(0).to(self.device)
        
        label = self.path_and_labels.loc[image_path, "label"]

        return image, label