"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image
from PIL import ImageFile

import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.caption_datasets import __DisplMixin
import torchvision.transforms as transforms
import torch

EXT = ['.jpg', '.jpeg', '.png']

class TextInvDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, real_dir, fake_dir, real_label, fake_label):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # inherit from what BaseDataset do
        self.annotation = []

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.PIL2Tensor = transforms.PILToTensor()

        real_paths = []
        for root, dirs, files in os.walk(real_dir):
            for file in files:
                if(os.path.splitext(file)[-1] in EXT):
                    real_paths.append(os.path.join(root, file))
        real_labels = [real_label] * len(real_paths)
        
        fake_paths = []
        for root, dirs, files in os.walk(fake_dir):
            for file in files:
                if(os.path.splitext(file)[-1] in EXT):
                    fake_paths.append(os.path.join(root, file))
        fake_labels = [fake_label] * len(fake_paths)
                    
        self.path_and_labels = pd.DataFrame({
            "img_path": real_paths + fake_paths,
            "label": real_labels + fake_labels
        })
        self.path_and_labels.set_index("img_path", inplace=True)
        
        # self.image_root = vis_root

        # for ann_path in ann_paths:
        #     self.path_and_labels = pd.read_csv(ann_path, index_col="img_path")

    def __len__(self):
        # overwrite __len__() in BaseDataset
        return len(list(self.path_and_labels.index))

    def __getitem__(self, index):

        image_path = list(self.path_and_labels.index)[index]
        rawImage = Image.open(image_path).convert("RGB")
        
        image = self.vis_processor(rawImage)
        
        groundTruth = self.path_and_labels.loc[image_path, "label"]

        return {
            "image": image,
            "text_input": self.text_processor.prompt,
            "text_output": groundTruth,
            "text_input_freeze": self.text_processor.prompt_freeze,
        }