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

class TextInvDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # inherit from what BaseDataset do
        self.annotation = []

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.PIL2Tensor = transforms.PILToTensor()

        self.image_root = vis_root

        for ann_path in ann_paths:
            self.path_and_labels = pd.read_csv(ann_path, index_col="img_path")

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
        }