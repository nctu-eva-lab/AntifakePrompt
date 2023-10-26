"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import warnings
import logging
import torch.distributed as dist

from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.textinv_datasets import TextInvDataset
import lavis.common.utils as utils

@registry.register_builder("textinv")
class TextInvBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextInvDataset
    eval_dataset_cls = TextInvDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/textinv/textinv.yaml",
    }
    
    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets
    
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            # annotation path
            # ann_paths = ann_info.get(split).storage
            # if isinstance(ann_paths, str):
            #     ann_paths = [ann_paths]

            # abs_ann_paths = []
            # for ann_path in ann_paths:
            #     if not os.path.isabs(ann_path):
            #         ann_path = utils.get_cache_path(ann_path)
            #     abs_ann_paths.append(ann_path)
            # ann_paths = abs_ann_paths
            
            real_label = ann_info.get(split).real_label
            fake_label = ann_info.get(split).fake_label

            # visual data storage path
            # vis_path = vis_info.storage

            # if not os.path.isabs(vis_path):
            #     # vis_path = os.path.join(utils.get_cache_path(), vis_path)
            #     vis_path = utils.get_cache_path(vis_path)

            # if not os.path.exists(vis_path):
            #     warnings.warn("storage path {} does not exist.".format(vis_path))
                
            real_dir = vis_info.get(split).real_dir
            fake_dir = vis_info.get(split).fake_dir
            
            if not os.path.isabs(real_dir):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                real_dir = utils.get_cache_path(real_dir)

            if not os.path.exists(real_dir):
                warnings.warn("storage path {} does not exist.".format(real_dir))
                
            if not os.path.isabs(fake_dir):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                fake_dir = utils.get_cache_path(fake_dir)

            if not os.path.exists(fake_dir):
                warnings.warn("storage path {} does not exist.".format(fake_dir))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=None,
                vis_root=None,
                real_dir=real_dir,
                fake_dir=fake_dir,
                real_label=real_label,
                fake_label=fake_label
            )

        return datasets