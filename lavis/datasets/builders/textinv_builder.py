"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.textinv_datasets import TextInvDataset

@registry.register_builder("textinv")
class TextInvBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextInvDataset
    eval_dataset_cls = TextInvDataset

    DATASET_CONFIG_DICT = {
        "default": "/home/denny/LAVIS/lavis/configs/datasets/textinv/textinv.yaml",
    }