"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.datasets.data_utils import prepare_sample

import torch
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.dist_utils import is_dist_avail_and_initialized
import torch.distributed as dist
import logging

@registry.register_task("textual_inversion")
class TextualInversionTask(BaseTask):
    def __init__(self, evaluate, report_metric=True):
        super().__init__()

        # self.prompt = prompt
        # self.pseudo_word = pseudo_word
        # self.init_word = init_word
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        # prompt = run_cfg.prompt
        # pseudo_word = run_cfg.pseudo_word
        # init_word = run_cfg.init_word
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            # prompt=prompt,
            # pseudo_word=pseudo_word,
            # init_word=init_word,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    @torch.no_grad()
    def valid_step(self, model, samples):
        output = model(samples)
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k:
                loss_dict[k] = v

        return output["loss"]

    def evaluation(self, model, data_loader, cuda_enabled=True):
        """ Act like _train_inner_loop(). """
    
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            evalLoss = self.valid_step(model=model, samples=samples)
            # if not isinstance(evalLoss, list):
            #     evalLoss = [evalLoss]
            # results.extend(evalLoss)

            metric_logger.update(loss=evalLoss)
        

        if is_dist_avail_and_initialized():
            dist.barrier()
            metric_logger.synchronize_between_processes()

        logging.info("Evaluation stats: " + str(metric_logger.global_avg()))

        # encapsulate the loss into the format expected in train() in the responding runner.py
        # metric = 1/loss (smaller loss, greater metric)
        return {"agg_metrics": 1/metric_logger.meters["loss"].global_avg}
    
    def after_evaluation(self, val_result, split_name, epoch):
        # return val_log
        return val_result