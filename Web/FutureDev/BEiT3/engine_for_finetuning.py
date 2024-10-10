# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import json
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.utils import ModelEma
from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets import get_sentencepiece_model_for_beit3

import utils


class TaskHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        raise NotImplementedError()


class RetrievalHandler(TaskHandler):
    def __init__(self) -> None:
        super().__init__()
        self.image_feats = []
        self.text_feats = []
        self.image_ids = []
        self.metric_logger = None

    def train_batch(self, model, image, language_tokens, padding_mask, image_id):
        loss, vision_cls, language_cls = model(
            image=image, text_description=language_tokens, padding_mask=padding_mask
        )
        return {
            "loss": loss,
        }

    def before_eval(self, metric_logger, **kwargs):
        self.image_feats.clear()
        self.text_feats.clear()
        self.image_ids.clear()
        self.metric_logger = metric_logger

    def eval_batch(self, model, image, language_tokens, padding_mask, image_id):
        vision_cls, _ = model(image=image, only_infer=True)
        _, language_cls = model(
            text_description=language_tokens, padding_mask=padding_mask, only_infer=True
        )

        self.image_feats.append(vision_cls.clone())
        self.text_feats.append(language_cls.clone())
        self.image_ids.append(image_id.clone())

    def after_eval(self, **kwargs):
        image_feats = {}
        for feats, ids in zip(self.image_feats, self.image_ids):
            for i, _idx in enumerate(ids):
                idx = _idx.item()
                if idx not in image_feats:
                    image_feats[idx] = feats[i]

        tiids = torch.cat(self.image_ids, dim=0)
        iids = []
        sorted_tensors = []
        for key in sorted(image_feats.keys()):
            sorted_tensors.append(image_feats[key].view(1, -1))
            iids.append(key)

        image_cls_feats = torch.cat(sorted_tensors, dim=0)
        text_cls_feats = torch.cat(self.text_feats, dim=0)

        scores = image_cls_feats @ text_cls_feats.t()
        iids = torch.LongTensor(iids).to(scores.device)

        print("scores: {}".format(scores.size()))
        print("iids: {}".format(iids.size()))
        print("tiids: {}".format(tiids.size()))

        topk10 = scores.topk(10, dim=1)
        topk5 = scores.topk(5, dim=1)
        topk1 = scores.topk(1, dim=1)

        topk10_iids = tiids[topk10.indices]
        topk5_iids = tiids[topk5.indices]
        topk1_iids = tiids[topk1.indices]

        tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
        tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
        tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

        topk10 = scores.topk(10, dim=0)
        topk5 = scores.topk(5, dim=0)
        topk1 = scores.topk(1, dim=0)
        topk10_iids = iids[topk10.indices]
        topk5_iids = iids[topk5.indices]
        topk1_iids = iids[topk1.indices]

        ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
        ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
        ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

        eval_result = {
            "tr_r10": tr_r10.item() * 100.0,
            "tr_r5": tr_r5.item() * 100.0,
            "tr_r1": tr_r1.item() * 100.0,
            "ir_r10": ir_r10.item() * 100.0,
            "ir_r5": ir_r5.item() * 100.0,
            "ir_r1": ir_r1.item() * 100.0,
            "average_score": 100.0
            * (tr_r1 + tr_r5 + tr_r10 + ir_r1 + ir_r5 + ir_r10).item()
            / 6.0,
        }

        print("* Eval result = %s" % json.dumps(eval_result))
        return eval_result, "average_score"


def get_handler(args):
    if args.task in ("flickr30k", "coco_retrieval"):
        return RetrievalHandler()

    raise NotImplementedError("Sorry, %s is not support." % args.task)


@torch.no_grad()
def evaluate(data_loader, model, device, handler):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger)

    for data in metric_logger.log_every(data_loader, 10, header):
        for tensor_key in data.keys():
            data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            handler.eval_batch(model=model, **data)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return handler.after_eval()
