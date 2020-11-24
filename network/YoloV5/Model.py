import pdb
import sys
import numpy as np
import torch
import os
from .yolo import Model as Yolo5
from torch import nn
import gc

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
from mscv.summary import write_image, write_loss
from .utils import *
# from mscv.cnn import normal_init
from loss import get_default_loss

from torchvision.ops import nms
from utils.ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion

import misc_utils as utils

hyp = {'momentum': 0.937,  # SGD momentum
       'weight_decay': 5e-4,  # optimizer weight decay
       'giou': 0.05,  # giou loss gain
       'cls': 0.58,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'anchor_t': 4.0,  # anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        cfgfile = 'yolov5x.yaml'
        self.detector = Yolo5(cfgfile).to(opt.device)
        self.detector.hyp = hyp
        self.detector.gr = 1.0
        self.detector.nc = opt.num_classes
        #####################
        #    Init weights
        #####################
        # normal_init(self.detector)

        print_network(self.detector)

        self.optimizer = get_optimizer(opt, self.detector)
        self.scheduler = get_scheduler(opt, self.optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)
        self.it = 0

    def update(self, sample, *arg):
        """
        Args:
            sample: {'input': a Tensor [b, 3, height, width],
                   'bboxes': a list of bboxes [[N1 × 4], [N2 × 4], ..., [Nb × 4]],
                   'labels': a list of labels [[N1], [N2], ..., [Nb]],
                   'path': a list of paths}
        """
        # self.it += 1
        # ni = self.it + self.nb * self.epoch
        #
        # if ni <= self.n_burn:
        #     xi = [0, self.n_burn]  # x interp
        #     # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
        #     accumulate = max(1, np.interp(ni, xi, [1, 64 / opt.batch_size]).round())
        #     for j, x in enumerate(self.optimizer.param_groups):
        #         # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
        #         x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * (((1 + math.cos(self.epoch * math.pi / 300)) / 2) ** 1.0) * 0.9 + 0.1])
        #         if 'momentum' in x:
        #             x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

        image = sample['image'].to(opt.device)  # target domain
        target = sample['yolo5_boxes'].to(opt.device)
        pred = self.detector(image)
        loss, loss_items = compute_loss(pred, target.to(opt.device), self.detector)
        self.avg_meters.update({'loss': sum(loss_items).item()})

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {}

    def forward(self, image):  # test
        batch_bboxes = []
        batch_labels = []
        batch_scores = []

        pred = self.detector(image)
        bboxes = pred[0]
        """xywh转x1y1x2y2"""
        bboxes[:, :, 0] = bboxes[:, :, 0] - bboxes[:, :, 2] / 2
        bboxes[:, :, 1] = bboxes[:, :, 1] - bboxes[:, :, 3] / 2
        bboxes[:, :, 2] += bboxes[:, :, 0]
        bboxes[:, :, 3] += bboxes[:, :, 1]

        b = image.shape[0]  # batch有几张图
        for bi in range(b):
            bbox = bboxes[bi]
            conf_bbox = bbox[bbox[:, 4] > opt.conf_thresh]
            xyxy_bbox = conf_bbox[:, :4]  # x1y1x2y2坐标
            scores = conf_bbox[:, 4]
            nms_indices = nms(xyxy_bbox, scores, opt.nms_thresh)

            xyxy_bbox = xyxy_bbox[nms_indices]
            scores = scores[nms_indices]

            if opt.box_fusion == 'wbf':
                boxes, scores, labels = weighted_boxes_fusion([xyxy_bbox.detach().cpu().numpy()], 
                                                            [scores.detach().cpu().numpy()], 
                                                            [np.zeros([xyxy_bbox.shape[0]], dtype=np.int32)], 
                                                            iou_thr=0.5)
            elif opt.box_fusion == 'nms':
                boxes = xyxy_bbox.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()
                labels = np.zeros([xyxy_bbox.shape[0]], dtype=np.int32)

            batch_bboxes.append(boxes)
            batch_labels.append(labels)
            batch_scores.append(scores)

        return batch_bboxes, batch_labels, batch_scores

    def inference(self, x, progress_idx=None):
        raise NotImplementedError

    def evaluate(self, dataloader, epoch, writer, logger, data_name='val'):
        return self.eval_mAP(dataloader, epoch, writer, logger, data_name)

    def load(self, ckpt_path):
        return super(Model, self).load(ckpt_path)

    def save(self, which_epoch):
        super(Model, self).save(which_epoch)


