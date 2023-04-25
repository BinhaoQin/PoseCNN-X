import torch
from torch import nn
from torchvision import models, ops
import torch.nn.functional as F

import random
import numpy as np

from rob599 import quaternion_to_matrix
from utils import _LABEL2MASK_THRESHOL, loss_cross_entropy, loss_Rotation, HoughVoting, IOUselection


class IConv2dReLU(nn.Sequential):
    def __init__(self, conv_layer: nn.Conv2d):
        nn.init.kaiming_normal_(conv_layer.weight, nonlinearity='relu')
        if not conv_layer.bias is None:
            nn.init.constant_(conv_layer.bias, 0)
        super().__init__(conv_layer, nn.ReLU(inplace=True))


class ILinearReLU(nn.Sequential):
    def __init__(self, linear_layer: nn.Linear):
        nn.init.kaiming_normal_(linear_layer.weight, nonlinearity='relu')
        if not linear_layer.bias is None:
            nn.init.constant_(linear_layer.bias, 0)
        super().__init__(linear_layer, nn.ReLU(inplace=True))


class SegmentationRegressor(nn.Sequential):
    def __init__(self, num_classes: int = 10, hidden_layer_dim: int = 64):
        self.num_classes = num_classes
        self.hidden_layer_dim = hidden_layer_dim
        super(SegmentationRegressor, self).__init__(
            IConv2dReLU(nn.Conv2d(256, hidden_layer_dim, 1)),
            nn.UpsamplingBilinear2d(scale_factor=8),
            nn.Conv2d(hidden_layer_dim, self.num_classes + 1, 1),
            nn.Softmax(dim=1))

    def forward(self, feature: torch.Tensor):
        probability = super(SegmentationRegressor, self).forward(feature)
        segmentation = probability.argmax(dim=1)
        bbx = self.bbx_from_label(segmentation)
        return probability, segmentation, bbx

    def bbx_from_label(self, label):
        bbx = []
        bs, H, W = label.shape
        device = label.device
        label_repeat = label.view(bs, 1, H, W).repeat(
            1, self.num_classes, 1, 1).to(device)
        label_target = torch.linspace(
            0, self.num_classes - 1, steps=self.num_classes).view(1, -1, 1, 1).repeat(bs, 1, H, W).to(device)
        mask = (label_repeat == label_target)
        for batch_id in range(mask.shape[0]):
            for cls_id in range(mask.shape[1]):
                if cls_id != 0:
                    # cls_id == 0 is the background
                    y, x = torch.where(mask[batch_id, cls_id] != 0)
                    if y.numel() >= _LABEL2MASK_THRESHOL:
                        bbx.append([batch_id, torch.min(x).item(), torch.min(y).item(),
                                    torch.max(x).item(), torch.max(y).item(), cls_id])
        bbx = torch.tensor(bbx).to(device)
        return bbx


class TranslationRegressor(nn.Sequential):
    def __init__(self, num_classes: int = 10, hidden_layer_dim: int = 128):
        self.num_classes = num_classes
        self.hidden_layer_dim = hidden_layer_dim
        super(TranslationRegressor, self).__init__(
            IConv2dReLU(nn.Conv2d(256, hidden_layer_dim, 1)),
            nn.UpsamplingBilinear2d(scale_factor=8),
            nn.Conv2d(hidden_layer_dim, 3*self.num_classes, 1))


class RotationRegressor(nn.Module):
    def __init__(self, feature_dim: int = 256, roi_shape: int = 7, hidden_dim: int = 4096, num_classes: int = 10):
        super(RotationRegressor, self).__init__()
        self.feature_dim = feature_dim
        self.roi_shape = roi_shape
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        # This input convolution layer may be removed with feature_dim = 256
        # self.feature_conv = IConv2dReLU(nn.Conv2d(256, feature_dim, 1))
        self.roi_align = ops.RoIAlign(
            roi_shape, spatial_scale=1/8.0, sampling_ratio=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            ILinearReLU(
                nn.Linear(feature_dim * roi_shape * roi_shape, hidden_dim)),
            nn.Dropout(),
            ILinearReLU(nn.Linear(hidden_dim, hidden_dim)),
            nn.Linear(hidden_dim, 4*num_classes)
        )

    def forward(self, feature: torch.Tensor, bbx: torch.Tensor):
        # x = self.feature_conv(feature)
        x = self.roi_align(feature, bbx)
        quaternion = self.fc(self.flatten(x))
        return quaternion


class PoseCNN(nn.Module):
    def __init__(self, models_pcd, cam_intrinsic, num_classes=10, iou_threshold=0.7):
        super(PoseCNN, self).__init__()
        self.models_pcd = models_pcd
        self.cam_intrinsic = cam_intrinsic
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        faster_rcnn = models.detection.fasterrcnn_resnet50_fpn_v2(
            num_classes=num_classes,
            weights_backbone=models.ResNet50_Weights.IMAGENET1K_V2)
        self.extractor = faster_rcnn.backbone
        self.segmentation = SegmentationRegressor(num_classes=self.num_classes)
        self.translation = TranslationRegressor(num_classes=self.num_classes)
        self.rotation = RotationRegressor(num_classes=self.num_classes)

    def forward(self, input_dict):
        """
        input_dict = dict_keys(['rgb', 'depth', 'objs_id', 'label', 'bbx', 'RTs', 'centermaps', 'centers'])
        """

        if self.training:
            loss_dict = {
                "loss_segmentation": 0,
                "loss_centermap": 0,
                "loss_R": 0
            }
            gt_bbx = self.getGTbbx(input_dict)
            feature = self.extractor(input_dict['rgb'])['1']
            probability, segmentation, bbx = self.segmentation(feature)
            bbx = IOUselection(bbx, gt_bbx, self.iou_threshold)
            loss_dict['loss_segmentation'] = loss_cross_entropy(
                probability, input_dict['label'])
            translation = self.translation(feature)
            loss_dict['loss_centermap'] = F.l1_loss(
                translation, input_dict['centermaps'])
            if bbx.shape[0] > 0:
                rotations = self.rotation(feature, bbx[..., :-1])
                rotations, labels = self.estimateRotation(rotations, bbx)
                loss_dict['loss_R'] = loss_Rotation(
                    rotations, self.gtRotation(bbx, input_dict), labels, self.models_pcd)
            return loss_dict
        else:
            with torch.no_grad():
                feature = self.extractor(input_dict['rgb'])['1']
                probability, segmentation, bbx = self.segmentation(feature)
                translation = self.translation(feature)
                centers, depth = HoughVoting(segmentation, translation)
                rotations = self.rotation(feature, bbx[..., :-1].float())
                rotations, labels = self.estimateRotation(rotations, bbx)
                output_dict = self.generate_pose(
                    rotations, centers, depth, bbx)
                return output_dict, segmentation

    def getGTbbx(self, input_dict):
        """
            bbx is N*6 (batch_ids, x1, y1, x2, y2, cls)
        """
        gt_bbx = []
        objs_id = input_dict['objs_id']
        device = objs_id.device
        # [x_min, y_min, width, height]
        bbxes = input_dict['bbx']
        for batch_id in range(bbxes.shape[0]):
            for idx, obj_id in enumerate(objs_id[batch_id]):
                if obj_id.item() != 0:
                    # the obj appears in this image
                    bbx = bbxes[batch_id][idx]
                    gt_bbx.append([batch_id, bbx[0].item(), bbx[1].item(),
                                  bbx[0].item() + bbx[2].item(), bbx[1].item() + bbx[3].item(), obj_id.item()])
        return torch.tensor(gt_bbx).to(device=device, dtype=torch.int16)

    def estimateRotation(self, quaternion_map, filter_bbx):
        """
        quaternion_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Rs = torch.zeros(N_filter_bbx, 3, 3)
        label = []
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            quaternion = quaternion_map[idx, (cls-1) * 4: cls * 4]
            quaternion = nn.functional.normalize(quaternion, dim=0)
            pred_Rs[idx] = quaternion_to_matrix(quaternion)
            label.append(cls)  # maybe change to instance id
        label = torch.tensor(label)
        return pred_Rs, label

    def gtRotation(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Rs = torch.zeros(N_filter_bbx, 3, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Rs[idx] = input_dict['RTs'][batch_id][cls - 1][:3, :3]
        return gt_Rs

    def generate_pose(self, pred_Rs, pred_centers, pred_depths, bbxs):
        """
        pred_Rs: a tensor [pred_bbx_size, 3, 3]
        pred_centers: [batch_size, num_classes, 2]
        pred_depths: a tensor [batch_size, num_classes]
        bbx: a tensor [pred_bbx_size, 6]
        """
        output_dict = {}
        for idx, bbx in enumerate(bbxs):
            bs, _, _, _, _, obj_id = bbx
            R = pred_Rs[idx].numpy()
            center = pred_centers[bs, obj_id - 1].numpy()
            depth = pred_depths[bs, obj_id - 1].numpy()
            if (center**2).sum().item() != 0:
                T = np.linalg.inv(
                    self.cam_intrinsic) @ np.array([center[0], center[1], 1]) * depth
                T = T[:, np.newaxis]
                if bs.item() not in output_dict:
                    output_dict[bs.item()] = {}
                output_dict[bs.item()][obj_id.item()] = np.vstack(
                    (np.hstack((R, T)), np.array([[0, 0, 0, 1]])))
        return output_dict


def eval(model, dataloader, device, alpha=0.35):
    import cv2
    model.eval()

    sample_idx = random.randint(0, len(dataloader.dataset)-1)
    # image version vis
    rgb = torch.tensor(
        dataloader.dataset[sample_idx]['rgb'][None, :]).to(device)
    inputdict = {'rgb': rgb}
    pose_dict, label = model(inputdict)
    poselist = []
    rgb = (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return dataloader.dataset.visualizer.vis_oneview(
        ipt_im=rgb,
        obj_pose_dict=pose_dict[0],
        alpha=alpha
    )
