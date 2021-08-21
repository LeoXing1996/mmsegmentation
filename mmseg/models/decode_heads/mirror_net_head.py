import enum
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmseg.ops import resize
from ..losses import accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS


class BasicConv(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(
            out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(
                    x,
                    2, (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(
            x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.mean(x, 1).unsqueeze(1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            1,
            1,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):

    def __init__(self,
                 gate_channels=128,
                 reduction_ratio=16,
                 pool_types=['avg'],
                 no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio,
                                       pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


###################################################################
# ###################### Contrast Module ##########################
###################################################################
class Contrast_Module(nn.Module):

    def __init__(self, planes):
        super(Contrast_Module, self).__init__()
        self.inplanes = int(planes)
        self.inplanes_half = int(planes / 2)
        self.outplanes = int(planes / 4)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.inplanes, self.inplanes_half, 3, 1, 1),
            nn.BatchNorm2d(self.inplanes_half), nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.inplanes_half, self.outplanes, 3, 1, 1),
            nn.BatchNorm2d(self.outplanes), nn.ReLU())

        self.contrast_block_1 = Contrast_Block(self.outplanes)
        self.contrast_block_2 = Contrast_Block(self.outplanes)
        self.contrast_block_3 = Contrast_Block(self.outplanes)
        self.contrast_block_4 = Contrast_Block(self.outplanes)

        self.cbam = CBAM(self.inplanes)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        contrast_block_1 = self.contrast_block_1(conv2)
        contrast_block_2 = self.contrast_block_2(contrast_block_1)
        contrast_block_3 = self.contrast_block_3(contrast_block_2)
        contrast_block_4 = self.contrast_block_4(contrast_block_3)

        output = self.cbam(
            torch.cat((contrast_block_1, contrast_block_2, contrast_block_3,
                       contrast_block_4), 1))

        return output


###################################################################
# ###################### Contrast Block ###########################
###################################################################
class Contrast_Block(nn.Module):

    def __init__(self, planes):
        super(Contrast_Block, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 4)

        self.local_1 = nn.Conv2d(
            self.inplanes,
            self.outplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)
        self.context_1 = nn.Conv2d(
            self.inplanes,
            self.outplanes,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2)

        self.local_2 = nn.Conv2d(
            self.inplanes,
            self.outplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)
        self.context_2 = nn.Conv2d(
            self.inplanes,
            self.outplanes,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4)

        self.local_3 = nn.Conv2d(
            self.inplanes,
            self.outplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)
        self.context_3 = nn.Conv2d(
            self.inplanes,
            self.outplanes,
            kernel_size=3,
            stride=1,
            padding=8,
            dilation=8)

        self.local_4 = nn.Conv2d(
            self.inplanes,
            self.outplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)
        self.context_4 = nn.Conv2d(
            self.inplanes,
            self.outplanes,
            kernel_size=3,
            stride=1,
            padding=16,
            dilation=16)

        self.bn = nn.BatchNorm2d(self.outplanes)
        self.relu = nn.ReLU()

        self.cbam = CBAM(self.inplanes)

    def forward(self, x):
        local_1 = self.local_1(x)
        context_1 = self.context_1(x)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn(ccl_1)
        ccl_1 = self.relu(ccl_1)

        local_2 = self.local_2(x)
        context_2 = self.context_2(x)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn(ccl_2)
        ccl_2 = self.relu(ccl_2)

        local_3 = self.local_3(x)
        context_3 = self.context_3(x)
        ccl_3 = local_3 - context_3
        ccl_3 = self.bn(ccl_3)
        ccl_3 = self.relu(ccl_3)

        local_4 = self.local_4(x)
        context_4 = self.context_4(x)
        ccl_4 = local_4 - context_4
        ccl_4 = self.bn(ccl_4)
        ccl_4 = self.relu(ccl_4)

        output = self.cbam(torch.cat((ccl_1, ccl_2, ccl_3, ccl_4), 1))

        return output


@HEADS.register_module()
class MirrorNet(BaseDecodeHead):

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes=2,
                 tar_shape=384,
                 **kwargs):
        super(MirrorNet, self).__init__(
            in_channels, channels, num_classes=num_classes, **kwargs)
        if isinstance(tar_shape, int):
            tar_shape = (tar_shape, tar_shape)
        self.tar_shape = tar_shape

        self.contrast_4 = Contrast_Module(2048)
        self.contrast_3 = Contrast_Module(1024)
        self.contrast_2 = Contrast_Module(512)
        self.contrast_1 = Contrast_Module(256)

        self.up_4 = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, 4, 2, 1), nn.BatchNorm2d(512),
            nn.ReLU())
        self.up_3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256),
            nn.ReLU())
        self.up_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128),
            nn.ReLU())
        self.up_1 = nn.Sequential(
            nn.Conv2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.cbam_4 = CBAM(512)
        self.cbam_3 = CBAM(256)
        self.cbam_2 = CBAM(128)
        self.cbam_1 = CBAM(64)

        self.layer4_predict = nn.Conv2d(512, num_classes, 3, 1, 1)
        self.layer3_predict = nn.Conv2d(256, num_classes, 3, 1, 1)
        self.layer2_predict = nn.Conv2d(128, num_classes, 3, 1, 1)
        self.layer1_predict = nn.Conv2d(64, num_classes, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def cls_seg(self, feat):
        if isinstance(feat, tuple):
            return (F.sigmoid(f) for f in feat)
        return F.sigmoid(feat)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        tar_shape = seg_label.shape[2:]
        seg_label = seg_label.squeeze(1)

        for level, logit in enumerate(seg_logit):
            logit = resize(
                input=logit,
                size=tar_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if self.sampler is not None:
                seg_weight = self.sampler.sample(logit, seg_label)
            else:
                seg_weight = None
            loss[f'loss_seg_{4-level}'] = self.loss_decode(
                logit,
                seg_label,
                weight=seg_weight,
                ignore_index=self.ignore_index)
            loss[f'acc_seg_{4-level}'] = accuracy(logit, seg_label)
        return loss

    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      depth=None):

        # focus return intermedia in this loss
        if depth is not None:
            print(depth.shape)
            print('get depth map successfully')
        seg_logits = self.forward(inputs, return_intermedia=True)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, depth=None):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        if depth is not None:
            print(depth.shape)
            print('get depth map successfully in test')
        return self.forward(inputs)

    def forward(self, inputs, return_intermedia=False):
        layer0, layer1, layer2, layer3, layer4 = inputs

        contrast_4 = self.contrast_4(layer4)
        up_4 = self.up_4(contrast_4)
        cbam_4 = self.cbam_4(up_4)
        layer4_predict = self.layer4_predict(cbam_4)

        if layer4_predict.size(1) > 1:
            layer4_map = torch.sum(
                layer4_predict[:, 1:, ...], dim=1, keepdim=True)
        else:
            layer4_map = layer4_predict
        layer4_map = F.sigmoid(layer4_map)

        contrast_3 = self.contrast_3(layer3 * layer4_map)
        up_3 = self.up_3(contrast_3)
        cbam_3 = self.cbam_3(up_3)
        layer3_predict = self.layer3_predict(cbam_3)

        if layer3_predict.size(1) > 1:
            layer3_map = torch.sum(
                layer3_predict[:, 1:, ...], dim=1, keepdim=True)
        else:
            layer3_map = layer3_predict
        layer3_map = F.sigmoid(layer3_map)

        contrast_2 = self.contrast_2(layer2 * layer3_map)
        up_2 = self.up_2(contrast_2)
        cbam_2 = self.cbam_2(up_2)
        layer2_predict = self.layer2_predict(cbam_2)

        if layer2_predict.size(1) > 1:
            layer2_map = torch.sum(
                layer2_predict[:, 1:, ...], dim=1, keepdim=True)
        else:
            layer2_map = layer2_predict
        layer2_map = F.sigmoid(layer2_map)

        contrast_1 = self.contrast_1(layer1 * layer2_map)
        up_1 = self.up_1(contrast_1)
        cbam_1 = self.cbam_1(up_1)
        layer1_predict = self.layer1_predict(cbam_1)

        out = []
        for layer in [
                layer4_predict, layer3_predict, layer2_predict, layer1_predict
        ]:
            out.append(
                torch.sigmoid(
                    F.upsample(
                        layer,
                        size=self.tar_shape,
                        mode='bilinear',
                        align_corners=True)))
        # layer4_predict = F.upsample(
        #     layer4_predict,
        #     size=x.size()[2:],
        #     mode='bilinear',
        #     align_corners=True)
        # layer3_predict = F.upsample(
        #     layer3_predict,
        #     size=x.size()[2:],
        #     mode='bilinear',
        #     align_corners=True)
        # layer2_predict = F.upsample(
        #     layer2_predict,
        #     size=x.size()[2:],
        #     mode='bilinear',
        #     align_corners=True)
        # layer1_predict = F.upsample(
        #     layer1_predict,
        #     size=x.size()[2:],
        #     mode='bilinear',
        #     align_corners=True)

        # return layer4_predict, layer3_predict, layer2_predict, layer1_predict
        return out if return_intermedia else out[0]

        # return F.sigmoid(layer4_predict), F.sigmoid(layer3_predict), \
        #     F.sigmoid(layer2_predict), F.sigmoid(layer1_predict)
