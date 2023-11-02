import torch
from torch import nn
from typing import Optional, Callable
import torch.nn.functional as F
from torch.nn import Module
from torchvision.models import resnet
from PIL import Image
from torchvision import transforms
import cv2  


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            gate: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResBlock, self).__init__()
        if gate is None:
            self.gate = nn.ReLU(inplace=True)
        else:
            self.gate = gate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('ResBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in ResBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> Module:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gate(out)

        return out


# class ML_Point(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         c0 = params['c0']
#         c1 = params['c1']
#         c2 = params['c2']
#         self.gate = nn.ReLU(inplace=True)
#         # first layer
#         self.res1 = ResBlock(c0, c1, 1, downsample=nn.Conv2d(c0, c1, 1), gate=self.gate)
#         self.conv1 = resnet.conv1x1(c1, c2)
#         self.conv_head1 = resnet.conv1x1(c2, 4)
#         # second layer
#         self.res2 = ResBlock(c1, 16, stride=2, downsample=nn.Conv2d(c1, 16, 1, stride=2), gate=self.gate)
#         self.conv2 = resnet.conv1x1(16, 32)
#         self.conv_head2 = resnet.conv1x1(32, 16)
#         # third layer
#         self.res3 = ResBlock(16, 32, stride=2, downsample=nn.Conv2d(16, 32, 1, stride=2), gate=self.gate)
#         self.conv3 = resnet.conv1x1(32, 64)
#         self.conv_head3 = resnet.conv1x1(64, 32)
#         # pool
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

class ML_Point(nn.Module):
    def __init__(self, params):
        super().__init__()
        c0 = params['c0']
        c1 = params['c1']
        h0 = params['h0']
        c2 = params['c2']
        h1 = params['h1']
        c3 = params['c3']
        c4 = params['c4']
        h2 = params['h2']
        dim1 = params['dim1']
        dim2 = params['dim2']
        dim3 = params['dim3']
        self.gate = nn.ReLU(inplace=True)
        # first layer
        self.res1 = ResBlock(c0, c1, 1, downsample=nn.Conv2d(c0, c1, 1), gate=self.gate)
        self.conv1 = resnet.conv1x1(c1, h0)
        self.conv_head1 = resnet.conv1x1(h0, dim1)
        # second layer
        self.res2 = ResBlock(c1, c2, stride=2, downsample=nn.Conv2d(c1, c2, 1, stride=2), gate=self.gate)
        self.conv2 = resnet.conv1x1(c2, h1)
        self.conv_head2 = resnet.conv1x1(h1, dim2)
        self.conv_head12 = resnet.conv1x1((dim1 - 1)*64, dim2)
        # third layer
        self.res3 = ResBlock(c2, c3, stride=2, downsample=nn.Conv2d(c2, c3, 1, stride=2), gate=self.gate)
        self.res4 = ResBlock(c3, c4, stride=1, downsample=nn.Conv2d(c3, c4, 1, stride=1), gate=self.gate)
        self.conv3 = resnet.conv1x1(c4, h2)
        self.conv_head3 = resnet.conv1x1(h2, dim3)
        self.conv_head23 = resnet.conv1x1(2*dim2*64, 2*dim2)
        # pool
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        # unfold
        self.unfold8 = nn.Unfold(kernel_size=(8, 8), padding=0, stride=8)
        self.unfold64 = nn.Unfold(kernel_size=(64, 64), padding=0, stride=64)
    # def forward(self, x):
    #     """
    #     :param x: [B, C, H, W] C = 3, H, W % 64 == 0
    #     :return:
    #     score map        [B, 1, H, W]
    #     local desc map 0 [B, 3, H, W]
    #     local desc map 1 [B, 16, H/8, W/8]
    #     desc map         [B, 32, H/64, W/64]
    #     """
    #     # backbone feature
    #     layer1 = self.res1(x)
    #     layer2 = self.res2(layer1)  # 1/2
    #     layer2 = self.pool4(layer2)  # 1/4
    #     layer3 = self.res3(layer2)  # 1/2
    #     layer3 = self.pool4(layer3)  # 1/4
    #     # head
    #     x1 = self.gate(self.conv1(layer1))
    #     x1 = self.conv_head1(x1)
    #     scores_map = torch.sigmoid(x1[:, 3, :, :]).unsqueeze(1)
    #     x1 = torch.sigmoid(x1[:, :-1, :, :])
    #     x2 = self.gate(self.conv2(layer2))
    #     x2 = self.conv_head2(x2)
    #     x3 = self.gate(self.conv3(layer3))
    #     x3 = self.conv_head3(x3)
    #     # upsample and concat feature
    #     x3_up = F.interpolate(x3, scale_factor=8, mode='bilinear', align_corners=True)
    #     x2_up = torch.cat([x3_up, x2], dim=1)
    #     x2_up = F.interpolate(x2_up, scale_factor=8, mode='bilinear', align_corners=True)
    #     desc = x2_up #  torch.cat([x2_up, x1], dim=1)
    #     return scores_map / torch.norm(scores_map) * 300, x1, x2, x3, desc
    
    def forward(self, x):
        """
        :param x: [B, C, H, W] C = 3, H, W % 64 == 0
        :return:
        score map        [B, 1, H, W]
        local desc map 0 [B, 3, H, W]
        local desc map 1 [B, 16, H/8, W/8]
        desc map         [B, 32, H/64, W/64]
        """
        # backbone feature
        layer1 = self.res1(x)
        layer2 = self.res2(layer1)  # 1/2
        layer2 = self.pool4(layer2)  # 1/4
        layer3 = self.res3(layer2)  # 1/2
        layer3 = self.pool4(layer3)  # 1/4
        # head
        x1 = self.gate(self.conv1(layer1))
        x1 = self.conv_head1(x1)
        scores_map = torch.sigmoid(x1[:, 3, :, :]).unsqueeze(1)
        x1 = torch.sigmoid(x1[:, :-1, :, :])
        return scores_map / torch.norm(scores_map) * 300, x1

import torch
import torchvision
import torch.onnx
import numpy as np


if __name__ == '__main__':
    model = ML_Point({'c0':1, 'c1':3, 'h0':4, 'c2':32, 'h1': 32, 'c3': 64 ,'c4': 64,'h2': 64,'dim1': 4,'dim2': 16,'dim3': 16})
    
    model_path = '/home/caia/lc_ws/LET/LET-NET/last_4500.pth'
    model_data = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_data)
    for name in model.state_dict():
        print(name)
    img = cv2.imread('/home/caia/lc_ws/LET/imagenet-sample-images/n01440764_tench.JPEG', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('1', img)
    
    resized_img = cv2.resize(img, (320, 240))  
    byte_num = resized_img.astype(np.float32) / 255.0
    print(byte_num)
    tensor = torch.from_numpy(byte_num)  
    tensor = tensor.unsqueeze(0).unsqueeze(0) 
    print(tensor.shape)
    x = tensor
    score, desc= model(x)
    
    
    print(score.shape)
    print(desc.shape)
    array1 = score.squeeze().detach().numpy()  # 去除大小为1的维度并转为NumPy数组
    array2 = desc.squeeze().detach().numpy()
    array2 = np.transpose(array2, [1,2,0])
    print(array2)
    cv2.imshow('img', resized_img)
    cv2.imshow('a1', array1)
    cv2.imshow('a2', array2)
    # cv2.waitKey()
    with torch.no_grad():
        torch.onnx.export(
            model, # 要转换的模型
            x,   # 随机一组输入
            "let_net.onnx", # 导出的onnx文件名称
            verbose=True,
            opset_version=13, # ONNX算子集版本
            input_names=['input'], # 输入tensor的名称
            output_names=['score', 'desc']) # 输出tensor的名称
  
    

