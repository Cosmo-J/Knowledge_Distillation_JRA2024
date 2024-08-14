import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1,device='cpu'):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False).to(device)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None,device='cpu'):
        super(BasicBlock, self).__init__()
        self.device= device
        self.conv1 = conv3x3(inplanes, planes, stride,device=self.device)
        self.bn1 = nn.BatchNorm2d(planes).to(self.device)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,stride,device=self.device)
        self.bn2 = nn.BatchNorm2d(planes).to(self.device)
        self.downsample = downsample.to(self.device) if downsample else None
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None,device='cpu'):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.device = device
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False).to(self.device)
        self.bn1 = nn.BatchNorm2d(planes).to(self.device)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=self.stride, padding=1, bias=False).to(self.device)
        self.bn2 = nn.BatchNorm2d(planes).to(self.device)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False).to(self.device)
        self.bn3 = nn.BatchNorm2d(planes*4).to(self.device)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample.to(self.device) if downsample else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_Model(nn.Module):  

    #97 classes for my dataset
    def __init__(self, block, layers, num_classes=97,device='cpu'):
        super(ResNet_Model,self).__init__()
        
        self.inplanes = 16 # number of feauture maps produced by the first convolutional layer
        self.device = device
        self.conv1 = conv3x3(3,16,1,device=self.device)
        self.bn1 = nn.BatchNorm2d(16).to(self.device)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        self.fc = nn.Linear(64 * block.expansion, num_classes).to(self.device)

        for m in self.modules(): # iterates through layers
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False).to(self.device),
                nn.BatchNorm2d(planes * block.expansion).to(self.device)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,self.device))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, device = self.device))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
