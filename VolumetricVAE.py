import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, _kernel_size=2, _stride=1):
        super(Encoder, self).__init__()
        self.conv3d_1 = torch.nn.Conv3d(21,128,_kernel_size,_stride,padding=1)
        self.pool3d_1 = torch.nn.MaxPool3d(2)
        self.conv3d_2 = torch.nn.Conv3d(128,64,_kernel_size,_stride,padding=1)
        self.pool3d_2 = torch.nn.MaxPool3d(2)
        self.conv3d_3 = torch.nn.Conv3d(64,32,_kernel_size,_stride,padding=1)
        self.pool3d_3 = torch.nn.MaxPool3d(2)

    def forward(self, x):
        x = F.relu(self.conv3d_1(x))
        x = self.pool3d_1(x)
        x = F.relu(self.conv3d_2(x))
        x = self.pool3d_2(x)
        x = F.relu(self.conv3d_3(x))
        x = self.pool3d_3(x)

        x = x.view(-1)
        
        return x

class Decoder(torch.nn.Module):
    def __init__(self, _kernel_size=2,_stride=1):
        super(Decoder, self).__init__()
        self.conv3d_1 = torch.nn.Conv3d(32,64,3,1,1)
        self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3d_2 = torch.nn.Conv3d(64,128,3,1,1)
        self.upsample_2 = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3d_3 = torch.nn.Conv3d(128,21,3,1,1)
        self.upsample_3 = torch.nn.Upsample(scale_factor=2, mode="nearest")
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.upsample_1(x)
        x = F.relu(self.conv3d_1(x))
        x = self.upsample_2(x)
        x = F.relu(self.conv3d_2(x))
        x = self.upsample_3(x)
        x = F.relu(self.conv3d_3(x))

        x = self.softmax(x)
        
        return x
    
class VolumetricAE(torch.nn.Module):    
    def __init__(self,_kernel_size=2,_stride=1):
        super(VolumetricAE, self).__init__()
        self.encoder = Encoder(_kernel_size, _stride)
        self.decoder = Decoder(_kernel_size, _stride)
        
        self._z1 = torch.nn.Linear(2048,256)
        self._z2 = torch.nn.Linear(256,2048)
    
    def forward(self, x):
        bottle_neck = self.encoder.forward(x)
        bottle_neck = self._z1(bottle_neck)
        bottle_neck = self._z2(bottle_neck)
        bottle_neck = bottle_neck.view([-1,32,4,4,4])
        output = self.decoder.forward(bottle_neck)

        return output
        
    
class VolumetricVAE(torch.nn.Module):    
    def __init__(self,_kernel_size=2,_stride=1):
        super(VolumetricVAE, self).__init__()
        self.encoder = Encoder(_kernel_size, _stride)
        self.decoder = Decoder(_kernel_size, _stride)
        
        self._enc_mu = torch.nn.Linear(2048, 256)
        self._enc_log_sigma = torch.nn.Linear(2048,256)
        
        self._z = torch.nn.Linear(256,2048)
        
    def _sampling(self, bottle_neck):
        mu = self._enc_mu(bottle_neck)
        log_sigma = self._enc_log_sigma(bottle_neck)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False).cuda()  # Reparameterization trick

    
    def forward(self, x):
        bottle_neck = self.encoder.forward(x)
        z = self._sampling(bottle_neck)
        z = self._z(z)
        z = z.view([-1,32,4,4,4])
        output = self.decoder.forward(z)

        return output, self.z_mean, self.z_sigma
        