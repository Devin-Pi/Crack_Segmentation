import torch
import torch.nn as nn

class SpectralBlock(nn.Module):

    def __init__(self, dim, h, w):
        super().__init__()
        self.time_dim = 256        
        self.layer_norm_1 = nn.LayerNorm([h, w])
        self.spectralfft = SpectralFFT(dim = dim)


        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_dim,
                      dim)
        )
        
    def forward(self, x):
        # print(x.shape)
        output = self.layer_norm_1(self.spectralfft(x))

        return output
       
class SpectralFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 512:
            self.h = 32 # H
            self.w = 17 # (w/2 + 1) this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(dim, self.h, self.w, 2, dtype = torch.float32) * 0.02)
        if dim == 256:
            self.h = 64 # H
            self.w = 33 # (w/2 + 1) this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(dim, self.h, self.w, 2, dtype = torch.float32) * 0.02)
        if dim == 128:
            self.h = 128 # H
            self.w = 65 # (w/2 + 1) this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(dim, self.h, self.w, 2, dtype = torch.float32) * 0.02)
        if dim == 64:
            self.h = 56 # H
            self.w = 29 # (w/2 + 1) this is due to rfft2
            self.complex_weight = nn.Parameter(torch.randn(dim, self.h, self.w, 2, dtype = torch.float32) * 0.02)
     
    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size
        # x = x.view(B, C, a, b)
        x = x.to(torch.float32) # x [2,128,128,128]
        # print('x before', x.shape)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho') # x [2,128,128,65]
        weight = torch.view_as_complex(self.complex_weight)
        # print('weight', weight.shape)
        # print('x', x.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x