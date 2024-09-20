import torch
from torch import nn
import torch.nn.functional as F


class MyModel(nn.Module):
    
    def __init__(self, in_channels: int, out_labels: int):
        super(MyModel, self).__init__()
        # ==========

        # ==========

        
    def forward(self, x):
        # ==========

        # ==========
        return y

    
if __name__ == '__main__':
    model       = MyModel(in_channels=1, out_labels=5)
    data        = torch.randn(8, 1, 32, 32)
    prediction  = model(data)
    
    print('model size =', model.weight.shape, ', model type =', model.weight.dtype)
    print('data size =', data.shape, ', data type =', data.dtype)
    print('pred size =', prediction.shape, ', pred type =', prediction.dtype)