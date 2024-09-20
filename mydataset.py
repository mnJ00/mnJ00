import os
import matplotlib.pyplot as plt
import argparse
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
from PIL import Image


class MyDataset(Dataset):
    '''
    [reference] 
        os.path.join
        os.path.isdir
        os.listdir
        glob.glob
    ''' 
    def __init__(self, root_dataset: str, transform):
        self.transform  = transform
        self.num_data   = 0
        # ==========

        # ==========
        
    '''
    [input]
    
    [output]
        self.num_data:  int
    '''
         
    def __len__(self):
        return self.num_data

    '''
    [input]
        index:  int
    [output]
        data:   torch.float32
        label:  torch.uint8
    [reference]
        Image.open
    '''
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        fname   = 
        data    = Image.open(fname)
        data    = self.transform(data)
        label   = 

        return (data, label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_dataset', type=str, default='./data/mnist')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='train')
    args = parser.parse_args()

    root_dataset    = os.path.join(args.dir_dataset, args.split)
    transform       = transforms.ToTensor()
    dataset         = MyDataset(root_dataset, transform)
    dataloader      = DataLoader(dataset, batch_size=64, drop_last=True, shuffle=True)
    
    iter_data       = iter(dataloader)
    (data, label)   = next(iter_data)
  
    print('[data]:', data.shape, data.dtype) 
    print('[label]', label.shape, label.dtype) 
   
    nRow    = 8
    nCol    = 8
    plt.figure(figsize=(nRow, nCol))
    for i in range(nRow):
        for j in range(nCol):
            idx = nCol * i + j
            plt.subplot(nRow, nCol, idx+1)
            plt.axis('off')
            plt.title(f'{int(label[idx])}')
            plt.imshow(data[idx].squeeze(0), cmap='gray')
    
    plt.show()