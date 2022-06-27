import torch
import torch.utils.data as data
import glob, os
from PIL import Image
import numpy as np
from typing import Any, Callable, Optional

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path, transforms: Optional[Callable] = None, mask_transforms: Optional[Callable] = None):
        super(DataLoaderSegmentation, self).__init__()
        self.transforms = transforms
        self.mask_transforms = mask_transforms
        self.img_files = glob.glob(os.path.join(folder_path,'img','*.tif'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'mask',os.path.basename(img_path)))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            with open(img_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
#                 data = use opencv or pil read image using img_path
                data = Image.open(image_file).convert("RGB")
#                 np_data = np.array(data)
#                 np_data = np.transpose(np_data)
#                 label =use opencv or pil read label  using mask_path
                label = Image.open(mask_file).convert("L")

#                 np_label = np.array(label)
#                 np_label = np.transpose(np_label)


                if self.transforms:
                    data = self.transforms(data)
            
                if self.mask_transforms:
                    label = self.mask_transforms(label)
    
#                 return torch.from_numpy(np_data).float(), torch.from_numpy(np_label).float()
#                 return torch.from_numpy(np_data), torch.from_numpy(np_label)
                return data, label

    def __len__(self):
        return len(self.img_files)