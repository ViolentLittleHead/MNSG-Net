import os
import numpy as np
from scipy import ndimage as ndi
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_pil_image,to_tensor

# 创建自定义数据集
class fetaL_dataset(Dataset):
    def __init__(self,path2data,transform=None):
        imagesPath = os.path.join(path2data, 'images')
        masksPath = os.path.join(path2data, 'masks')

        imgsList = os.listdir(imagesPath)
        masksList = os.listdir(masksPath)

        self.path2imgs = [os.path.join(imagesPath,fn) for fn in imgsList]
        self.path2annts = [os.path.join(masksPath,fn) for fn in masksList]
        self.transform = transform
    def __len__(self):
        return len(self.path2imgs)

    def __getitem__(self, idx):
        path2img = self.path2imgs[idx]
        image = Image.open(path2img)

        path2annt = self.path2annts[idx]
        mask = Image.open(path2annt)

        image = np.array(image)
        mask = np.array(mask)
        # mask = mask.astype("uint8")

        if self.transform:
            augmented = self.transform(image=image,mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        image = to_tensor(image)
        # 规范化，将掩码缩小到【0，1】的范围
        normalized_mask = mask / 255.0
        mask = to_tensor(normalized_mask)
        return image,mask
