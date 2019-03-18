import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np


"""
input csv should left collumn the image names and right collumn the label
"""
class CSVDataSet(Dataset):
    def __init__(self, df_data: pd.DataFrame, impostfix: str, img_dir: str = "./", transform=None):
        super().__init__()
        self.df = df_data
        self.data_dir = img_dir
        self.transform = transform
        self.impostfix = impostfix

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        img_name, label = self.df.loc[index]
        img_path = os.path.join(self.data_dir, img_name + self.impostfix)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def getDataLoaderFromCSV(df:pd.DataFrame, img_folder:str, im_postfix, transforms:callable, batch_size, shuffle, num_workers):
    dataset = CSVDataSet(df, impostfix=im_postfix, img_dir=img_folder, transform=transforms)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader
