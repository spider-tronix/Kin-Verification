import torch
import torch.nn as nn
from torch.utils.data import Dataset
from skimage import io
import scipy.io as sio
import pandas as pd
import os
import PIL.Image
import numpy as np

class FIW(Dataset):
    '''
    work_dir        - path to the dataset
    label           - relative path to the label .csv file
    split           - string value - "train", "val", "test"
    val_list        - list of strings of family ids to be included in val and not included in train
    transform       - transforms to be done on the images
    '''
    def __init__(self, work_dir, label, split, transform=None, val_list=None):
        self.work_dir = work_dir
        self.label = os.path.join(self.work_dir, label)
        self.split = split
        self.val_list = val_list                                            # family ids which are to be included in val
        self.transform = transform
        
        if self.split == "test":
            if os.path.isfile(self.label):                                  # if only single relation is considered for test label
                self.df = pd.read_csv(self.label, low_memory=False, encoding = "utf-8")
            else:
                self.df = pd.DataFrame()                                    # if all the relations are considered for test label
                for f in os.listdir(self.label):
                    self.df = pd.append(self.df, pd.read_csv(os.path.join(self.label, f), low_memory=False, encoding = "utf-8"))
            self.df = self.df.reset_index(drop=True)

        else:
            self.df = pd.read_csv(self.label, low_memory=False, encoding = "utf-8")
            # eliminate the val_list from train
            if split == "val":
                self.df = self.df[(self.df["person1"].str.split("/", expand=True)[1].isin(self.val_list)) | (self.df["person2"].str.split("/", expand=True)[1].isin(self.val_list))]
                self.df = self.df.reset_index(drop=True)
            else:
                self.df = self.df[(~self.df["person1"].str.split("/", expand=True)[1].isin(self.val_list)) & (~self.df["person2"].str.split("/", expand=True)[1].isin(self.val_list))]
                self.df = self.df.reset_index(drop=True)               

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.split == "test":
            img_path_1 = os.path.join(self.work_dir, "test-private-faces", str(self.df.at[index, "p1"]))       
            img_path_2 = os.path.join(self.work_dir, "test-private-faces", str(self.df.at[index, "p2"]))
            y = torch.Tensor([self.df.at[index, "label"].astype("uint8")])
        
        else:
            img_path_1 = os.path.join(self.work_dir, str(self.df.at[index, "person1"]))
            img_path_2 = os.path.join(self.work_dir, str(self.df.at[index, "person2"]))
            # also include age and gender information in the dataset 
            age_1 = torch.Tensor([int(self.df.at[index, "p1_age"])])
            age_2 = torch.Tensor([int(self.df.at[index, "p2_age"])])
            gender_1 = torch.Tensor([int(self.df.at[index, "p1_gender"])])
            gender_2 = torch.Tensor([int(self.df.at[index, "p2_gender"])])
            y = torch.Tensor([self.df.at[index, "is_related"].astype("uint8")])

        k1 = io.imread(img_path_1)
        k2 = io.imread(img_path_2)
        
        if self.transform:
            k1 = self.transform(k1)
            k2 = self.transform(k2)

        if self.split=="test":
            return k1, k2, y
        else:
            return k1,k2,age_1,age_2,gender_1,gender_2,y
        

class KinFaceW1(Dataset):
    '''
    work_dir        - path to the dataset
    split           - string value - "train", "test"
    transforms      - transforms to be done on the images
    meta_data folder contains the .mat files with kin/non-kin labels
    '''
    def __init__(self, work_dir, split, transforms=None):
        self.work_dir = work_dir
        self.split = split
        self.label_dir = os.path.join(self.work_dir, "meta_data")
        self.pairs = []                                                          # will contain the final annotations
        self.len = 0
        self.map = {0: "fd", 1: "fs", 2: "md", 3: "ms"}                          # mapping for relationship

        if self.split == "train":
            for f in os.listdir(self.label_dir):
                self.pair = sio.loadmat(f)["pairs"]
                key = list(self.map.keys())[list(self.map.values()).index(f.split("_")[0])]     # append a column based on the relation using the mapping
                keys = key*np.ones((self.pair.shape[0], 1))
                self.pair = np.concatenate([self.pair, keys], axis=1)
                self.len = self.len + np.sum(self.pair[:,0] != [[5]])                           # 1,2,3,4 are used for train and 5 is used for test
                self.pairs.append(self.pair[self.pair[:,0] != [[5]]])                           

        else:
            for f in os.listdir(self.label_dir):
                self.pair = sio.loadmat(f)["pairs"]
                key = list(self.map.keys())[list(self.map.values()).index(f.split("_")[0])]
                keys = key*np.ones((self.pair.shape[0], 1))
                self.pair = np.concatenate([self.pair, keys], axis=1) 
                self.len = self.len + np.sum(self.pair[:,0] == [[5]]) 
                self.pairs.append(self.pair[self.pair[:,0] == [[5]]])
        self.pairs = np.concatenate(self.pairs, axis=0)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        folder = self.map[int(self.pairs[index, -1])]
        img_path_1 = os.path.join(self.work_dir, "images", folder, str(self.pairs[index, 2]))
        img_path_2 = os.path.join(self.work_dir, "images", folder, str(self.pairs[index, 3]))
        label = torch.Tensor(int(self.pairs[index, 1]))
        relation = torch.Tensor(int(self.pairs[index, -1]))

        k1 = io.imread(img_path_1)
        k2 = io.imread(img_path_2)

        if self.transforms:
            k1 = self.transforms(k1)
            k2 = self.transforms(k2)
        
        return k1,k2,relation,label


        
