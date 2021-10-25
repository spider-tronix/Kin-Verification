import torch
import torch.nn as nn
from torch.utils.data import Dataset
from skimage import io
import scipy.io as sio
import pandas as pd
import re

class FIW(Dataset):
    '''
    work_dir        - path to the dataset
    label           - relative path to the label .csv file
    split           - string value - "train", "val", "test"
    val_list        - list of strings of family ids to be included in val and not included in train
    transform       - transforms to be done on the images
    '''
    def __init__(self, work_dir, label, split, val_list, transform=None):
        self.work_dir = work_dir
        self.label = os.path.join(self.work_dir, label)
        self.split = split
        self.val_list = val_list                                            # family ids which are to be included in val
        self.transform = transform
        
        if self.split == "test":
            if os.path.isfile(self.label):                                  # if only single relation is considered for test label
                self.df = pd.read_csv(self.label, low_memory=False)
            else:
                self.df = pd.Dataframe()                                    # if all the relations are considered for test label
                for f in os.listdir(self.label):
                    self.df.append(pd.read_csv(os.path.join(self.label, f), low_memory=False))

        else:
            self.df = pd.read_csv(self.label, low_memory=False)
            # eliminate the val_list from train
            if split == "val":
                self.df = self.df[(self.df["person1"].split("/")[1] in self.val_list) or (self.df["person2"].split("/")[1] in self.val_list)]
            else:
                self.df = self.df[(self.df["person1"].split("/")[1] not in self.val_list) or (self.df["person2"].split("/")[1] not in self.val_list)]


    def __len__(self):
        return self.df.count()

    def __getitem__(self, index):
        if self.split == "test":
            img_path_1 = os.path.join(self.work_dir, "test-private-faces", str(self.df["person1"].iloc(index)))         
            img_path_2 = os.path.join(self.work_dir, "test-private-faces", str(self.df["person2"].iloc(index)))
        
        else:
            img_path_1 = os.path.join(self.work_dir, str(self.df["person1"].iloc(index)))
            img_path_2 = os.path.join(self.work_dir, str(self.df["person2"].iloc(index)))
            # also include age and gender information in the dataset 
            age_1 = int(self.df["p1_age"].iloc(index))
            age_2 = int(self.df["p2_age"].iloc(index))
            gender_1 = int(self.df["p1_gender"].iloc(index))
            gender_2 = int(self.df["p2_gender"].iloc(index))

        k1 = io.imread(img_path_1)
        k2 = io.imread(img_path_2)
        y = torch.Tensor(self.df["is_related"].iloc(index))
        
        if self.transform:
            k1 = self.transform(k1)
            k2 = self.transform(k2)

        return ((k1,k2,age_1,age_2,gender_1,gender_2), y)
        

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
        
        return ((k1,k2,relation), label)

        