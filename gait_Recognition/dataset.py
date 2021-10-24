import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import glob


def load_image(data_dir,frame,image_size):
    list_frames=sorted(os.listdir(data_dir))
    img=[]
    for i in range(0,16):
        try:
            path=os.path.join(data_dir,list_frames[i+frame])
            inImage=Image.open(path)
            #print(inImage.size) 
            inImage=inImage.resize(image_size)
        except:
            inImage=np.zeros(shape=(image_size[1],image_size[0]))
        inImage = np.array(inImage,dtype=np.float64)/255 
        #print(inImage.min(),type(inImage))
        iw = inImage.shape[1]
        ih = inImage.shape[0]
        
        img.append([inImage]*3)
    img=np.array(img,dtype=np.float64)
    img = torch.from_numpy(img).unsqueeze(0)
    #print(img.min()) 
    return img.permute((0,2,1,3,4))



class CASIA_train(Dataset):
    def __init__(self,root_dir,train=True,image_size=(50,150)):
        self.root_dir=root_dir
        if train:
            self.cond=["nm-01","nm-02","nm-03","nm-04"]
        self._classes=len(root_dir)
        self.angles=["000","018","036","054","072","090","108","126","144","162","180"]
        self.n_cond = len(self.cond)
        self.n_ang = len(self.angles)
        self.image_size=image_size
    
    def __getitem__(self,idx):
        while(True):           
            id1_int = torch.randint(0, self._classes, (1,)).item() + 1
            label=id1_int
            id1 = '%03d' % id1_int
            cond1 = torch.randint(0, self.n_cond, (1,)).item()
            cond1 = self.cond[int(cond1)]
            ang1 = torch.randint(0, self.n_ang, (1,)).item()
            ang1 = self.angles[int(ang1)]
            #r1 = id1 + '/' + cond1 + '/' + id1 + '-' + cond1 + '-' + self.target+'.png
            data_dir=os.path.join(self.root_dir,id1,cond1,ang1)
            if not os.path.exists(data_dir):
                continue
            n_frames=len(data_dir)
            if n_frames < 10:
                continue
            frame=torch.randint(0,n_frames-16,(1,)).item()
            img=load_image(data_dir,frame,self.image_size)                     
            break
        
        return img,label,data_dir
        
    
    def __len__(self):
        return 448151

class CASIA_test(Dataset):
    def __init__(self,root_dir,image_size=(50,150)):
        self.root_dir=root_dir
        if train:
            self.cond=["nm-05","nm-06"]
        self._classes=len(root_dir)
        self.angles=["000","018","036","054","072","090","108","126","144","162","180"]
        self.n_cond = len(self.cond)
        self.n_ang = len(self.angles)
        self.image_size=image_size
    
    def __getitem__(self,idx):
        while(True):           
            id1_int = torch.randint(0, self._classes, (1,)).item() + 1
            label=id1_int
            id1 = '%03d' % id1_int
            cond1 = torch.randint(0, self.n_cond, (1,)).item()
            cond1 = self.cond[int(cond1)]
            ang1 = torch.randint(0, self.n_ang, (1,)).item()
            ang1 = self.angles[int(ang1)]
            #r1 = id1 + '/' + cond1 + '/' + id1 + '-' + cond1 + '-' + self.target+'.png
            data_dir=os.path.join(self.root_dir,id1,cond1,ang1)
            if not os.path.exists(data_dir):
                continue
            n_frames=len(data_dir)
            if n_frames < 10:
                continue
            frame=torch.randint(0,n_frames-16,(1,)).item()
            img=load_image(data_dir,frame,self.image_size)                     
            break
        
        return img,label,data_dir
        
    
    def __len__(self):
        return 228112





if __name__ == "__main__":


    img=Image.open("/mnt/sda2/Spider/GaitDatasetB-silh/GaitDatasetB-silh/001/nm-01/090/001-nm-01-090-050.png")
    img1=img.resize((50,150))
    img2=np.asarray(img1)
    #plt.imshow(img2)
    #print(sorted(os.listdir("/mnt/sda2/Spider/GaitDatasetB-silh/GaitDatasetB-silh/001/nm-01/090/")))
    img=load_image("/mnt/sda2/Spider/GaitDatasetB-silh/GaitDatasetB-silh/005/nm-01/090/",1)
    print(img.shape)
    print(np.std(img.numpy()))
    #print(len(glob.glob("/mnt/sda2/Spider/GaitDatasetB-silh/GaitDatasetB-silh/*/nm-05/*/*")))
    #print(len(glob.glob("/mnt/sda2/Spider/GaitDatasetB-silh/GaitDatasetB-silh/*/nm-06/*/*")))
    #print(len(glob.glob("/mnt/sda2/Spider/GaitDatasetB-silh/GaitDatasetB-silh/*/nm-03/*/*")))
    #print(len(glob.glob("/mnt/sda2/Spider/GaitDatasetB-silh/GaitDatasetB-silh/*/nm-04/*/*")))

    data=CASIA_train("/mnt/sda2/Spider/GaitDatasetB-silh/GaitDatasetB-silh/")
    loader=DataLoader(data,batch_size=64,shuffle=True)

    for img,label,dir in loader:
        print(label,dir)
        break