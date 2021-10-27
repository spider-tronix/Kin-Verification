import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
from torchvision import transforms
import torchvision

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
        self._classes=124
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
        self._classes=124
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



def image_loader(videopath,frame_idx):
    '''
    Image loader for making Kingaitwild dataset
    params:
    videopath:directory with video frames
    frame_idx:frame sequence indices
    '''    
    try:
        frames=os.listdir(videopath)
    except:
        print("Video path doesnot exists")
    clip=[]
    for i in frame_idx:
        path=os.path.join(videopath,frames[i])
        clip.append(Image.open(path).convert("RGB"))
    return clip


class KingaitWild(Dataset):
    '''
    video_path:Path of directory with video frames
    split_file: Train and test splitfile given by the paper
    transform: Preprocessing transforms
    sample_duration: Sampling of frames as per the paper
    train: training flag
    image_size: Image_size as per the model
    '''

    def __init__(self,video_path,split_file,transform,sample_duration=16,train=True,image_size=(50,150)):
        self.split=self.make_split(video_path,split_file)
        self.data=self.make_dataset(video_path,sample_duration)
        if transform is None:
            self.transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        else:
            self.transform=transform


    def __getitem__(self,index):
        frame_idx=self.data[index]['segment']
        video_path1=self.data[index]['video_sub1']
        video_path2=self.data[index]['video_sub2']
        class_label=self.data[index]['class']
        clip1=image_loader(video_path1,frame_idx)
        clip2=image_loader(video_path2,frame_idx)
        if self.transform is not None:
            clip1=[self.transform(frame) for frame in clip1]
            clip2=[self.transform(frame) for frame in clip2]
       
        return torch.stack(clip1,0).permute(1,0,2,3),torch.stack(clip2,0).permute(1,0,2,3),class_label


    def __len__(self):
        return len(self.data)

    def make_dataset(self,video_path,sample_duration):
        '''
        func:to make the dataset based on split
        params:
        video_path:directory with the video frames of kin
        sample_duration:sampling period
        '''
        dataset=[]
        for data in self.split:
            label=data[2]
            kin1_path=data[0]
            kin2_path=data[1]
            video_path1=os.path.join(video_path,kin1_path)
            video_path2=os.path.join(video_path,kin2_path)
            frames_1=os.listdir(video_path1)
            frames_2=os.listdir(video_path2)
            no_frames=min(len(frames_1),len(frames_2))
            for i in range(0,no_frames-sample_duration+1):
                sample = {
                'video_sub1': video_path1,
                'video_sub2': video_path2,
                'class':label,
                'segment': list(range(i,i+sample_duration)),
                }
                dataset.append(sample)
                #print(sample,"\n")
        print(len(dataset))
        return dataset

    def make_split(self,videopath,split_file):
        with open(split_file) as f:
            t=f.read().replace("cropped_videos/",videopath)
            t=t.replace(".mp4","/").split("\n")
            data=[]
            for p in t:
                res=p.split(";")
                print(res)
                try:
                    data.append([res[1],res[2],int(res[3])])
                except:
                    pass

        return data


if __name__ == "__main__":


    img=Image.open("/mnt/sda2/Spider/GaitDatasetB-silh/GaitDatasetB-silh/001/nm-01/090/001-nm-01-090-050.png")
    img1=img.resize((50,150))
    img2=np.asarray(img1)

    data=CASIA_train("/mnt/sda2/Spider/GaitDatasetB-silh/GaitDatasetB-silh/")
    loader=DataLoader(data,batch_size=1,shuffle=True)

    for img,label,dir in loader:
        print(label,dir)
        break
    data1=KingaitWild(video_path="/mnt/sda2/Spider/Kingait/",split_file="/mnt/sda2/Spider/KinGaitWild-20210913T171340Z-001/KinGaitWild/folds/KinGaitWild_Train5_02.csv",transform=None)
    print(data1)
    loader=DataLoader(data1,batch_size=10,shuffle=True)
    print(data1.__len__())
    for img,label,dir in loader:
        print(img.size(),label.size(),dir)
        break