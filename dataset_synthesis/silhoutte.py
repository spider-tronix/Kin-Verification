############################################################
#   Code to generate silhoutte sequence from raw videos    #
############################################################

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torchvision
import cv2
import numpy as np
import glob
import detectron2
import json
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json,random
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Instances

parser = argparse.ArgumentParser()
parser.add_argument('-wd', '--working_dir', help="specify the path to your input videos directory by <-wd your_directory>", default="/content/drive/MyDrive/input")
parser.add_argument('-r', '--run_type', nargs="?", help="Type <-r run-all> to run the program for all files from the beginning")
parser.add_argument('-c', '--config', help="specify the path to the config file in detectron2 format", default="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
parser.add_argument('-wts', '--weights', help="specify the path to the model weights", default="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
args = parser.parse_args()
if args.run_type=="run-all":
    r = "run-all"
    print("Running the program for all files...")
elif args.run_type=="contd" or args.run_type=="continue" or not args.run_type:
    r = "contd"
    print("Continuing the program for remaining files...")

device = "cuda" if torch.cuda.is_available() else "cpu"
key_boxes = "pred_boxes"
key_masks = "pred_masks"
epsilon = torch.Tensor([10**-10])
wts = [0.2,0.2,0.3,0.3]                                         #weights for centre, vertices,masks and area constraints

def normalize_tensor(cur,prev):
    mean = torch.mean(cur,dim=0,keepdim=True)
    std = torch.std(cur,dim=0,keepdim=True) + epsilon
    cur = (cur-mean)/std
    prev = (prev-mean)/std
    return cur, prev

def choose(seg_info, prev_info):
    # This function uses temporal continuity in bbox coordinates as constraints to stick with the same person
    # Arguements:
    #   seg_info    (type: detectron2.structures.Instances): Instance segmentation info of the current frame
    #   prev_info   (type: detectron2.structures.Instances): Bbox info for the selected person captured in the prev frame
    # Returns (a single value) the index of the instance corresponding to the selected person. (type: torch.Tensor)
    centres_prev = prev_info[key_boxes].get_centers()
    centres_cur = (seg_info.get(key_boxes)).get_centers()
    centres_cur, centres_prev = normalize_tensor(centres_cur, centres_prev)

    boxes_cur = [t.view(1,4) for t in list(seg_info.get(key_boxes))]
    boxes_cur = torch.cat(boxes_cur, dim=0)
    boxes_prev = list(prev_info[key_boxes])[0].view(1,4)
    boxes_cur, boxes_prev = normalize_tensor(boxes_cur, boxes_prev)

    masks_cur = seg_info.get(key_masks)
    masks_prev = prev_info[key_masks]
    w_h = masks_prev.shape[1]*masks_prev.shape[2]
    
    ar_prev = prev_info[key_boxes].area()
    ar_cur = seg_info.get(key_boxes).area()
    ar_cur, ar_prev = normalize_tensor(ar_cur, ar_prev)

    cost = torch.zeros(1,len(seg_info))

    for i in range(len(seg_info)):
        cost[0,i] = (torch.matmul(centres_cur[i].view(1,2) - centres_prev, torch.transpose(centres_cur[i].view(1,2) - centres_prev, 0, 1)))*wts[0]/2             
        cost[0,i] = cost[0,i] + (torch.matmul(boxes_cur[i].view(1,4) - boxes_prev, torch.transpose(boxes_cur[i].view(1,4) - boxes_prev, 0, 1)))*wts[1]/4 
        cost[0,i] = cost[0,i] + ((ar_cur[i] - ar_prev)**2)*wts[2]
        cost[0,i] = cost[0,i] + torch.sum((torch.logical_xor(masks_cur, masks_prev)).long())*wts[3]/w_h

    return torch.argmin(cost)

cfg = get_cfg()
cfg.MODEL.DEVICE = device
cfg.merge_from_file(model_zoo.get_config_file(args.config))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.weights)
predictor = DefaultPredictor(cfg)

#folder_input = "E:\\Prem\\spiderRD\\ideation\\gait\\input"
folder_input = args.working_dir
#folder_output = "E:\\Prem\\spiderRD\\ideation\\gait\\output"
folder_output = os.path.join(os.path.dirname(args.working_dir), "output")

for f in glob.glob(os.path.join(folder_input,"*.mp4")): 
    filename = os.path.basename(f)
    if r=="contd" and os.path.isfile(os.path.join(folder_output, filename)):
        continue
    dictionary = {}                                             #to store instances info for each file and frame
    f_out = filename.replace("mp4","json")                      #store as json
    cap = cv2.VideoCapture(f)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {f}")
    print(f"Frame count: {num_frames}")
    vdo = cv2.VideoWriter(os.path.join(folder_output,filename),cv2.VideoWriter_fourcc('m','p','4','v'), fps=float(fps), frameSize=(width, height), isColor=False)
    #vdo = cv2.VideoWriter(os.path.join(folder_output,filename),cv2.VideoWriter_fourcc('M','P','4','V'), fps=float(fps), frameSize=(width, height))
    dic1 = {}                                                           #dictionary to store previous frame info
    while cap.isOpened():
        ret,im = cap.read()
        if ret==True:
            cur_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            instance_seg = predictor(im)["instances"]
            instance_seg = instance_seg[instance_seg.pred_classes == 0]         #class label for person in coco
            instance_seg = instance_seg[instance_seg.scores > 0.9]              #confident predictions
            
            #assuming our object of interest to be focussed. It will have the largest area. You may need to adjust your input raw video to focus.
            if len(instance_seg) > 1:
                if bool(dic1):                        #for other frames use temporal continuity
                    ind = choose(instance_seg, dic1)
                else:                                       #for 0th frame only use max area constraint to choose the person of interest
                    ar = instance_seg.pred_boxes.area()                             
                    ind = torch.argmax(ar)
                instance_seg = instance_seg[int(ind)]
            assert(len(instance_seg)==1)
            
            #we need only the segmentation mask. We dont need the label and bounding box in the output image. So we should only pass
            #a detectron2.structures.Instances which only has "pred_masks" key. We use IMAGE_BW instance_mode to perform logical operations at ease
            ins = Instances(im.shape[:2])
            ins.pred_masks = instance_seg.pred_masks
            v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW, scale=1.0)
            #out = v1.draw_instance_predictions(im,ins.to(device))
            out = v.draw_instance_predictions(ins.to(device))
            out = cv2.cvtColor(out.get_image(),cv2.COLOR_RGB2BGR)       #detectron2 uses RGB format
            bg = out[:,:,0] == out[:,:,1]                               #IMAGE_BW converts non-masked areas to grayscale (same intensities across 3 channels)
            gr = out[:,:,1] == out[:,:,2]
            out = np.where(np.logical_and(bg,gr), np.zeros_like(out[:,:,0]), np.ones_like(out[:,:,0])*255)     #Graysclae = 0, masked areas = 255
            vdo.write(out)
            
            #instance information for current frame
            dic = instance_seg.get_fields()
            dic1["pred_boxes"] = dic.get("pred_boxes")
            dic1["pred_masks"] = dic.get("pred_masks")
            
            for key in dic:
                if key == "pred_masks":
                    dic[key] = dic[key].squeeze()
                    dic[key] = dic[key].nonzero()
                    dic[key] = torch.Tensor.tolist(dic[key])
                    
                elif key != "pred_boxes":
                    dic[key] = dic[key].tolist()
                else:
                    for element in dic[key]:
                        dic[key] = element.tolist()
            dictionary[int(cur_frame)] = dic                            #store the instances for the current frame.
            
            if cur_frame % 10 == 0:
                print("Finished processing frame ", cur_frame)
                #cv2.imshow("out",out)
                #cv2.waitKey(1000)
                #cv2.destroyAllWindows()
        else:
            break
    with open(os.path.join(folder_output,f_out), "w") as outfile:       #write the instances in json file
            json.dump(dictionary,outfile)
    vdo.release()
    print(f"Saved video {os.path.join(folder_output,filename)}")
    print(f"Saved instances in {os.path.join(folder_output,f_out)}")
    cap.release()
