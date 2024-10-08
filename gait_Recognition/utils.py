import numpy as np

from functools import lru_cache
from scipy.stats import multivariate_normal
import cv2
import glob
import os
from matplotlib import pyplot as plt
import pandas as pd


@lru_cache()
def get_gauss_pdf(sigma):
    n = sigma * 8

    x, y = np.mgrid[0:n, 0:n]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    rv = multivariate_normal([n / 2, n / 2], [[sigma ** 2, 0], [0, sigma ** 2]])
    pdf = rv.pdf(pos)

    heatmap = pdf / np.max(pdf)

    return heatmap


def to_int(num):
    return int(round(num))


@lru_cache()
def get_binary_mask(diameter):
    d = diameter
    _map = np.zeros((d, d), dtype = np.float32)

    r = d / 2
    s = int(d / 2)

    y, x = np.ogrid[-s:d - s, -s:d - s]
    mask = x * x + y * y <= r * r

    _map[mask] = 1.0

    return _map


def get_binary_heat_map(shape, is_present, centers, diameter = 9):
    n = diameter
    r = int(n / 2)
    hn = int(2 * n)
    qn = int(4 * n)
    pl = np.zeros((shape[0], shape[1] + qn, shape[2] + qn, shape[3]), dtype = np.float32)

    for i in range(shape[0]):
        for j in range(shape[3]):
            my = centers[i, 0, j] - r
            mx = centers[i, 1, j] - r

            if -n < my < shape[1] and -n < mx < shape[2] and is_present[i, j]:
                pl[i, my + hn:my + 3 * n, mx + hn:mx + 3 * n, j] = get_binary_mask(diameter)

    return pl[:, hn:-hn, hn:-hn, :]


def get_gauss_heat_map(shape, is_present, mean, sigma = 5):
    n = to_int(sigma * 8)
    hn = to_int(n / 2)
    dn = int(2 * n)
    qn = int(4 * n)
    pl = np.zeros((shape[0], shape[1] + qn, shape[2] + qn, shape[3]), dtype = np.float32)

    for i in range(shape[0]):
        for j in range(shape[3]):
            my = mean[i, 0, j] - hn
            mx = mean[i, 1, j] - hn

            if -n < my < shape[1] and -n < mx < shape[2] and is_present[i, j]:
                pl[i, my + dn:my + 3 * n, mx + dn:mx + 3 * n, j] = get_gauss_pdf(sigma)
                # else:
                #     print(my, mx)

    return pl[:, dn:-dn, dn:-dn, :]


def videotoframe(video_path,save_path):
    videos=sorted(glob.glob(os.path.join(video_path,"*.mp4")))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for video in videos:
        id=video.split("/")[-1].split(".")[0]
        print(id)
        save_dir=os.path.join(save_path,str(id))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        print(video)
        capture=cv2.VideoCapture(video)
        frame_cnt=0
        while True:
            success, frame = capture.read()
 
            if success:
                cv2.imwrite(os.path.join(save_dir,str(frame_cnt)+".jpg"),frame)
                print("Frame saved")
            else:
                break
 
            frame_cnt = frame_cnt+1
 
        capture.release()        

#videotoframe("/mnt/sda2/Spider/KinGaitWild-20210913T171340Z-001/KinGaitWild/cropped_videos","/mnt/sda2/Spider/Kingait/")

