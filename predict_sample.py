import argparse
import logging
import os
import sys
import cv2

import numpy as np
import torch
import torch.nn as nn

from estimator import Estimator
from bsnet import BSNet
from data import QCM_Gen

from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt


#data_path = '../imagenet.pkl'
data_path = '../../clefi.pkl'
size = [224, 224]
scale = [500, 500]

def test_net(bsnet, estimator, device, img1, img2):
    Upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    I0 = torch.mean(img1, dim = 1).unsqueeze(0)
    I1 = torch.mean(img2, dim = 1).unsqueeze(0)

    I0 = I0.to(device=device, dtype=torch.float32)
    img1 = img1.to(device=device, dtype=torch.float32)
    I1 = I1.to(device=device, dtype=torch.float32)
    img2 = img2.to(device=device, dtype=torch.float32)

    I_combine = torch.cat((I0, I1), dim=1)
    
    mu_recon = estimator(I_combine)[0]
    mapping_recon = bsnet(mu_recon)
    mapping = Upsample(mapping_recon)

    mapping_recon_permute = mapping.permute((0, 2, 3, 1))*2 - 1
    mapping_recon_permute[:, :, :, 1] = -mapping_recon_permute[:, :, :, 1]
    I2 = torch.nn.functional.grid_sample(I1, mapping_recon_permute, mode='bilinear', padding_mode='border', align_corners=True)
    img3 = torch.nn.functional.grid_sample(img1, mapping_recon_permute, mode='bilinear', padding_mode='border', align_corners=True)

    mapping_recon = mapping_recon.detach().numpy()
    Mqc_x, Mqc_y = mapping_recon[0, 0].reshape(-1), mapping_recon[0, 1].reshape(-1) 

    fig = plt.figure()
    fig.add_subplot(2, 2, 1)
    plt.title("I0")
    plt.axis('off')
    plt.margins(0,0)
    plt.imshow(cv2.cvtColor(img1[0].detach().numpy().transpose((1, 2, 0)), cv2.COLOR_BGR2RGB))

    fig.add_subplot(2, 2, 2)
    plt.title("I1")
    plt.axis('off')
    plt.margins(0,0)
    plt.imshow(cv2.cvtColor(img2[0].detach().numpy().transpose((1, 2, 0)), cv2.COLOR_BGR2RGB))

    fig.add_subplot(2, 2, 3)
    plt.title("QCRegNet")
    plt.axis('off')
    plt.margins(0,0)
    plt.imshow(cv2.cvtColor(img3[0].detach().numpy().transpose((1, 2, 0)), cv2.COLOR_BGR2RGB))

    fig.add_subplot(2, 2, 4); plt.plot(Mqc_x, Mqc_y, 'r.', markersize=0.5)
    plt.title("QCRegNet deform field")
    plt.axis('off')
    plt.margins(0,0)

    plt.show()
   
def get_args():
    parser = argparse.ArgumentParser(description='Test the QCRN on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--bsnet', dest='bsnet', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-e', '--estimator', dest='estimator', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-o', '--original', dest='original', type=str, default=False,
                        help='Path to load the original image')
    parser.add_argument('-d', '--distort', dest='distort', type=str, default=False,
                        help='Path to load the distorted image')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cpu')

    estimator = Estimator(size, device, n_channels=2, n_classes=2, bilinear=True)
    bsnet = BSNet([int(size[0]/2), int(size[1]/2)], n_channels=2, n_classes=2, bilinear=True)

    if args.bsnet:
        bsnet.load_state_dict(
            torch.load(args.bsnet, map_location=device)
        )
        logging.info('Model loaded from {}'.format(args.bsnet))
    else:
        print('Need a pretrained BSNet')
        sys.exit(0)

    if args.estimator:
        estimator.load_state_dict(
            torch.load(args.estimator, map_location=device)
        )
        logging.info('Model loaded from {}'.format(args.estimator))
    else:
        print('Need a pretrained Estimator')
        sys.exit(0)

    if args.original:
        img1 = torch.from_numpy(cv2.resize(cv2.imread(args.original), size)).permute((2, 0, 1)).unsqueeze(0).to(dtype=torch.float32) / 255
        logging.info('Image 1 loaded from {}'.format(args.original))
    else:
        print('Need a original image')
        sys.exit(0)

    if args.distort:
        img2 = torch.from_numpy(cv2.resize(cv2.imread(args.distort), size)).permute((2, 0, 1)).unsqueeze(0).to(dtype=torch.float32) / 255
        logging.info('Image 2 loaded from {}'.format(args.distort))
    else:
        print('Need a distorted image')
        sys.exit(0)


    bsnet.to(device=device)
    estimator.to(device=device)
    test_net(bsnet, estimator, device, img1, img2)
