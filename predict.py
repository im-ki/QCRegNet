import argparse
import logging
import os
import sys

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

def test_net(bsnet, estimator, device):

    dataset = QCM_Gen(data_path, size, scale, half=True)
    n_test = len(dataset)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    Upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    for epoch in range(10000):
        estimator.eval()
        bsnet.eval()

        epoch_loss = 0
        sample_cnt = 0
        for batch in test_loader:
            #mu, mapping, I0 = batch
            mu, I0 = batch

            h, w = mu.shape[2], mu.shape[3]

            #assert imgs.shape[1] == 2, 'Network has been defined with {} input channels, but loaded images have {} channels. Please check that the images are loaded correctly.'.format(net.n_channels, imgs.shape[1])

            mu = mu.to(device=device, dtype=torch.float32)
            I0 = I0.to(device=device, dtype=torch.float32)
            
            with torch.no_grad():
                mapping = Upsample(bsnet(mu))
                mapping_permute = mapping.permute((0, 2, 3, 1))*2 - 1
                mapping_permute[:, :, :, 1] = -mapping_permute[:, :, :, 1]
                I1 = torch.nn.functional.grid_sample(I0, mapping_permute, mode='bilinear', padding_mode='border', align_corners=True)
                I_combine = torch.cat((I0, I1), dim=1)

            # For paper
            #I0_save = I0[0].numpy().transpose((1, 2, 0))
            #matplotlib.image.imsave('I0.png', I0_save.repeat(3, axis=2))
            #I1_save = I1[0].numpy().transpose((1, 2, 0))
            #matplotlib.image.imsave('I1.png', I1_save.repeat(3, axis=2))

            mu_recon = estimator(I_combine)[0]
            mapping_recon = bsnet(mu_recon)
            mapping = Upsample(mapping_recon)

            mapping_recon_permute = mapping.permute((0, 2, 3, 1))*2 - 1
            mapping_recon_permute[:, :, :, 1] = -mapping_recon_permute[:, :, :, 1]
            I2 = torch.nn.functional.grid_sample(I1, mapping_recon_permute, mode='bilinear', padding_mode='border', align_corners=True)

            print(I0.shape, I1.shape, I2.shape)

            # For paper
            #I2_save = I2[0].detach().numpy().transpose((1, 2, 0))
            #matplotlib.image.imsave('I2.png', I2_save.repeat(3, axis=2))

            fig = plt.figure()
            fig.add_subplot(2, 2, 1)
            plt.imshow(I0[0, 0].detach().numpy(), cmap = 'gray')
            fig.add_subplot(2, 2, 2)
            plt.imshow(I1[0, 0].detach().numpy(), cmap = 'gray')
            fig.add_subplot(2, 2, 3)
            plt.imshow(I2[0, 0].detach().numpy(), cmap = 'gray')
            fig.add_subplot(2, 2, 4)
            mapping = mapping_recon.detach().numpy()
            #fig1 = plt.figure()
            #fig1.add_subplot(1, 1, 1)
            plt.plot(mapping[0, 0].reshape((-1, 1)), mapping[0, 1].reshape((-1, 1)), 'r.', markersize=0.5)
            plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Test the QCRN on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--bsnet', dest='bsnet', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-e', '--estimator', dest='estimator', type=str, default=False,
                        help='Load model from a .pth file')

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

    bsnet.to(device=device)
    estimator.to(device=device)
    test_net(bsnet, estimator, device)
