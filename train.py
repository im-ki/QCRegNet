import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

from estimator import Estimator
from bsnet import BSNet
from data import QCM_Gen
from utils import coo_iv2torch, detD

#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


#data_path = '../imagenet.pkl'
data_path = '../../clefi.pkl'
size = [224, 224]
scale = [500, 500]
dir_checkpoint = 'checkpoints/'

class detDFunc(nn.Module):
    def __init__(self, weight, device):
        super(detDFunc, self).__init__()
        self.weight = weight
        self.detD_loss = detD(device, int(size[0]/2), int(size[1]/2))

    def forward(self, pred):
        return self.detD_loss(pred)

class LapLossFunc(nn.Module):
    def __init__(self, weight, device):
        super(LapLossFunc, self).__init__()
        self.lap = Laplacian(device)
        self.weight = weight
        
    def forward(self, pred):
        lap = self.lap(pred)
        return torch.mean(lap**2)

class Laplacian(nn.Module):
    def __init__(self, device):
        super(Laplacian, self).__init__()
        kernel = np.array([[0., 1., 0.], [1., -4, 1.], [0., 1., 0.]])
        #kernel = np.array([[1., 0., 1.], [0., -4, 0.], [1., 0., 1.]])
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel.to(device = device), requires_grad=False)
        #self.rep = nn.ReplicationPad2d(1)
 
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        #x1 = F.conv2d(self.rep(x1.unsqueeze(1)), self.weight, padding=0)
        #x2 = F.conv2d(self.rep(x2.unsqueeze(1)), self.weight, padding=0)
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=0)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=0)
        x = torch.cat([x1, x2], dim=1)
        return x

class LossFunc(nn.Module):
    def __init__(self, weight, device):
        super(LossFunc, self).__init__()
        self.device = device
        self.weight = weight
        
    def forward(self, mu):
        return torch.mean(mu**2)

def train_net(bsnet,
              estimator,
              device,
              epochs=5,
              batch_size=1,
              weight = 0,
              lr=0.001,
              save_cp=True):

    dataset = QCM_Gen(data_path, size, scale, half = True)
    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

    # writer = SummaryWriter(comment='LR_{}_BS_{}'.format(lr, batch_size))
    global_step = 0

    logging.info('''Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Checkpoints:     {}
        Device:          {}
    '''.format(epochs, batch_size, lr, save_cp, device.type))

    optimizer = optim.RMSprop(estimator.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if estimator.n_classes > 1 else 'max', patience=2)
    criterion1 = nn.MSELoss()
    criterion2 = LossFunc(weight, device)
    criterion3 = LapLossFunc(weight, device)
    criterion4 = detDFunc(weight, device)

    Upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    for epoch in range(epochs):
        estimator.train()
        bsnet.eval()

        epoch_loss = 0
        sample_cnt = 0
        sum_detD = 0
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch+1, epochs), unit='img') as pbar:
            for batch in train_loader:
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
                    I1 = torch.round(torch.nn.functional.grid_sample(I0, mapping_permute, mode='bilinear', padding_mode='border', align_corners=True))
                    I_combine = torch.cat((I0, I1), dim=1) / 255

                I0 = I0 / 255
                I1 = I1 / 255

                mu_recon, raw_mu = estimator(I_combine)
                #mu_recon_masked = mu_recon * mask
                map_pred = bsnet(mu_recon)
                mapping_recon = Upsample(map_pred)

                mapping_recon_permute = mapping_recon.permute((0, 2, 3, 1))*2 - 1
                mapping_recon_permute[:, :, :, 1] = -mapping_recon_permute[:, :, :, 1]
                I2 = torch.nn.functional.grid_sample(I1, mapping_recon_permute, mode='bilinear', padding_mode='border', align_corners=True)


                #if epoch < 10:
                #    loss = criterion1(map_pred, mapping)
                #else:
                #    loss = criterion2(map_pred, mapping)
                #loss = criterion2(map_pred, mapping)
                loss1 = 500 * criterion1(I2, I0) # fidelity loss
                loss2 = criterion2(raw_mu) # suppress distortion
                loss3 = criterion3(mu_recon) #laplacian loss. forget to run this one
                #detDloss = 5 * criterion4(map_pred)
                #sum_detD += detDloss.item()
                loss = loss1 + loss2 + loss3# + detDloss
                epoch_loss += loss.item()
                sample_cnt += 1
                # writer.add_scalar('Loss/train', loss.item(), global_step)

                #pbar.set_postfix(**{'loss (batch)': loss.item(), 'epoch avg loss:': epoch_loss / sample_cnt, 'Intensity loss:': loss1.item(), 'mu_sqr loss': loss2.item(), 'Smoothing loss': loss3.item(), 'avg det': sum_detD / sample_cnt})
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'epoch avg loss:': epoch_loss / sample_cnt, 'Intensity loss:': loss1.item(), 'mu_sqr loss': loss2.item(), 'Smoothing loss': loss3.item()})

                optimizer.zero_grad()
                loss.backward()
                #for name, param in net.named_parameters():
                #    print(sample_cnt, name, torch.isfinite(param.grad).all())
                #for name, param in criterion2.named_parameters():
                #    print(sample_cnt, name, torch.isfinite(param.grad).all())
                nn.utils.clip_grad_value_(estimator.parameters(), 0.1)
                optimizer.step()

                pbar.update(batch_size)
                global_step += 1
                #if global_step % (n_train // (10 * batch_size)) == 0:
                #    for tag, value in net.named_parameters():
                #        tag = tag.replace('.', '/')
                        # writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    # scheduler.step(val_score)
                    # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    #writer.add_images('images', imgs, global_step)
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(estimator.state_dict(),
                       dir_checkpoint + 'CP_epoch{}.pth'.format(epoch + 1))
            logging.info('Checkpoint {} saved !'.format(epoch + 1))

    # writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=2000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-w', '--weight', dest='weight', type=float, default=0,
                        help='The weight of the custom loss')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    logging.info('Using device {}'.format(device))

    estimator = Estimator(size, device, n_channels=2, n_classes=2, bilinear=True)
    bsnet = BSNet([int(size[0]/2), int(size[1]/2)], n_channels=2, n_classes=2, bilinear=True)
    #logging.info('Network:\n\t{} input channels\n\t{} output channels (classes)\n\tnet.bilinear = {} upscaling'.format(net.n_channels, net.n_classes, net.bilinear))

    if args.load:
        bsnet.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {}'.format(args.load))
    else:
        print('Need a pretrained BSNet')
        sys.exit(0)

    bsnet.to(device=device)
    estimator.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(bsnet=bsnet,
                  estimator=estimator,
                  epochs=args.epochs,
                  weight = args.weight,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device)
    except KeyboardInterrupt:
        torch.save(estimator.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
