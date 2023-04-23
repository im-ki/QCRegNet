import numpy as np
import torch
import cv2
cv2.setNumThreads(0)
from scipy.ndimage import gaussian_filter
import random
import pickle
import matplotlib.pyplot as plt
from utils import mu2A, plot_map, mu2map
from torch.utils.data import Dataset
from PIL import Image

class QCM_Gen(Dataset):
    def __init__(self, dataset_file_path, output_size, multi_scale = None, half=False):
        self.output_size = output_size
        if multi_scale is None:
            self.multi_scale = output_size
            self.zoom = 0
        else:
            self.multi_scale = multi_scale
            self.zoom = 1
        
        if dataset_file_path[-3:] == 'pkl':
            with open(dataset_file_path, 'rb') as f:
                images = pickle.load(f)
            self.images = np.array([cv2.resize(img, tuple(self.multi_scale)) for img in images])
            #self.images = np.array([np.array(Image.fromarray(img).resize(tuple(self.multi_scale), resample=Image.BILINEAR)) for img in images])
            self.num_sample = self.images.shape[0]
            
        # elif dataset_file_path[-3:] == 'npy':
        #     self.images = np.load(dataset_file_path)
        #     self.num_sample = self.images.shape[0]
        else:
            self.paths = []
            # Read the dataset file
            dataset_file = open(dataset_file_path)
            lines = dataset_file.readlines()
            for line in lines:
                items = line.split()
                self.paths.append(items[0])
	    
            self.read_image()
            self.num_sample = len(self.images)

        self.mask = np.zeros((int(self.output_size[0]/2), int(self.output_size[1]/2)), dtype = np.float32)
        #self.mask[1:-2, 1:-2] = 0.2
        self.mask[2:-3, 2:-3] = 1

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        real = self.process_image(self.images[idx], half = True) * (random.random() + 0.5)
        imag = self.process_image(self.images[random.randint(0, self.num_sample - 1)], half = True) * (random.random() + 0.5)
        I0 = self.process_image(self.images[random.randint(0, self.num_sample - 1)], process = False)
        norm = np.sqrt(np.max(real**2 + imag**2))
        randnum = random.random()

        mu = np.array((real, imag)) / norm * randnum
        mu[:, -1, :] = 0
        mu[:, :, -1] = 0
        #mu *= self.mask

        #mapping, _, _, _, _ = mu2map(mu[0] + 1j * mu[1])
#        return mu, mapping.astype(np.float32), I0[np.newaxis, ...]
        return torch.from_numpy(mu), torch.from_numpy(I0[np.newaxis, ...])

    def process_image(self, img, process = True, half = False):
        if process:
            if half == True:
                output_size = [int(self.output_size[0]/2), int(self.output_size[1]/2)]
            else:
                output_size = self.output_size

            if self.zoom:
                # Resize to random scale
                # new_size = np.random.randint(self.multi_scale[0], self.multi_scale[1], 1)[0]
                new_size_x = self.multi_scale[0]
                new_size_y = self.multi_scale[1]
                # random crop at output size
                diff_size_x = new_size_x - output_size[0]
                diff_size_y = new_size_y - output_size[1]
                random_offset_x = random.randint(0, diff_size_x-1)
                random_offset_y = random.randint(0, diff_size_y-1)
                img = img[random_offset_x:(random_offset_x+output_size[0]),
                          random_offset_y:(random_offset_y+output_size[1])]

        # Flip image at random if flag is selected
            img = img - random.random() * 255
            img = img[::random.choice([1, -1]), ::random.choice([1, -1])] ## flip
            img = gaussian_filter(img.astype(np.float32), sigma=random.choice([1, 2, 3, 4]))
        else:
            img = cv2.resize(img, tuple(self.output_size)).astype(np.float32)
        return img.astype(np.float32)

    def read_image(self):
        paths = self.paths
        images = []
        for i in range(len(paths)):
            img = cv2.imread(paths[i])
            if img is not None:
                for i in range(3):
        # Subtract random num
                    images.append(cv2.resize(img[:, :, i], tuple(self.multi_scale)).astype(np.float32))
            else:
                raise 'Read images exception'
        self.images = np.array(images)


if __name__ == '__main__':
    size = [120, 120]
    gen = QCM_Gen(TRAINING_FILE, size)
    mu, mapping = gen[10]
    print(mu[0][:10, :10], mu[1][:10, :10])
#    out = multi_mu2map(mu_gen(preprocessor))
#    
#    for i in range(datanum):
    plot_map(mapping)

#for i in range(image.shape[0]):
#    plt.imshow(image[i])
#    plt.show()
