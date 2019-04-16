import os
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import skimage
import skimage.io
import skimage.transform
import preprocess
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    img = Image.open(path).convert('RGB')
    return img
def input_loader(path):
    img = skimage.io.imread(path)
    imgG = skimage.color.rgb2gray(img)
    img = img.astype(np.float32)
    normals = img * 1.0 / 127.5 - np.ones_like(img) * 1.0

    mask = np.zeros_like(img).astype(np.float32)
    mask[:, :, 0] = np.where(imgG > 0, 1.0, 0.0)
    mask[:, :, 1] = np.where(imgG > 0, 1.0, 0.0)
    mask[:, :, 2] = np.where(imgG > 0, 1.0, 0.0)

    return normals,mask

def sparse_loader(path):
    img = skimage.io.imread(path)
    img = img * 1.0 / 256.0

    mask = np.where(img > 0.0, 1.0, 0.0)
    mask = np.reshape(mask, [img.shape[0], img.shape[1], 1])
    mask = mask.astype(np.float32)
    img = np.reshape(img, [img.shape[0], img.shape[1], 1])
    return img,mask

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
def scale_crop2(normalize=__imagenet_stats):
    t_list = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    #if scale_size != input_size:
    #t_list = [transforms.Scale((960,540))] + t_list

    return transforms.Compose(t_list)

class myImageFloder(data.Dataset):
    def __init__(self, left, normal, gts, training, model, loader=default_loader, inloader = input_loader,sloader = sparse_loader):
        self.left = left
        self.normal = normal
        self.gts = gts
        self.loader = loader
        self.inloader = inloader
        self.sloader = sloader
        self.training = training
        self.mode = model
    def __getitem__(self, index):
        left = self.left[index]
        normal = self.normal[index]
        gt = self.gts[index]
        left_img = self.loader(left)
        w,h = left_img.size
        input1,mask1 = self.inloader(gt)
        sparse,mask = self.sloader(normal)

        th, tw = 256, 512
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
        data_in1 = input1[y1:y1 + th, x1:x1 + tw,:]
        sparse_n = sparse[y1:y1 + th, x1:x1 + tw,:]
        mask = mask[y1:y1 + th, x1:x1 + tw,:]
        mask1 = mask1[y1:y1 + th, x1:x1 + tw, :]

        processed = preprocess.get_transform(augment=False)
        # processed = scale_crop2()
        left_img = processed(left_img)
        sparse_n = processed(sparse_n)
        return left_img,sparse_n,mask,mask1,data_in1
    def __len__(self):
        return len(self.left)


if __name__ == '__main__':
    print("")
