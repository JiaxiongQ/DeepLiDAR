import os
import torch.utils.data as data
import random
from PIL import ImageFile
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
INSTICS = {"2011_09_26": [721.5377, 596.5593, 149.854],
           "2011_09_28": [707.0493, 604.0814, 162.5066],
           "2011_09_29": [718.3351, 600.3891, 159.5122],
           "2011_09_30": [707.0912, 601.8873, 165.1104],
           "2011_10_03": [718.856, 607.1928, 161.2157]
}
# INSTICS = {"NYU": [582.6245, 313.0448, 238.4439]}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    img = skimage.io.imread(path)
    return img
def input_loader(path):
    img = skimage.io.imread(path)
    depth = img *1.0 / 256.0
    depth = np.reshape(depth, [img.shape[0], img.shape[1], 1]).astype(np.float32)
    return depth

def sparse_loader(lidar2_path):
    img2 = skimage.io.imread(lidar2_path)
    img2 = img2 * 1.0 / 256.0
    mask2 = np.where(img2 > 0.0, 1.0, 0.0)
    lidar2 = np.reshape(img2, [img2.shape[0], img2.shape[1], 1]).astype(np.float32)
    return lidar2,mask2


class myImageFloder(data.Dataset):
    def __init__(self, left,sparse,depth,training,
                 loader=default_loader, inloader = input_loader,sloader = sparse_loader):
        self.left = left
        self.sparse = sparse
        self.input = depth
        self.loader = loader
        self.inloader = inloader
        self.sloader = sloader
        self.training = training
    def __getitem__(self, index):
        left = self.left[index]
        input = self.input[index]
        sparse = self.sparse[index]
        left_img = self.loader(left)

        index_str = self.left[index].split('/')[-4][0:10]
        params_t = INSTICS[index_str]
        params = np.ones((256,512,3),dtype=np.float32)
        params[:, :, 0] = params[:,:,0] * params_t[0]
        params[:, :, 1] = params[:, :, 1] * params_t[1]
        params[:, :, 2] = params[:, :, 2] * params_t[2]

        h,w,c= left_img.shape
        input1 = self.inloader(input)
        sparse,mask = self.sloader(sparse)

        th, tw = 256,512
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        mask = np.reshape(mask, [sparse.shape[0], sparse.shape[1], 1]).astype(np.float32)
        params = np.reshape(params, [256, 512, 3]).astype(np.float32)

        left_img = left_img[y1:y1 + th, x1:x1 + tw, :]
        data_in1 = input1[y1:y1 + th, x1:x1 + tw,:]
        sparse = sparse[y1:y1 + th, x1:x1 + tw, :]
        mask = mask[y1:y1 + th, x1:x1 + tw,:]
        processed = preprocess.get_transform(augment=False)

        left_img = processed(left_img)
        sparse = processed(sparse)
        mask = processed(mask)

        return left_img,data_in1,sparse,mask,params

    def __len__(self):
        return len(self.left)


if __name__ == '__main__':
    print("")
