import os
import os.path
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
parent_path  = os.path.dirname(ROOT_DIR)
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):
    imagesl = []
    normalS = []
    normal_gts = []
    temp = filepath

    filepathl = temp + 'data_depth_velodyne/train'
    filepathgt = temp + 'gt/out/train'
    seqs = [seq for seq in os.listdir(filepathl) if seq.find('sync') > -1]
    left_fold = '/image_02/data'
    right_fold = '/image_03/data'
    lidar_foldl = '/proj_depth/velodyne_raw/image_02'
    lidar_foldr = '/proj_depth/velodyne_raw/image_03'

    for seq in seqs:
        left_path = os.path.join(filepathl, seq) + left_fold
        right_path= os.path.join(filepathl, seq) + right_fold
        lc= [os.path.join(left_path, img) for img in os.listdir(left_path)]
        lc.sort()

        lc=lc[5:-5]
        rc= [os.path.join(right_path, img) for img in os.listdir(right_path)]
        rc.sort()
        rc=rc[5:-5]
        imagesl = np.append(imagesl, lc)
        imagesl = np.append(imagesl, rc)

        gt_path = os.path.join(filepathgt, seq)
        lids2l = os.path.join(filepathl, seq) + lidar_foldl
        lidar2l = [os.path.join(lids2l, lid) for lid in os.listdir(temp)]
        lidar2l.sort()
        normalS = np.append(normalS, lidar2l)
        lids2r = os.path.join(filepathl, seq) + lidar_foldr
        lidar2r = [os.path.join(lids2r, lid) for lid in os.listdir(temp)]
        lidar2r.sort()
        normalS = np.append(normalS, lidar2r)

        gt_imgs = [os.path.join(gt_path, norm) for norm in os.listdir(gt_path)]
        gt_imgs.sort()
        normal_gts= np.append(normal_gts, gt_imgs)
        normal_gts= np.append(normal_gts, gt_imgs)

    left_train = imagesl
    normalS_train = normalS
    return left_train,normalS_train,normal_gts

if __name__ == '__main__':
    datapath = ''