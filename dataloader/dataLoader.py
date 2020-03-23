from __future__ import print_function

import os
from os.path import join, exists


def dataloader(data_dir, separate_raw_dir=False):
    images = []
    lidars = []
    depths_gt = []
    normals_gt = []

    if separate_raw_dir:
        imgs_root = join(data_dir, 'raw')
    sparse_depth_root = join(data_dir, 'data_depth_velodyne/train')
    depth_gt_root = join(data_dir, 'data_depth_annotated/train')
    normals_gt_root = join(data_dir, 'normals_gt/train')

    seqs = sorted(seq for seq in os.listdir(sparse_depth_root) if seq.endswith('_sync'))

    for seq in seqs:
        date = seq.split('_drive')[0]
        for cam_dir in ('image_02', 'image_03'):
            if separate_raw_dir:
                imgs_path = join(imgs_root, date, seq, cam_dir, 'data')
            else:
                imgs_path = join(sparse_depth_root, seq, cam_dir, 'data')
            lidars_path = join(sparse_depth_root, seq, 'proj_depth/velodyne_raw', cam_dir)
            depth_gt_path = join(depth_gt_root, seq, 'proj_depth/groundtruth', cam_dir)
            normals_gt_path = join(normals_gt_root, seq, cam_dir)

            all_paths_exist = True
            paths = [imgs_path, lidars_path, depth_gt_path, normals_gt_path]
            for path in paths:
                if not exists(path):
                    print("Warning: missing data dir", path)
                    all_paths_exist = False
            if not all_paths_exist:
                continue

            img_files = set(os.listdir(imgs_path))
            lidar_files = set(os.listdir(lidars_path))
            depth_gt_files = set(os.listdir(depth_gt_path))
            normals_gt_files = set(os.listdir(normals_gt_path))

            img_depth_size_diff = 14 if seq == '2011_09_26_drive_0009_sync' else 10
            file_counts = [len(img_files) - img_depth_size_diff, len(lidar_files), len(depth_gt_files),
                           len(normals_gt_files)]
            full_size = max(file_counts)
            for path, file_count in zip(paths, file_counts):
                num_missing = full_size - file_count
                if num_missing != 0:
                    print("Warning:", num_missing, "files missing in", path)

            common_files = sorted(
                img_files &
                lidar_files &
                depth_gt_files &
                normals_gt_files
            )
            images += [join(imgs_path, img) for img in common_files]
            lidars += [join(lidars_path, lid) for lid in common_files]
            depths_gt += [join(depth_gt_path, dep) for dep in common_files]
            normals_gt += [join(normals_gt_path, norm) for norm in common_files]

    return images, lidars, normals_gt, depths_gt


if __name__ == '__main__':
    import sys
    from pprint import pprint

    result = dataloader(sys.argv[1])
    print("Found", len(result[0]), "samples")
    pprint(list(zip(*result))[:3])
