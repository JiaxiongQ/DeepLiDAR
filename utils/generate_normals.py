from __future__ import print_function, absolute_import, division

import os
from multiprocessing import Pool, cpu_count
from os.path import join, exists

from surface_normal import normals_from_depth
from tqdm.auto import tqdm

INTRINSICS = {
    "2011_09_26": (721.5377, 609.5593, 172.8540),
    "2011_09_28": (707.0493, 604.0814, 180.5066),
    "2011_09_29": (718.3351, 600.3891, 181.5122),
    "2011_09_30": (707.0912, 601.8873, 183.1104),
    "2011_10_03": (718.8560, 607.1928, 185.2157),
}


def _process(args):
    normals_from_depth(*args)
    return args[1]


def generate_normals(data_dir, output_dir=None, num_cores=cpu_count(), window_size=15, max_rel_depth_diff=0.1):
    depth_gt_root = join(data_dir, 'data_depth_annotated/train')
    normals_gt_root = output_dir or join(data_dir, 'normals_gt/train')

    process_args = []

    seqs = sorted(seq for seq in os.listdir(depth_gt_root) if seq.endswith('_sync'))
    for seq in seqs:
        date = seq.split('_drive')[0]
        for cam_dir in ('image_02', 'image_03'):
            depths_path = join(depth_gt_root, seq, 'proj_depth/groundtruth', cam_dir)
            normals_path = join(normals_gt_root, seq, cam_dir)
            if not exists(normals_path):
                os.makedirs(normals_path)
            img_files = os.listdir(depths_path)
            for img_file in img_files:
                depth_file_in = join(depths_path, img_file)
                normals_file_out = join(normals_path, img_file)
                if exists(normals_file_out):
                    continue
                process_args.append((
                    depth_file_in,
                    normals_file_out,
                    INTRINSICS[date],
                    window_size,
                    max_rel_depth_diff
                ))

    print("Using", num_cores, "cores")
    pool = Pool(num_cores)
    for _ in tqdm(pool.imap_unordered(_process, process_args), total=len(process_args)):
        pass


if __name__ == '__main__':
    import sys

    generate_normals(sys.argv[1])
