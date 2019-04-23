# Training strategy
0. Prepare your dataset by modify the names of related folders in codes of the folder named 'dataloader'. your dataset should include RGB images, surface normal maps and sparse(LiDAR) depth maps.

1. Use the following command to train the part I of our net on [1].
```
python trainN.py --datapath (your surface normal folder)\
               --epochs 20\
               --loadmodel (optional)\
               --savemodel (path for saving model)
```
2. Use the following command to train the part II of our net on KITTI(after load the parameters of [2]).
```
python trainD.py --datapath (your KITTI dataset folder)\
               --epochs 20\
               --batch_size 6\
               --gpu_nums 3\
               --loadmodel (optional)\
               --savemodel (path for saving model)
```
3. Use the following command to train the part III of our net on KITTI(after load the parameters of [2,3]).
```
python train.py --datapath (your KITTI dataset folder)\
               --epochs 20\
               --batch_size 6\
               --gpu_nums 3\
               --loadmodel (optional)\
               --savemodel (path for saving model)
