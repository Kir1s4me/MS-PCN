# MS-PCN
Point cloud completion task for marine structure

## Environment

- PyTorch = 1.8.0
- CUDA = 11.1
- Ubuntu 20.04

### Chamfer Distance
bash install.sh
### PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
### GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl



[pretrained MS-dataset model](https://pan.baidu.com/s/1UGsfQCmkIqe9FoUUD8PGfA?pwd=8x8m)
[MS-dataset](https://pan.baidu.com/s/1Xs4V9SLIebFQ5BYvxRhvOg?pwd=7x4i)


## train

```
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/MS_models/MSPCN.yaml \
    --exp_name EXAMPLE
```

## test

```
bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/PoinTr_ShapeNet55.pth \
    --config ./cfgs/MS_models/MSPCN.yaml \
    --mode easy/median/hard \ 
    --exp_name example
```


More information, please contact the corresponding author via email. email: zxb@sdust.edu.cn

