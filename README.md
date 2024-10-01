## MS-PCN
Point cloud completion task for marine structure

## Environment

- PyTorch = 1.8.0
- CUDA = 11.1
- Ubuntu 20.04

# Chamfer Distance
bash install.sh
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

##test

[pretrained MS-dataset model](https://pan.baidu.com/s/1fIOayjfnLPwQRVIYS-yf8Q?pwd=f5nt)
[MS-dataset for test](https://pan.baidu.com/s/1DmffrOyEPgOCSwBFC8MAsA?pwd=mh4e)
(More information, please contact the corresponding author via email. email: zxb@sdust.edu.cn)


```
bash ./scripts/test.sh 0 \
    --ckpts ./pretrained/PoinTr_ShapeNet55.pth \
    --config ./cfgs/MS_models/MSPCN.yaml \
    --mode easy/median/hard \ 
    --exp_name example
```

