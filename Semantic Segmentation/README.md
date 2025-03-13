# Semantic Segmentation
- CUDA 11.8
- python 3.8.20
- torch 2.11
- torchvision 0.16.1 
- torchaudio 2.1.1
- timm 0.4.12
- mamba-ssm 1.1.1
- causal-conv1d 1.4.0
- mmcv 2.1.0
- mmcv-full 1.7.2
- mmdet 2.2.0+unknown

About the installation of Mamba, please refer to [README.md](../Scene%20Classification/README.md) in Scene Classification

Since we use MMSegmenation to implement corresponding segmentation models, we only provide necessary config and backbone files. The installation can refer to [MMSegmentation-installation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation).

Jsut put these files into corresponding folders.

For convenience, we preserve the relative path for users to find files.
## Datasets
The SPACENETV1 datasets file structure is as follows:

```python
├── train
|   ├── images
|   ├── labels
├── val
|   ├── images
|   ├── labels
```


## Training & Evaluation

Take SPACENETV1 as an example(with mamba baseline):
```
cd Semantic Segmentation
sh ss.sh
```

***You can change the [`ss.sh`]() to test your own setting.***

## Checkpoints

We provide the pretrained models for the [base](https://pan.baidu.com/s/1EWHjTeJNRr5nRzWkTxP1VA?pwd=yf3n) and [large](https://pan.baidu.com/s/1oNilz9RSpWOGDkS01pdB6g?pwd=qh6r)
