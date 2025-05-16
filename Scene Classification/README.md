# Scene Classification
- CUDA 11.8
- python 3.8.20
- torch 2.11
- torchvision 0.16.1 
- torchaudio 2.1.1
- timm 0.4.12
- mamba-ssm 1.1.1
- causal-conv1d 1.4.0

The installation procedure is as follows:
```
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install packaging
conda install cudatoolkit==11.8
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install nvidia/label/cuda-11.8.0::cuda-cudart-dev
conda install nvidia/label/cuda-11.8.0::libcusparse-dev
conda install nvidia/label/cuda-11.8.0::libcublas-dev
conda install nvidia/label/cuda-11.8.0::libcusolver-dev 
git clone https://github.com/hustvl/Vim.git
cd Vim/causal-conv1d
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .
cd ../mamba-1p1p1
MAMBA_FORCE_BUILD=TRUE pip install .
pip install tensorboard
pip install einops
pip install timm==0.4.12
```
You can also refer to [this](https://blog.csdn.net/yyywxk/article/details/140418043?ops_request_misc=%257B%2522request%255Fid%2522%253A%25224d444fd4d52d5735a2fe62ba0ca93415%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=4d444fd4d52d5735a2fe62ba0ca93415&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-140418043-null-null.142^v100^pc_search_result_base5&utm_term=vmamba%E5%AE%89%E8%A3%85&spm=1018.2226.3001.4187) to solve some problem during installation

## Datasets
The file structure is as follows:
(applicable to <u>AID</u>, <u>UCM,</u> and <u>NWPU</u>)

```python
├── Datasets
|   ├── all_img          # images from all kinds
|   ├── train_labels_55_0.txt           # labels for train
|   ├── valid_labels_55_0.txt           # labesl for val
|   ……             #other label file
```

## Lineprobe
For instance, lineprobe on AID-55 with mamba baseline
```
cd Scene Classification
sh base.sh
```
***You can change the [`bash.sh`]() to test your own setting.***

## Checkpoints

We provide the finetuned models for the -.

The large model will be public soon
