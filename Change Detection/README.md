# Change Detection

## 1. Environment

torch:

```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

main packages:

```txt
mmcv==2.1.0
mmdet==3.3.0
mmengine==0.10.6
mmpretrain==1.2.0
mmsegmentation==1.2.2
timm==1.0.15
#torch==2.1.2+cu121
#torchaudio==2.1.2+cu121
#torchvision==0.16.2+cu121
transformers==4.49.0
```

additional packages:

1. `causal-conv1d`, `mamba-ssm`, please refer to [Vim](https://github.com/hustvl/Vim)

2. `open-cd`

   ```
   # https://github.com/likyoo/open-cd/tree/924feb579f891d741003a3fc7c5c5445efeeffb5
   git clone https://github.com/likyoo/open-cd.git
   cd open-cd
   pip install -v -e .
   ```

## 2. Datasets

The file structure for OSCD  is as follows:

```bash
└── oscd_rgb_patch_dataset
    ├── test
    ├── test_all
    └── train
```

For OSCD, we follow [MTP](https://github.com/ViTAE-Transformer/MTP) to process the original image into patches of 96*96. 

Data link for OSCD: [Baidu](https://pan.baidu.com/s/1dqSYiSje7ue3G1k3mNxv-Q?pwd=x17r) & [Google Drive](https://drive.google.com/drive/folders/183b6K6gk8K3vWAAM7fVhUlPi_dnRVuwM?usp=drive_link)

## 3. Training & Evaluation

Make sure to install the open-cd and change the absolute path in config file before proceeding. After installing, copy the files in open-cd-custom to the  editable open-cd in your environment.

Run the following commands for training and evaluation on OSCD:

```bash
# for training
torchrun --standalone --nproc_per_node=8 --master_port=40004 tools/train.py \
config/mamba_oscd.py --launcher 'pytorch' --cfg-options 'find_unused_parameters'=True
# for evaluation
python -u tools/test.py config/mamba_oscd.py \
mamba_base_oscd.pth \
--work-dir=work_dirs/eval/predict --show-dir=work_dirs/eval/predict/show \
--cfg-options val_cfg=None val_dataloader=None val_evaluator=None
```

## 4. Checkpoints

Checkpoint link for OSCD: [Baidu](https://pan.baidu.com/s/1dqSYiSje7ue3G1k3mNxv-Q?pwd=x17r) & [Google Drive](https://drive.google.com/drive/folders/183b6K6gk8K3vWAAM7fVhUlPi_dnRVuwM?usp=drive_link)