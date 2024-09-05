# HAformer

This repo is the official code of **HAformer: Semantic fusion of hex machine code and assembly code for cross-architecture binary vulnerability detection**. 

## Get Started
### Requirements
- Linux
- Python 3.8+
- PyTorch 1.10+
- CUDA 10.2+
- IDA pro 7.5+ (only used for dataset processing)

### Quick Start

#### 1. Create a conda virtual environment, installing PyTorch and activating it.
```
conda create -n haformer pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 tqdm -c pytorch -c nvidia -y
conda activate haformer
```

#### 2. Install transformers package.
```
pip install transfromers
```

#### 3. Get code and models of HAformer.
```
git clone https://github.com/haformer/haformer.git && cd haformer
```

Download `haformer-base-bin.zip` [baidu drive](https://pan.baidu.com/s/1WxOi5ICCWezrWTz-9TWmLA?pwd=gcvh) or [google drive](https://drive.google.com/file/d/1M2_XJncj-wOYSsDsLPKIK1YcQpQpT6b5/view?usp=drive_link) and extract them `./output/model/finetune/haformer-base-bin/`. 

```
unzip haformer-base-bin.zip
```

## Acknowledgement

* [transformers](https://github.com/huggingface/transformers)
* [jTrans](https://github.com/vul337/jTrans)
* [BinKit](https://github.com/SoftSec-KAIST/BinKit)
* [binary_function_similarity](https://github.com/Cisco-Talos/binary_function_similarity)
* [SAFE](https://github.com/gadiluna/SAFE)
