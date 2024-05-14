# Model
Self-supervised Remote Sensing Image Destriping via Constrastive Learning
用中文记录下来

![动图，展示去条纹前后图像对比](这里放图片对比)
# Workflow
![去条纹训练框架](./meta/workflows.png)
# Install
First, you should clone this repostry to your local machine.
```bash
git clone git@github.com:BugBubbles/CraterDestrip.git
```
Then, you should install the dependencies.
```bash
conda envs create -f environment.yaml
```
And activate your conda virtual environment.
```bash
conda activate destrip
```
# Usage
First your should prepare your datasets. For example, I use the LORA datasets in the `/path/to/datasets/LORA`. The datasets should be organized as follows:

## Eval
```bash
. ./scripts/eval.sh
```

## Train
```bash
. ./scripts/train.sh
```

# Acknowledge
ldm, NLCL, LPIPS, SSIM