
## Download the checkpoints
```sh
cd MCM
sudo apt update
sudo apt install git-lfs
git lfs clone https://huggingface.co/openai/clip-vit-base-patch16
# or you can download the checkpoints from the following link
# git lfs clone https://hf-mirror.com/openai/clip-vit-base-patch16
```


If you download successfully, it can be found in the following path:
`OODD/MCM/clip-vit-base-patch16`

# Data Preparation

 The default dataset location is `OODD/MCM/datasets`. The MCM file structure is as follows:
([ImageNet1K](https://image-net.org/challenges/LSVRC/2012/index.php#) as ID dataset, [iNaturalist](https://arxiv.org/abs/1707.06642), [SUN](https://vision.princeton.edu/projects/2010/SUN/), [Places](https://arxiv.org/abs/1610.02055), and [Texture](https://arxiv.org/abs/1311.3618)  as OOD datasets)
```
MCM
|-- datasets
    |-- ImageNet
        |-- train
        |-- val
    |-- ImageNet_OOD_dataset
        |-- iNaturalist
        |-- dtd
        |-- SUN
        |-- Places
    ...
```
## Run the OODD code
```sh
cd MCM

CUDA_VISIBLE_DEVICES=0 python eval_ood_detection.py --in_dataset ImageNet --name eval_ood --CLIP_ckpt ViT-B/16 --root-dir datasets # for gpu_id=0
```
## Results
The results will be saved in the `OODD/MCM/results` directory. 