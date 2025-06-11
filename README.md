**DSAD: A Diffusion Model using Semantic and Sketch Information for Anomaly Detection**


<!-- [Haoyang He<sup>1#</sup>](https://scholar.google.com/citations?hl=zh-CN&user=8NfQv1sAAAAJ),
[Jiangning Zhang<sup>1,2#</sup>](https://zhangzjn.github.io),
[Hongxu Chen<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=uFT3YfMAAAAJ),
[Xuhai Chen<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=LU4etJ0AAAAJ),
[Zhishan Li<sup>1</sup>](https://scholar.google.com/citations?hl=zh-CN&user=9g-IRLsAAAAJ),
[Xu Chen<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=1621dVIAAAAJ),
[Yabiao Wang<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=xiK4nFUAAAAJ),
[Chengjie Wang<sup>2</sup>](https://scholar.google.com/citations?hl=zh-CN&user=fqte5H4AAAAJ),
[Lei Xie<sup>1*</sup>](https://scholar.google.com/citations?hl=zh-CN&user=7ZZ_-m0AAAAJ)

(#Equal contribution, *Corresponding author) -->

<!-- [<sup>1</sup>College of Control Science and Engineering, Zhejiang University](http://www.cse.zju.edu.cn/), 
[<sup>2</sup>Youtu Lab, Tencent](https://open.youtu.qq.com/#/open)


Our DSAD will also be supported in [ADer](https://github.com/zhangzjn/ADer) -->

## Abstract
In anomaly detection, methods that employ diffusion models for anomaly localization and reconstruction have demonstrated significant achievements. However, these methods face challenges such as the misclassification of multiple types of anomalies and the inability to effectively reconstruct large-scale anomalies due to the absence of semantic and sketch information from the original images. To tackle these challenges, we propose a  framework, A Diffusion Model using Semantic and Sketch Information for Anomaly Detection (DSAD), which includes a semantic and sketch-guided network (SSG), a pre-trained autoencoder, and Stable Diffusion (SD). Initially, within SSG, we introduce a Semantic & Sketch Feature Fusion Module to enhance the model's comprehension of the original images and present a Multi-scale Feature Fusion Module to maximize reconstruction accuracy. Subsequently, we connect SSG with the denoising network in SD in order to guide the network in reconstructing anomalous regions. Experiments on MVTec-AD dataset demonstrate the effectiveness of our approach which surpasses the state-of-the-art methods.
## 1. Installation

First create a new conda environment

    conda env create -f environment.yaml
    conda activate dsad
    pip3 install timm==0.8.15dev0 mmselfsup pandas transformers openpyxl imgaug numba numpy tensorboard fvcore accimage Ninja
## 2.Dataset
### 2.1 MVTec-AD
- **Create the MVTec-AD dataset directory**. Download the MVTec-AD dataset from [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad). Unzip the file and move them to `./training/MVTec-AD/`. The MVTec-AD dataset directory should be as follows. 

```
|-- training
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
            |-- bottle
                |-- ground_truth
                    |-- broken_large
                        |-- 000_mask.png
                    |-- broken_small
                        |-- 000_mask.png
                    |-- contamination
                        |-- 000_mask.png
                |-- test
                    |-- broken_large
                        |-- 000.png
                    |-- broken_small
                        |-- 000.png
                    |-- contamination
                        |-- 000.png
                    |-- good
                        |-- 000.png
                |-- train
                    |-- good
                        |-- 000.png
        |-- train.json
        |-- test.json
```

### 2.2 VisA
- **Create the VisA dataset directory**. Download the VisA dataset from [VisA_20220922.tar](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar). Unzip the file and move them to `./training/VisA/`. The VisA dataset directory should be as follows. 

```
|-- training
    |-- VisA
        |-- visa
            |-- candle
                |-- Data
                    |-- Images
                        |-- Anomaly
                            |-- 000.JPG
                        |-- Normal
                            |-- 0000.JPG
                    |-- Masks
                        |--Anomaly 
                            |-- 000.png        
        |-- visa.csv
```

## 3. Finetune the Autoencoders
- Finetune the Autoencoders first by downloading the pretrained Autoencoders from [kl-f8.zip](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip). Move it to `./models/autoencoders.ckpt`.
And finetune the model with running


`python finetune_autoencoder.py`

- Once finished the finetuned model is under the folder `./lightning_logs/version_x/checkpoints/epoch=xxx-step=xxx.ckpt`.
Then move it to the folder with changed name `./models/mvtec_ae.ckpt`. The same finetune process on VisA dataset.
- If you use the given pretrained autoencoder model, you can go step 4 to build the model.

| Autoencoder        | Pretrained Model                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------------|
| MVTec First Stage Autoencoder | [mvtecad_fs](https://drive.google.com/file/d/1vDfywjGqoWRHMxj-5fifujK29_XyHuCQ/view?usp=sharing) |
| VisA First Stage Autoencoder  | [visa_fs](https://drive.google.com/file/d/1zycpAbWwIVodwTo0Bh1oK8xKliuTT3ul/view?usp=sharing)    |

## 4. Build the model
- We use the pre-trianed stable diffusion v1.5, the finetuned autoencoders and the Semantic-Guided Network to build the full needed model for training.
The stable diffusion v1.5 could be downloaded from ["v1-5-pruned.ckpt"](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). Move it under the folder `./models/v1-5-pruned.ckpt`. 
Then run the code to get the output model `./models/dsad.ckpt`.

`python build_model.py`


## 5. Train
- Training the model by simply run

`python train.py`
- Batch size, learning rate, data path, gpus, and resume path could be easily edited in `train.py`.


## 6. Test
The output of the saved checkpoint could be saved under `./val_ckpt/epoch=xxx-step=xxx.ckpt`For evaluation and visualization, set the checkpoint path `--resume_path` and run the following code:

`python test.py --resume_path ./val_ckpt/epoch=xxx-step=xxx.ckpt`

The images are saved under `./log_image/, where
- `xxx-input.jpg` is the input image.
- `xxx-reconstruction.jpg` is the reconstructed image through autoencoder without diffusion model.
- `xxx-features.jpg` is the feature map of the anomaly score.
- `xxx-samples.jpg` is the reconstructed image through the autoencoder and diffusion model.
- `xxx-heatmap.png` is the heatmap of the anomaly score.


## Acknowledgements
We thank the great works [UniAD](https://github.com/zhiyuanyou/UniAD), [LDM](https://github.com/CompVis/latent-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet) for providing assistance for our research.
