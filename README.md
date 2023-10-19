# AntifakePrompt: Prompt-Tuned Vision-Language Models are Fake Image Detectors

This is the official implementation of AntifakePrompt [paper].

## Introduction

In this paper, being inspired by the zero-shot advantages of Vision-Language Models (VLMs), we propose **AntifakePrompt**, a novel approach using VLMs (e.g. InstructBLIP) and prompt tuning techniques to improve the deepfake detection accuracy over unseen data. We formulate deepfake detection as a visual question answering problem, and tune soft prompts for InstructBLIP to answer the real/fake information of a query image. We conduct full-spectrum experiments on datasets from 3 held-in and 13 held-out generative models, covering modern text-to-image generation, image editing and image attacks. Results demonstrate that (1) the deepfake detection accuracy can be significantly and consistently improved (from 58.8\% to 91.31\%, in average accuracy over unseen data) using pretrained vision-language models with prompt tuning; (2) our superior performance is at less cost of trainable parameters, resulting in an effective and efficient solution for deepfake detection.

<p align="center">
<img src="docs/antifakeprompt.png" width="600">
</p>

## Environment preparation

### Construct the environment

```
git clone https://github.com/thisismingggg/AntifakePrompt.git
cd AntifakePrompt
pip install -e .
```

### Vicuna weights preparation

AntifakePrompt uses frozen Vicuna 7B models. Please first follow the [instructions](https://github.com/lm-sys/FastChat) to prepare Vicuna v1.3 weights. Then modify the `llm_model` in the [Model Config](lavis/configs/models/blip2/blip2_instruct_vicuna7b_textinv.yaml) to the folder that contains Vicuna weights.

### Checkpoints downloading
Install `gdown` package for downloading checkpoints.
```
pip install gdown
```
We provide the best two checkpoints in our experiments:
- COCO+SD2 (150k)
- COCO+SD2+LaMa (180k)

```
cd ckpt
sh download_checkpoints.sh
```
The downloaded checkpoints will be saved in `LAVIS/ckpt`.



| Checkpoint name              | Training dataset  | Average Acc. (%) |
| ---------------------------- |:----------------- | ---------------- |
| COCO_150k_SD2_SD2IP.pth      | COCO + SD2        | 91.59            |
| COCO_150k_SD2_SD2IP_lama.pth | COCO + SD2 + LaMa | 92.60            |


## Testing

### Set the checkpoint path
Go to [Model Config](lavis/configs/models/blip2/blip2_instruct_vicuna7b_textinv.yaml) and set the key value of `model: finetune` to the checkpoint of prompt-tuned model (downloaded in [Checkpoints downloading](#Checkpoints-downloading)).

### Classify a single image

```
python test.py --question <question> --img_path <path_to_image>
```
### Classify batch of images
1. Put the real images in a folder, and put the fake images in another folder.
2. Run the command
```
python test.py --question <question> --real_dir <real_image_directory> --fake_dir <fake_image_directory>
```
If the data only contains real images or fake images, you can just assign one of the arguments between `--real_dir` and `--fake_dir`.



## Training

### Prepare Dataset

Following the steps below, you will get a **.csv file** containing all the image paths and corresponding label for training and testing.

1. Put the real images in a folder, and put the fake images in another folder.
2. Run the command
```
cd LAVIS/utils
python gen_path_label.py --real_dir <real_image_directory> --fake_dir <fake_image_diretory> --real_label <real_label> --fake_label <fake_label>  --out <out_csv_file>
```
- `real_dir` / `fake_dir` : the directory to your real / fake images.
- `real_label` / `fake_label` : the ground truth label for real / fake images.
- `out` : the path to the output .csv file.

3. You will get an output .csv file recording each image path and corresponding ground truth label.
4. Go to [Dataset Config](lavis/configs/datasets/textinv/textinv.yaml), set the `url` and `storage` key value to the path of generated .csv file for train/val/test dataset.

### Start training
1. Go to [Training Config](lavis/projects/textual-inversion/textinv_train.yaml), set the parameters properly. (Please refer to [Training parameters](##Training-parameters) for detail description)
2. Run the command to start training:

```
sh LAVIS/run_scripts/textual-inversion/train.sh
```

## Training parameters

This part list the key parameters for training.
| **Parameter name**                                         | **Description**                                                                                                      |
|:---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| model: prompt                                              | The question prompt for training (including pseudo word).                                                            |
| model: pseudo_word                                         | The word which is the optimization target, so it should be in the question prompt.                                   |
| model: init_word                                           | The intializing word for the pseudo word embedding. (If not specified, the pseudo word will be randomly initailized) |
| datasets: textinv: vis_processor: mean/std                 | The mean/std for image normalization.                                                                                |
| datasets: textinv: text_processor: train/eval: prompt      | The question prompt for training/evaluation (should be the same as model:prompt).                                    |
| datasets: textinv: text_processor: train/eval: pseudo_word | The word which is the optimization target for training/evaluation (should be the same as model:pseudo_word).         |
| run: lr_sched                                              | The type of learning rate scheduler.                                                                                 |
| run: max_epoch                                             | The number of training epochs.                                                                                       |
| run: batch_size_train/eval                                 | The batch size for training/evaluation.                                                                              |

 Please refer to the original [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) repo for the other parameters that are not listed above.
 
 ## Citation
 
 
 ## Acknowledgement

This project is built upon the gaint sholders of InstructBLIP. Great thanks to them!

InstructBLIP: https://github.com/salesforce/LAVIS/tree/main/projects/instructblip