# AntifakePrompt: Prompt-Tuned Vision-Language Models are Fake Image Detectors

This is the official implementation of AntifakePrompt paper. AntifakePrompt propose a prompt-tuned vision-language model from [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) as a deepfake detector.

![model structure](docs/antifakeprompt.png)

## Installation

```
git clone https://github.com/thisismingggg/LAVIS.git
cd LAVIS
pip install -e .
```

## Preparing Dataset

Following the steps below, you will get a **.csv file** containing all the image paths and corresponding label for training and testing.

1. Go to **`LAVIS/utils/gen_path_label.sh`**, you will see the code below:

```
real_dir=""
fake_dir=""
real_label="yes"
fake_label="no"
out=""

python LAVIS/utils/gen_path_label.py --real_dir $real_dir --fake_dir $fake_dir \
    --fake_label $fake_label --real_label $real_label --out $out
```
- `real_dir` / `fake_dir` : the directory to your real / fake images.
- `real_label` / `fake_label` : the ground truth label for real / fake images.
- `out` : the path to the output .csv file.

2. After setting all the arguments above, run the command below:
```
sh LAVIS/utils/gen_path_label.sh
```
3. You will get an output .csv file recording each image path and corresponding ground truth label.

## Testing

1. Go to **`LAVIS/lavis/configs/models/blip2/blip2_instruct_vicuna7b_textinv.yaml`**, check the key value of `finetune` (it should be the checkpoint of prompt-tuned model).

2. Go to **`LAVIS/deepfake-detection/test.sh`**, you will see the code below:
```
question="Is this photo real?"
data_csv=""
log="log.txt"

python LAVIS/deepfake-detection/test.py --question $question --data_csv $data_csv --log $log
```
- `question` : the question prompt fed into the model.
- `data_csv` : the csv file you just generated from [Preparing Dataset](##Preparing-Dataset).
- `log` : the log file path, which will record the testing result.

3. After setting all the arguments above, run the command below:
```
sh LAVIS/deepfake-detection/test.sh
```

## Training

1. Go to **`LAVIS/lavis/configs/datasets/textinv/textinv.yaml`**, set the `url` and `storage` key value to the path of generated .csv file for train/val/test dataset.
2. Go to **`LAVIS/lavis/projects/textual-inversion/textinv_train.yaml`**, set the parameters properly (please refer to [Training parameters](##Training-parameters) for detail description).
3. Run the command below:

```
sh LAVIS/run_scripts/textual-inversion/train.sh
```

## Checkpoint

- Checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1JgMJie4wDt7dNeHkT25VVuzG9CdnA9mQ?usp=drive_link).

## Training parameters

This part list the key parameters for training.
| **Parameter name**                                         | **Description**                                                                                                      |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| model: prompt                                              | The question prompt for training (including pseudo word).                                                            |
| model: pseudo_word                                         | The word which is the optimization target, so it should be in the question prompt.                                   |
| model: init_word                                           | The intializing word for the pseudo word embedding. (If not specified, the pseudo word will be randomly initailized) |
| datasets: textinv: vis_processor: mean/std                 | The mean/std for image normalization.                                                                                |
| datasets: textinv: text_processor: train/eval: prompt      | The question prompt for training/evaluation (should be the same as model:prompt).                                    |
| datasets: textinv: text_processor: train/eval: pseudo_word | The word which is the optimization target for training/evaluation (should be the same as model:pseudo_word).         |
| run: lr_sched                                              | The type of learning rate scheduler.                                                                                 |
| run: max_epoch                                             | The number of training epochs.                                                                                       |
| run: batch_size_train/eval                                 | The batch size for training/evaluation.                                                                              |

 Please refer to the original [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) repo for the other parameters that not listed above.
