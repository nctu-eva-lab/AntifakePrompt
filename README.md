# AntifakePrompt

## Preparing Dataset

1. Go to **`LAVIS/utils/gen_path_label.sh`**, you will see the code below:

```python=
real_dir=""
fake_dir=""
real_label="yes"
fake_label="no"
out=""

python LAVIS/utils/gen_path_label.py --real_dir $real_dir --fake_dir $fake_dir \
    --fake_label $fake_label --real_label $real_label --out $out
```
- `real_dir` / `fake_dir` : the directory to your real / fake images.
- `real_label` / `fake_label` : the groung truth label for real / fake images.
- `out` : the path to the output .csv file.

2. After setting all the arguments above, run the command below:
```shell=
sh LAVIS/utils/gen_path_label.sh
```
3. You will get an output .csv file recording each image path and corresponding ground truth label.

## Testing

1. Go to **`LAVIS/lavis/configs/models/blip2/blip2_instruct_vicuna7b_textinv.yaml`**, check the key value of `finetune` (it should be the checkpoint of prompt-tuned model).

2. Go to **`LAVIS/deepfake-detection/test.sh`**, you will see the code below:
```python=
export CUDA_VISIBLE_DEVICES=1

question="Is this photo real?"
data_csv=""
log="log.txt"

python LAVIS/deepfake-detection/test.py --question $question --data_csv $data_csv --log $log
```
- `question` : the question prompt fed into the model.
- `data_csv` : the csv file you just generated from [Preparing Dataset](##Preparing-Dataset).
- `log` : the log file path.

3. After setting all the arguments above, run the command below:
```shell=
sh LAVIS/deepfake-detection/test.sh
```

## Training

1. Go to **`LAVIS/lavis/configs/datasets/textinv/textinv.yaml`**, set the `url` and `storage` key value to the path of generated .csv file for train/val/test dataset.
2. Run the command below:

```shell=
sh LAVIS/run_scripts/textual-inversion/train.sh
```

## Checkpoint

- Checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1JgMJie4wDt7dNeHkT25VVuzG9CdnA9mQ?usp=drive_link).
- Dataset can be downloaded [here](https://drive.google.com/drive/folders/1qVolr_iYy7vZ5SjZBengZ3pncUYLoXBt?usp=drive_link).