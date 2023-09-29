export CUDA_VISIBLE_DEVICES=1

question="Is this photo real?"
data_csv="/eva_data0/denny/textual_inversion/debug_label.csv"
log="log.txt"

python LAVIS/deepfake-detection/test.py --question $question --data_csv $data_csv --log $log