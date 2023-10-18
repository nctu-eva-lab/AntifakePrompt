export CUDA_VISIBLE_DEVICES=1

question="Is this photo real?"
log="log.txt"

img_path="/eva_data0/denny/textual_inversion/debug/0_real/COCO_train2014_000000000009.jpg"
real_dir="/eva_data0/denny/textual_inversion/debug/0_real/"
fake_dir="/eva_data0/denny/textual_inversion/debug/1_fake/"

python test.py --question $question --log $log \
    --real_dir $real_dir \
    # --img_path $img_path \
