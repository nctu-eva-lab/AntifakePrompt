export CUDA_VISIBLE_DEVICES=1

log="log.txt"

# Classify a single image
img_path="/eva_data0/denny/textual_inversion/debug/0_real/COCO_train2014_000000000009.jpg"
python test.py --img_path $img_path --log $log

# Classify a batch of images
real_dir="/eva_data0/denny/textual_inversion/debug/0_real/"
fake_dir="/eva_data0/denny/textual_inversion/debug/1_fake/"
# python test.py --real_dir $real_dir --fake_dir $fake_dir --log $log
