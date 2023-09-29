real_dir=""
fake_dir=""
real_label="yes"
fake_label="no"
out=""

python LAVIS/utils/gen_path_label.py --real_dir $real_dir --fake_dir $fake_dir \
    --fake_label $fake_label --real_label $real_label --out $out