export CUDA_VISIBLE_DEVICES=7
export CUDA_HOME=/usr/local/cuda-11.4

python -m torch.distributed.run --nproc_per_node=1 --master_port=25642 /home/denny/LAVIS/train.py --cfg-path /home/denny/LAVIS/lavis/projects/textual-inversion/textinv_train.yaml
