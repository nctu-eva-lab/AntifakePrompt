export CUDA_VISIBLE_DEVICES=6,7
export CUDA_HOME=/usr/local/cuda-11.4

python -m torch.distributed.run --nproc_per_node=2 /home/denny/LAVIS/train.py --cfg-path /home/denny/LAVIS/lavis/projects/textual-inversion/textinv_freezeLLM_train.yaml