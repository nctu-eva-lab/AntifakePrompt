# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_HOME=/usr/local/cuda-11.4

# python -m torch.distributed.run --nproc_per_node=2 /home/denny/LAVIS/train.py --cfg-path /home/denny/LAVIS/lavis/projects/textual-inversion/textinv_freezeLLM_train.yaml

torchrun \
    --nnodes=1 \
    --nproc_per_node=3 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:25703 \
    /home/denny/LAVIS/train.py --cfg-path /home/denny/LAVIS/lavis/projects/textual-inversion/textinv_freezeQformer_train.yaml

    # --standalone \