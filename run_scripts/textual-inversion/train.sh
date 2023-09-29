export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda-11.4

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:25702 \
    LAVIS/train.py --cfg-path LAVIS/lavis/projects/textual-inversion/textinv_train.yaml