export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda-11.4

torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:25702 \
    AntifakePrompt/train.py --cfg-path AntifakePrompt/lavis/projects/textual-inversion/textinv_train.yaml