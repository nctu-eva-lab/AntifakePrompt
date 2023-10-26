export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_HOME=/usr/local/cuda-11.4

torchrun \
    --nnodes=1 \
    --nproc_per_node=3 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:25703 \
    AntifakePrompt/train.py --cfg-path AntifakePrompt/lavis/projects/textual-inversion/textinv_freezeQformer_train.yaml