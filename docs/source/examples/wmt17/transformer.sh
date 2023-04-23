python3 ../../../onmt/bin/build_vocab.py --config wmt17/wmt17_ende.yaml --n_sample -1
CUDA_VISIBLE_DEVICES=6,7 python3 ../../../onmt/bin/train.py --config wmt17/wmt17_ende.yaml --world_size 2 --gpu_ranks 0 1 
# bash scripts/onmt/train.sh