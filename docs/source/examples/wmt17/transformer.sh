python3 ../../../onmt/bin/build_vocab.py --config wmt17/wmt17_ende.yaml --n_sample -1
python3 ../../../onmt/bin/train.py --config wmt17/wmt17_ende.yaml --world_size 4 --gpu_ranks 0 1 2 3
bash scripts/onmt/train.sh