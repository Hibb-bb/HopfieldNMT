cd docs/source/examples
bash wmt17/prepare_wmt_ende_data.sh
python3 ../../../onmt/bin/build_vocab.py --config wmt17/wmt17_ende2.yaml --n_sample -1
python3 ../../../onmt/bin/train.py --config wmt17/wmt17_ende2.yaml --world_size 3 --gpu_ranks 0 1 2

python3 ../../../onmt/bin/translate.py --src wmt17_en_de/test.src.bpe --model wmt17_en_de/sparse-wmt17_step_50000.pt --beam_size 5 --batch_size 4096 --batch_type tokens --output wmt17_en_de/pred.trg.bpe --gpu 0
sed -re 's/@@( |$)//g' < wmt17_en_de/pred.trg.bpe > wmt17_en_de/pred.trg.tok
sacrebleu -tok none wmt17_en_de/test.trg < wmt17_en_de/pred.trg.tok

python3 ../../../onmt/bin/translate.py --src wmt17_en_de/test.src.bpe --model wmt17_en_de/sparse-wmt17_step_45000.pt --beam_size 5 --batch_size 4096 --batch_type tokens --output wmt17_en_de/pred.trg.bpe --gpu 0
sed -re 's/@@( |$)//g' < wmt17_en_de/pred.trg.bpe > wmt17_en_de/pred.trg.tok
sacrebleu -tok none wmt17_en_de/test.trg < wmt17_en_de/pred.trg.tok

python3 ../../../onmt/bin/translate.py --src wmt17_en_de/test.src.bpe --model wmt17_en_de/sparse-wmt17_step_40000.pt --beam_size 5 --batch_size 4096 --batch_type tokens --output wmt17_en_de/pred.trg.bpe --gpu 0
sed -re 's/@@( |$)//g' < wmt17_en_de/pred.trg.bpe > wmt17_en_de/pred.trg.tok
sacrebleu -tok none wmt17_en_de/test.trg < wmt17_en_de/pred.trg.tok

python3 ../../../onmt/bin/translate.py --src wmt17_en_de/test.src.bpe --model wmt17_en_de/sparse-wmt17_step_35000.pt --beam_size 5 --batch_size 4096 --batch_type tokens --output wmt17_en_de/pred.trg.bpe --gpu 0
sed -re 's/@@( |$)//g' < wmt17_en_de/pred.trg.bpe > wmt17_en_de/pred.trg.tok
sacrebleu -tok none wmt17_en_de/test.trg < wmt17_en_de/pred.trg.tok

python3 ../../../onmt/bin/translate.py --src wmt17_en_de/test.src.bpe --model wmt17_en_de/sparse-wmt17_step_30000.pt --beam_size 5 --batch_size 4096 --batch_type tokens --output wmt17_en_de/pred.trg.bpe --gpu 0
sed -re 's/@@( |$)//g' < wmt17_en_de/pred.trg.bpe > wmt17_en_de/pred.trg.tok
sacrebleu -tok none wmt17_en_de/test.trg < wmt17_en_de/pred.trg.tok
