# OpenNMT-py: Open-Source Neural Machine Translation

## ToDos

### ToRuns

- [ ] Sparsemax Transformer en-de
- [ ] Sparsemax Hopfield en-de
- [ ] Dense Transformer en-de
- [ ] Dense Hopfield en-de
- [ ] Entmax Transformer en-de
- [ ] Entmax Hopfield en-de

___

### ToCodes
- [ ] Summarization CNNDM
- [ ] Language Modeling Hopfield Decoder
- [ ] Check MLM and Autoregressive LM modes
- [ ] Evaluation Metric for NMT, LM and Summarization

## Toy Example

```
wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
tar xf toy-ende.tar.gz
```

## To run experiment for En-De

```
cd docs/source/examples/wmt17
sh prepare_wmt_ende_data.sh
cd ..
python3 ../../../onmt/bin/build_vocab.py --config wmt17/wmt17_ende.yaml --n_sample -1
```

To run standard transformer:
```
CUDA_VISIBLE_DEVICES=0,1 python3 ../../../onmt/bin/train.py --config wmt17/wmt17_ende.yaml --world_size 2 --gpu_ranks 0 1 
```
To run standard hopfield
```
CUDA_VISIBLE_DEVICES=0,1 python3 ../../../onmt/bin/train.py --config wmt17/wmt17_ende2.yaml --world_size 2 --gpu_ranks 0 1 
```

Change the config in `wmt17_ende2.yaml` to make it sparsemax, entmax etc.

The sparsemax, entmax version of transformer is finished yet.

## Tutorials:

* How to finetune NLLB-200 with your dataset: [Tuto Finetune NLLB-200](https://forum.opennmt.net/t/finetuning-and-curating-nllb-200-with-opennmt-py/5238)
* How to create a simple OpenNMT-py REST Server: [Tuto REST](https://forum.opennmt.net/t/simple-opennmt-py-rest-server/1392)
* How to create a simple Web Interface: [Tuto Streamlit](https://forum.opennmt.net/t/simple-web-interface/4527)
* Replicate the WMT17 en-de experiment: [WMT17 ENDE](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/examples/wmt17/Translation.md)

----

## Setup

OpenNMT-py requires:

- Python >= 3.8
- PyTorch >= 1.13 <2

Install `OpenNMT-py` from `pip`:
```bash
pip install OpenNMT-py
```

or from the sources:
```bash
git clone https://github.com/OpenNMT/OpenNMT-py.git
cd OpenNMT-py
pip install -e .
```

Note: if you encounter a `MemoryError` during installation, try to use `pip` with `--no-cache-dir`.

*(Optional)* Some advanced features (e.g. working pretrained models or specific transforms) require extra packages, you can install them with:

```bash
pip install -r requirements.opt.txt
```

## Features

- [End-to-end training with on-the-fly data processing]([here](https://opennmt.net/OpenNMT-py/FAQ.html#what-are-the-readily-available-on-the-fly-data-transforms).)

- [Transformer models](https://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- [Encoder-decoder models with multiple RNN cells (LSTM, GRU) and attention types (Luong, Bahdanau)](https://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- [SRU "RNNs faster than CNN"](https://arxiv.org/abs/1709.02755)
- [Conv2Conv convolution model](https://arxiv.org/abs/1705.03122)
- [Copy and Coverage Attention](https://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Pretrained Embeddings](https://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Source word features](https://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [TensorBoard logging](https://opennmt.net/OpenNMT-py/options/train.html#logging)
- Mixed-precision training with [APEX](https://github.com/NVIDIA/apex), optimized on [Tensor Cores](https://developer.nvidia.com/tensor-cores)
- [Multi-GPU training](https://opennmt.net/OpenNMT-py/FAQ.html##do-you-support-multi-gpu)
- [Inference (translation) with batching and beam search](https://opennmt.net/OpenNMT-py/options/translate.html)
- Model export to [CTranslate2](https://github.com/OpenNMT/CTranslate2), a fast and efficient inference engine

## Documentation

[Full HTML Documentation](https://opennmt.net/OpenNMT-py/quickstart.html)

## Acknowledgements

OpenNMT-py is run as a collaborative open-source project.
Project was incubated by Systran and Harvard NLP in 2016 in Lua and ported to Pytorch in 2017.

Current maintainers:

Ubiqus Team: [François Hernandez](https://github.com/francoishernandez) and Team.

[Vincent Nguyen](https://github.com/vince62s) (Seedfall)

## Citation

If you are using OpenNMT-py for academic work, please cite the initial [system demonstration paper](https://www.aclweb.org/anthology/P17-4012) published in ACL 2017:

```
@inproceedings{klein-etal-2017-opennmt,
    title = "{O}pen{NMT}: Open-Source Toolkit for Neural Machine Translation",
    author = "Klein, Guillaume  and
      Kim, Yoon  and
      Deng, Yuntian  and
      Senellart, Jean  and
      Rush, Alexander",
    booktitle = "Proceedings of {ACL} 2017, System Demonstrations",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P17-4012",
    pages = "67--72",
}
```

