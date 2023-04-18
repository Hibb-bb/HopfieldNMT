# onmt_build_vocab -config /storage/HopfieldNMT/config/toy_en_de/vocab.yaml -n_sample 10000
onmt_train -config /storage/HopfieldNMT/config/toy_en_de/toy_en_de_transformer.yml -world_size 1 -gpu_ranks 0
onmt_translate -model /storage/HopfieldNMT/toy-ende/run/model_step_1000.pt -src /storage/HopfieldNMT/toy-ende/src-test.txt -output /storage/HopfieldNMT/toy-ende/pred_1000.txt -gpu 0 -verbose
