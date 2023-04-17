onmt_build_vocab -config ../config/toy_en_de/vocab.yaml -n_sample 10000
onmt_train -config ../config/toy_en_de/toy_en_de.yaml -world_size 4 -gpu_ranks 0 1 2 3
onmt_translate -model ../toy-ende/run/model_step_1000.pt -src ../toy-ende/src-test.txt -output ../toy-ende/pred_1000.txt -gpu 0 -verbose