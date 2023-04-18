onmt_train -config /storage/HopfieldNMT/config/toy_en_de/toy_en_de_hopfield.yml -world_size 1 -gpu_ranks 0
# onmt_translate -model /storage/HopfieldNMT/toy-ende/run/hopfield/model_step_10.pt -src /storage/HopfieldNMT/toy-ende/src-test.txt -output /storage/HopfieldNMT/toy-ende/pred_1000.txt -gpu 0 -verbose
