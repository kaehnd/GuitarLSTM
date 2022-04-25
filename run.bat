python3 train.py data/ts9_test1_in_FP16.wav data/ts9_test1_out_FP16.wav fast --training_mode=0 --input_size=150 --create_plots=0
python3 train.py data/ts9_test1_in_FP16.wav data/ts9_test1_out_FP16.wav middle --training_mode=1 --input_size=150 --create_plots=0
python3 train.py data/ts9_test1_in_FP16.wav data/ts9_test1_out_FP16.wav accuracy --training_mode=2 --input_size=150 --create_plots=0
