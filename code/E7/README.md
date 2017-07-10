## Training
python3 train.py --inception_checkpoint ../../dataset/inception_v4.ckpt --save_freq 300

## Validation
export CUDA_VISIBLE_DEVICES=""
python3 evaluate.py --eval_interval_secs 300 --min_global_step 200 --max_batch_process 8 --eval_all_models


## Notes
1. Separated Encoding and Decoding stage in training.
2. Added dropout wrapper in RNN. Increasing dropout
3. Added video mask and caption mask use for sequence length in dynamic rnn.

