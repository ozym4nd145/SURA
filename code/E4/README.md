## Training
python3 train.py --inception_checkpoint ../../dataset/inception_v4.ckpt --save_freq 300

## Validation
export CUDA_VISIBLE_DEVICES=""
python3 evaluate.py --eval_interval_secs 300 --min_global_step 200 --max_batch_process 8 --eval_all_models


## Notes
1. removing dense layer berfore RNN input (maybe helps with overfitting)
