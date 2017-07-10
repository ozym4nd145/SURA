## Training
python3 train.py 20 --save_freq 500

## Validation
export CUDA_VISIBLE_DEVICES="" or set CUDA_VISIBLE_DEVICES=""
python3 evaluate.py --eval_all_models

## Tensorboard
tensorboard --port 6007 --logdir="./models"

## 

## Notes
1. Separated Encoding and Decoding stage in training.
2. Added dropout wrapper in RNN. Increasing dropout
3. Added video mask and caption mask use for sequence length in dynamic rnn.
4. With working beam search
5. Batchnorm layer added
6. Inception model removed from training graph
7 Without Fully Connected


