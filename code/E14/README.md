## Training
python3 train.py 20 --save_freq 500

## Validation
export CUDA_VISIBLE_DEVICES=""
python3 evaluate.py --eval_all_models

## Tensorboard
tensorboard --port 6006 --logdir="./models"
OR
python3 -m tensorflow.tensorboard --logdir="./models"

## Generate Batch results
python3 generate_batch_result.py

## Generate Evalutaion results
python eval.py --eval_all

## Generate Video subtitles
for video in $(ls ../../dataset/test_videos/*.webm)
do
python3 test_video.py ../../dataset/test_videos/$video ./models/train/model-11680 ../../dataset/inception_v4.ckpt --num_fps 10 --len_clip 3 --cap_len 30 --beam_size 3
done

## Notes
1. Made separate encoding decoding stage in lstm. Used multilayered lstms.
2. Added dropout wrapper in RNN. Increasing dropout
3. Added video mask and caption mask use for sequence length in dynamic rnn.
4. With working beam search
5. Batchnorm layer added
6. Inception model removed from training graph


