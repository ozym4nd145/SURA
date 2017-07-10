python train.py 20 --save_freq 800
python generate_batch_result.py
for video in $(ls ../../dataset/test_videos/*.webm)
do
python test_video.py ../../dataset/test_videos/$video ./models/train/model-17288 ../../dataset/inception_v4.ckpt --num_fps 10 --len_clip 4 --cap_len 30 --beam_size 3
done
