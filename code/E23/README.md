## Training
python3 train.py 20 --save_freq 500

## Validation
export CUDA_VISIBLE_DEVICES=""
python evaluate.py --eval_all_models

## Tensorboard
tensorboard --port 6006 --logdir="./models"

## Generate Batch results
python generate_batch_result.py

## Generate Evalutaion results
python eval.py --eval_all

## Generate Video subtitles
for video in $(ls ../../dataset/test_videos/*.webm)
do
python3 test_video.py ../../dataset/test_videos/$video ./models/train/model-11680 ../../dataset/inception_v4.ckpt --num_fps 10 --len_clip 3 --cap_len 30 --beam_size 3
done


$files = Get-ChildItem "./temp" -Filter *.wmv 
for ($i=0; $i -lt $files.Count; $i++) {
python test_video.py --num_fps 10 --len_clip 10 --cap_len 30 --beam_size 3 --batch_size 100 ($files[$i].FullName) ./models/train/model-32540 ../../dataset/inception_v4.ckpt 
}

python test_video.py --num_fps 10 --len_clip 4 --cap_len 30 --beam_size 4 ./($files[$i].Name) ./models/train/model-32540 ../../dataset/inception_v4.ckpt 

$files = Get-ChildItem "../../dataset/test_videos/" -Filter *.webm 
for ($i=0; $i -lt $files.Count; $i++) {
python test_video.py --num_fps 10 --len_clip 4 --cap_len 30 --beam_size 4 ($files[$i].FullName) ./models/train/model-32540 ../../dataset/inception_v4.ckpt 
}

## Notes
1. Used glove embedding for words
2. Attention model used
3. Proper Attention this time
4 Constant embedding
