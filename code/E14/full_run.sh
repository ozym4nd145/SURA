python3 train.py 20 --save_freq 800 --checkpoint_model ./models/train/model-61080
python3 generate_batch_result.py
python eval.py --eval_all
