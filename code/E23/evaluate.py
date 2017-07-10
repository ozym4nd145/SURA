from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import time
import math

from model import Caption_Model
from data_generator import Data_Generator
from inference_util import Inference

import configuration
import inception_base
import argparse

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

def evaluate_model(sess,model,summary_writer,data_gen):
  data_gen.init_batch(FLAGS.batch_size,"val")
  total_loss = 0

  global_step = sess.run(model.global_step)

  start_time = time.time()
  if FLAGS.max_batch_process:
    max_iter = FLAGS.max_batch_process
  else:
    max_iter = data_gen.iter_per_epoch["val"]

  for i in range(max_iter):
    dataset = data_gen.get_next_batch("val")
    feed_dict={}
    feed_dict[model.video_mask] = np.ones([dataset["video"].shape[0],dataset["video"].shape[1]],dtype=np.int32)
    feed_dict[model.caption_input] = dataset["indexed_caption"]
    feed_dict[model.caption_mask] = dataset["caption_mask"]
    feed_dict[model.rnn_input] = dataset["video"]
    feed_dict[model.is_training] = False
    batch_loss = sess.run(model.batch_loss,feed_dict=feed_dict)
    total_loss += batch_loss
    tf.logging.info("Computed losses for %d of %d batches : %.2f", i + 1,
                      max_iter,batch_loss)
  
  eval_time = time.time() - start_time
  
  loss_summary = tf.Summary()
  value = loss_summary.value.add()
  value.simple_value = total_loss/max_iter
  value.tag = "loss/Batch_Loss"
  summary_writer.add_summary(loss_summary,global_step)

  perplexity = math.exp(total_loss/max_iter)
  tf.logging.info("Perplexity = %f (%.2g sec)", perplexity, eval_time)         
  
  perp_summary = tf.Summary()
  value = perp_summary.value.add()
  value.simple_value = perplexity
  value.tag = "Perplexity"
  summary_writer.add_summary(perp_summary,global_step)

  summary_writer.flush()
  tf.logging.info("Finished processing evaluation at global step %d.",
                  global_step)

def run_once(model, saver, summary_writer, data_gen):
  """Evaluates the latest model checkpoint.
  """
  if not FLAGS.eval_all_models:
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if not model_path:
      tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                      FLAGS.checkpoint_dir)
      return
  else:
    model_path = FLAGS.checkpoint_file

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
    # Load model from checkpoint.
          tf.logging.info("Loading model from checkpoint: %s", model_path)
          saver.restore(sess, model_path)
          global_step = sess.run(model.global_step)
          tf.logging.info("Successfully loaded %s at global step = %d.",
                          os.path.basename(model_path), global_step)
          if global_step < FLAGS.min_global_step:
            tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step,
                            FLAGS.min_global_step)
            return

          # Run evaluation on the latest checkpoint.
          evaluate_model(
              sess=sess,
              model=model,
              summary_writer=summary_writer,
              data_gen=data_gen)

def run():
  """Runs evaluation in a loop, and logs summaries to TensorBoard."""
  # Create the evaluation directory if it doesn't exist.

  data_config = configuration.DataConfig().config
  data_gen = Data_Generator(data_config["processed_video_dir"],
                          data_config["caption_file"],
                          data_config["unique_frequency_cutoff"],
                          data_config["max_caption_length"])

  data_gen.load_vocabulary(data_config["caption_data_dir"])
  data_gen.load_dataset(data_config["caption_data_dir"])

  FLAGS.checkpoint_dir = data_config["checkpoint_dir"]

  eval_dir = data_config["val_log_dir"]
  if not tf.gfile.IsDirectory(eval_dir):
    tf.logging.info("Creating eval directory: %s", eval_dir)
    tf.gfile.MakeDirs(eval_dir)

  g = tf.Graph()

  evaluated_models = set([])

  with g.as_default():
    # Build the model for evaluation.
    model_config = configuration.ModelConfig(data_gen).config
    model = Caption_Model(mode="val",**model_config)
    model.build()

    # Create the Saver to restore model Variables.
    saver = tf.train.Saver()

    # Create the summary operation and the summary writer.
    val_writer = tf.summary.FileWriter(data_config["val_log_dir"])

    g.finalize()

    if(FLAGS.eval_all_models):
      model_names = list(set([n.split(".")[0] for n in os.listdir(data_config["checkpoint_dir"]) if "model" in n]))
      model_names.sort(key= lambda x: int(x[6:]) )
      for name in model_names:
        FLAGS.checkpoint_file = os.path.join(data_config["checkpoint_dir"],name)
        evaluated_models.add(FLAGS.checkpoint_file)
        tf.logging.info("Starting evaluation of %s at " %(name) + time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()))
        run_once(model, saver, val_writer,data_gen)

    # Run a new evaluation run every eval_interval_secs.
    while True:
      start = time.time()
      tf.logging.info("Starting evaluation at " + time.strftime(
          "%Y-%m-%d-%H:%M:%S", time.localtime()))
      
      FLAGS.checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
      if not FLAGS.checkpoint_file:
        tf.logging.info("Skipping evaluation. No checkpoint found in: %s",
                    FLAGS.checkpoint_dir)
      elif FLAGS.checkpoint_file in evaluated_models:
        tf.logging.info("Skipping evaluation. No new checkpoint found in: %s",
                    FLAGS.checkpoint_dir)
      else:
        evaluated_models.add(FLAGS.checkpoint_file)
        run_once(model, saver, val_writer,data_gen)
      time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
      if time_to_next_eval > 0:
        time.sleep(time_to_next_eval)


def main(unused_argv):
    run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_interval_secs", help="Interval between evaluation runs.",
                        type=int, default=300)
    parser.add_argument("--min_global_step", help="Minimum global step to run evaluation.",
                        type=int, default=100)
    parser.add_argument("--batch_size", help="Number of batches to evaluate at once.",
                        type=int, default=100)
    parser.add_argument("--max_batch_process", help="Total number of batches to evaluate.",
                        type=int, default=10)
    parser.add_argument("--eval_all_models", help="Whether to evaluate all models in checkpoint_dir",
                        action="store_true")
    FLAGS = parser.parse_args()
    main(None)
