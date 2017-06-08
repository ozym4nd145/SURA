from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import time
import math

from model import Model_S2VT
from data_generator import Data_Generator
from inference_util import Inference

import configuration
import inception_base

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("eval_interval_secs", 600,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("min_global_step", 1000,
                        "Minimum global step to run evaluation.")
tf.flags.DEFINE_integer("batch_size", 500,
                        "Number of batches to evaluate at once.")
tf.flags.DEFINE_boolean("eval_all_models",False,
                        "Whether to evaluate all models in checkpoint_dir")

tf.logging.set_verbosity(tf.logging.INFO)

def evaluate_model(sess,model,summary_writer,data_gen):
  data_gen.init_batch(FLAGS.batch_size,"val")
  total_loss = 0

  global_step = sess.run(model.global_step)

  start_time = time.time()
  for i in range(data_gen.iter_per_epoch["val"]):
    dataset = data_gen.get_next_batch("val")
    feed_dict={}
    feed_dict[model.caption_input] = dataset["indexed_caption"]
    feed_dict[model.caption_mask] = dataset["caption_mask"]
    feed_dict[model.rnn_input] = dataset["video"]
    batch_loss = sess.run(model.batch_loss,feed_dict=feed_dict)
    total_loss += batch_loss
    if not i % 2:
      tf.logging.info("Computed losses for %d of %d batches : %.2f", i + 1,
                      data_gen.iter_per_epoch["val"],batch_loss)
  
  eval_time = time.time() - start_time
  
  loss_summary = tf.Summary()
  value = loss_summary.value.add()
  value.simple_value = total_loss/data_gen.iter_per_epoch["val"]
  value.tag = "loss/Batch_Loss"
  summary_writer.add_summary(loss_summary,global_step)

  perplexity = math.exp(total_loss/data_gen.iter_per_epoch["val"])
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

  with tf.Session() as sess:
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
  with g.as_default():
    # Build the model for evaluation.
    model_config = configuration.ModelConfig(data_gen).config
    model = Model_S2VT(**model_config)
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
        tf.logging.info("Starting evaluation of %s at " %(name) + time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()))
        run_once(model, saver, val_writer,data_gen)
    else:
      # Run a new evaluation run every eval_interval_secs.
      while True:
        start = time.time()
        tf.logging.info("Starting evaluation at " + time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()))
        run_once(model, saver, val_writer,data_gen)
        time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
        if time_to_next_eval > 0:
          time.sleep(time_to_next_eval)


def main(unused_argv):
  run()

if __name__ == "__main__":
  tf.app.run()
