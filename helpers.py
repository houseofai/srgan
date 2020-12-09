import tensorflow as tf
import datetime
import logging
import os
import sys
import threading
import boto3
from munch import munchify
import yaml
from contextlib import nullcontext

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# See fit method
# https://github.com/keras-team/keras/blob/4fd825ddf93d3dc9ecc7b50178ce1d015393bed2/keras/engine/training.py#L834
def callback_tensorboard(config, experiment):
    # Log dir
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    full_log_dir = '{}/{}-{}/'.format(config.tensorboard.log_dir, current_time, experiment)

    return tf.keras.callbacks.TensorBoard(log_dir=full_log_dir,
                                          write_graph=config.tensorboard.write_graph,
                                          write_images=config.tensorboard.write_images,
                                          update_freq=config.tensorboard.update_freq,
                                          profile_batch=config.tensorboard.profile_batch)


def callback_checkpoints(config, experiment):
    checkpoint_filepath = "{}/{}".format(config.checkpoint.dir, experiment)

    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='auto',
        save_best_only=True), checkpoint_filepath


def get_config(conf_file):
    log.info("--- Configuration file ---")
    log.info("* Loading configuration file '{}'".format(conf_file))
    config = munchify(yaml.safe_load(open("config/{}.yml".format(conf_file))))
    log.info("** Loaded")
    return config


#### Boto 3 - S3 ########

def upload(filename):
    pp = ProgressPercentage(filename)
    s3 = boto3.client('s3')
    with open(filename) as f:
        s3.upload_fileobj(f, "ml-ckpts", filename, Callback=pp)


class ProgressPercentage(object):

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


#### Multi-GPU ####

def memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


def get_strategy_scope(strategy):
    ctx = nullcontext()
    if is_multi_gpus():
        ctx = strategy.scope()
    return ctx


def is_multi_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    return gpus and len(gpus) > 1
