import tensorflow as tf
import datetime
import logging
import numpy as np
import os
import sys
import threading
import boto3
from munch import munchify
import yaml
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class LogEvents:
    """
    Class to log events for Tensorboard
    """

    def __init__(self, log_dir, name):
        """
        Initialize the folders and writers to write the events
        :param log_dir: The log dir
        """
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        full_log_dir = '{}/{}-{}/'.format(log_dir, current_time, name)
        self.summary_writer = tf.summary.create_file_writer(full_log_dir)

    def log_train(self, step, loss):
        self.__log(step, 'train_loss', loss)
        # self.__log(step, 'train_accuracy', accuracy)

    def log_test(self, step, loss, accuracy):
        self.__log(step, 'test_loss', loss)
        self.__log(step, 'test_accuracy', accuracy)

    def __log(self, step, name, value):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, step=step)

    def reset(self, train_loss):
        train_loss.reset_states()


class CheckpointManager:
    """
    Class to manage the checkpoints
    """

    # def __init__(self, model, optimizer, config):
    #    self.__init__(model, optimizer, config.checkpoint.dir, config.checkpoint.name, config.checkpoint.max_to_keep)

    def __init__(self, model, optimizer, model_name, dir="checkpoints", max_to_keep=3):
        """
        Initialize the parameter for the checkpoint manager
        :param model: The model to train and save weights
        :param optimizer: The model optimizer
        """
        log.info("--- Checkpoint ---")
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, directory=dir,
                                                  checkpoint_name=model_name,
                                                  max_to_keep=max_to_keep)
        self.last_loss = np.Inf
        self.model = model
        self.dir = dir
        self.model_name = model_name

        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            last_epoch = int(self.ckpt.step.numpy())
            log.info("* Restore checkpoint #{} at epoch {}".format(self.manager.latest_checkpoint, last_epoch))
        else:
            log.info("* No checkpoint found for '{}'. Initializing from scratch.".format(model_name))

    def get_last_epoch(self):
        """
        Get the last save epoch
        :return: The last epoch number
        """
        last_epoch = 0
        if self.manager.latest_checkpoint:
            last_epoch = int(self.ckpt.step.numpy())
        return last_epoch

    def save(self, loss):
        """
        Save the current checkpoint
        :param loss: The loss to compare with the best checkpoint
        """
        self.ckpt.step.assign_add(1)
        self.__save_best(loss)
        save_path = self.manager.save()
        log.info("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))

    def __save_best(self, loss):
        """
        Save the checkpoint if the loss is better than the previous one
        :param loss: The loss to compare
        """
        if loss < self.last_loss:
            log.info("Saving best weights")
            filename = "{}/best-weights/{}-{}".format(self.dir, self.model_name, int(self.ckpt.step))
            self.model.save_weights(filename)
            self.last_loss = loss
            # TODO Push to S3
            # S3.upload(filename)


class ConfigManager:
    """
    Class to load the configuration from a file
    """

    def __init__(self, conf_file):
        """
        Initialize the file to read the parameters from
        :param conf_file: The name of the file: 'original' or 'test'
        """
        log.info("--- Configuration file ---")
        config_file = "original"
        if conf_file.lower() == "test":
            config_file = "test"

        log.info("* Loading configuration file '{}'".format(config_file))
        self.config = munchify(yaml.safe_load(open("config/{}.yml".format(config_file))))
        log.info("** Loaded")

    def get_conf(self):
        """
        Get the configuration parameters
        :return: The configuration parameters
        """
        return self.config


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


####
def quiet_tf():
    print("Disable TF verbose logs")
    tf.get_logger().setLevel('INFO')