import tensorflow as tf
import datetime
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class LogEvents:
    """
    Class to log events for Tensorboard
    """

    def __init__(self, log_dir):
        """
        Initialize the folders and writers to write the events
        :param log_dir: The log dir
        """
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        full_log_dir = '{}/{}/'.format(log_dir, current_time)
        self.summary_writer = tf.summary.create_file_writer(full_log_dir)

    def log_train(self, step, loss):
        self.__log(step, 'train_loss', loss)
        #self.__log(step, 'train_accuracy', accuracy)

    def log_test(self, step, loss, accuracy):
        self.__log(step, 'test_loss', loss)
        self.__log(step, 'test_accuracy', accuracy)

    def __log(self, step, name, value):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, step=step)

    def reset(self, train_loss):
        train_loss.reset_states()