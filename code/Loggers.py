from tensorboard_logger import configure, log_value
import os

class FileLogger:
    "Log text in file."
    def __init__(self, path):
        self.path = path

    def log_string(self, file_name, string):
        """Stores log string in specified file."""
        text_file = open(self.path+file_name+".log", "a")
        text_file.write(string+''+str(string)+'\n')
        text_file.close()

    def log_stats(self, file_name, text_to_save, value):
        """Stores log in specified file."""
        text_file = open(self.path+file_name+".log", "a")
        text_file.write(text_to_save+' '+str(value)+'\n')
        text_file.close()


class Logger(object):
    "Tensorboard Logger"
    def __init__(self, log_dir):
        # clean previous logged data under the same directory name
        self._remove(log_dir)

        # configure the project
        configure(log_dir)

        self.global_step = 0

    def log_value(self, name, value):
        log_value(name, value, self.global_step)
        return self

    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path):
        """ param <path> could either be relative or absolute. """
        if os.path.isfile(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)  # remove dir and all contains
