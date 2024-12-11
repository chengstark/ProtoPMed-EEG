import os

def create_logger(log_filename, display=True):
    """
    Create a logger that writes to a file and optionally displays messages.

    Args:
        log_filename (str): The path to the log file where messages will be written.
        display (bool): If True, log messages will also be printed to the console. Default is True.

    Returns:
        tuple: A logger function and a close function to close the log file.
    """
    f = open(log_filename, 'a')
    counter = [0]

    def logger(text):
        """
        Log a message to the file and optionally display it.

        Args:
            text (str): The message to log.
        """
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())

    return logger, f.close
