import logging


def SetLogger(dir):
    print("set logger...")
    logger = logging.getLogger("simple_example")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename=dir)
    fh.setLevel(logging.INFO)

    logger.addHandler(ch)
    logger.addHandler(fh)
    formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

    return logger
