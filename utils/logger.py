import logging


# def Logger():


def SetLogger(dir):
    print("set logger...")
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%d-%m-%Y:%H:%M:%S",
        level=logging.DEBUG,
    )
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
