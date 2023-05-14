# common functions will be moved here


class Runner(object):
    """ Base class for training recurrent policies. """
    
    def __init__(self, config):
        self.all_args = config['all_args']