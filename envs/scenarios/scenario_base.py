


class BaseScenario(object):
    # Create Elements of the world.. It will be used as common settings
    def make_world(self, args, logger):
        raise NotImplementedError()

    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()

    def info(self, agent, world):
        return {}
