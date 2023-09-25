from envs.uavenv import UAV_ENV
from envs.scenarios import load
from utils.logger import Logger

# Start of Environment
def UAVEnvMain(args):
    print("Load Scenario: ", args.scenario_name)
    # load scenario from script
    scenario = load(args.scenario_name).Scenario()
    # create world (from scenario,py)
    logger = Logger("python_uav.log")  
    logger.debug("Log system for environment just set up...")
    world = scenario.make_world(args, logger)

    logger.debug("Log system just set up...")
    # create multiagent environment
    env = UAV_ENV(world, logger, scenario.reset_world,
                        scenario.reward, scenario.observation, scenario.info)

    return env