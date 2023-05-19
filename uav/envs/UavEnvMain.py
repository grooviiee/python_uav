from envs.uavenv import UAV_ENV
from envs import load

# Start of Environment
def UAVEnvMain(args):
    print("Load Scenario: ", args.scenario_name)
    # load scenario from script
    scenario = load("scenario.py").Scenario()
    # create world (from scenario,py)
    world = scenario.make_world(args)
    # create multiagent environment
    env = UAV_ENV(world, scenario.reset_world,
                        scenario.reward, scenario.observation, scenario.info)

    return env