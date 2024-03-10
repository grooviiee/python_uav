from env.uavenv import UAV_ENV
from env.scenarios import scenario_ref as Scenario
# from envs.scenarios import load


# Start of Environment
def UAVEnvMain(args):
    print("[ENV] Load Scenario: ", args.scenario_name)

    # load scenario from script

    scenario = Scenario

    # create world (from scenario,py)
    world = Scenario.make_world(args)

    # create multiagent environment
    env = UAV_ENV(
        world,
        args.logger,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        scenario.info,
    )

    return env

def make_world(num_mbs, num_uav, num_user):
    return Scenario