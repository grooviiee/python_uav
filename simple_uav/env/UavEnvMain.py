from env.uavenv import UAV_ENV
# from envs.scenarios import load


# Start of Environment
def UAVEnvMain(args):
    print("[ENV] Load Scenario: ", args.scenario_name)

    # load scenario from script
    scenario = load(args.scenario_name).Scenario()

    # create world (from scenario,py)
    world = scenario.make_world(args)

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
