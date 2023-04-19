def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
        )
    else:
        env = MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation
        )
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    traubers = []
    model = mlp_model
    trainer = MAPPOAgentTrainer


def train(arglist):
    with U.single_threaded_session():
        env = make_env(arglist.scenario, arglist, arglist.benchmark)


if __name__ == "__main__":
    arglist = parse_args()
    train(arglist)
