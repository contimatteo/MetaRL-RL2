import gym

###


def __simulation(env_name: str):
    env = gym.make(env_name)
    env.reset()

    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())


def test():
    __simulation('Ant-v4')
    __simulation('HalfCheetah-v4')
    __simulation('Hopper-v4')
    __simulation('HumanoidStandup-v4')
    __simulation('Humanoid-v4')
    __simulation('InvertedDoublePendulum-v4')
    __simulation('InvertedPendulum-v4')
    __simulation('Reacher-v4')
    __simulation('Swimmer-v4')
    __simulation('Walker2d-v4')


###

if __name__ == "__main__":
    test()
