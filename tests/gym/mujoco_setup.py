import gym

###


def __load_gym_env(name: str):
    assert isinstance(name, str)

    env = gym.make(name)

    if env.reset() is None:
        raise Exception(f"[FAILED] {name}")


def test():
    __load_gym_env('Ant-v4')
    __load_gym_env('HalfCheetah-v4')
    __load_gym_env('Hopper-v4')
    __load_gym_env('HumanoidStandup-v4')
    __load_gym_env('Humanoid-v4')
    __load_gym_env('InvertedDoublePendulum-v4')
    __load_gym_env('InvertedPendulum-v4')
    __load_gym_env('Reacher-v4')
    __load_gym_env('Swimmer-v4')
    __load_gym_env('Walker2d-v4')


###

if __name__ == "__main__":
    test()
