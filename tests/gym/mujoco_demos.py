#Importing OpenAI gym package and MuJoCo engine
import gym

###


def __run_demo(env_name: str, last: bool = False):
    env = gym.make(env_name)
    env.reset()

    for _ in range(300):
        env.render()
        env.step(env.action_space.sample())

    if last:
        env.close()


def test():
    __run_demo('Ant-v4')
    __run_demo('HalfCheetah-v4')
    __run_demo('Hopper-v4')
    __run_demo('HumanoidStandup-v4')
    __run_demo('Humanoid-v4')
    __run_demo('InvertedDoublePendulum-v4')
    __run_demo('InvertedPendulum-v4')
    __run_demo('Reacher-v4')
    __run_demo('Swimmer-v4')
    __run_demo('Walker2d-v4', last=True)


###

if __name__ == "__main__":
    test()
