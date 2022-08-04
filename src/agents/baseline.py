from memory import SequentialMemory

###


class Agent():

    def __init__(self) -> None:
        self.episodes_memory = None

    #

    def initialize(self) -> None:
        self.episodes_memory = SequentialMemory()

    def remember(self, state, action, reward, next_state) -> None:
        self.episodes_memory.store(state, action, reward, next_state)

    #

    def train(self) -> None:
        raise NotImplementedError

    def test(self) -> None:
        raise NotImplementedError
