from typing import Any

from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Dense, Input

from memory import SequentialMemory
from __types import T_Action, T_State, T_Reward

from .baseline import Agent

###


class CriticNetwork(Model):

    def __init__(self):
        super().__init__()
        self.l1 = Dense(512, activation='relu')
        self.l2 = Dense(256, activation='relu')
        self.out = Dense(1, activation=None)

    def call(self, input_data, training=None, mask=None):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.out(x)
        return x


class ActorNetwork(Model):

    def __init__(self):
        super().__init__()
        self.l1 = Dense(512, activation='relu')
        self.l2 = Dense(256, activation='relu')
        self.out = Dense(2, activation='softmax')

    def call(self, input_data, training=None, mask=None):
        x = self.l1(input_data)
        x = self.l2(x)
        x = self.out(x)
        return x


###


class ActorCritic(Agent):

    def __init__(self, *args, **kwargs) -> None:
        # super(Agent, self).__init__(*args, **kwargs)
        super(ActorCritic, self).__init__(*args, **kwargs)

        self.actor_network = ActorNetwork()
        self.critic_network = CriticNetwork()

    #

    def act(self, _: T_State) -> T_Action:
        return self.env.action_space.sample()

    #

    def __compute_TD_error(self) -> Any:
        return

    def train(self) -> None:
        """
        1. Input: a differentiable policy parametrisation π(a|s, θ), a differentiable state-value function parametrisation v(s, w) 
        2. Parameters: step sizes α^θ > 0 and α^w > 0 
        3. Initialise policy parameter θ ∈ ℝ^d′ and state-value weights w ∈ ℝ^d
        4. Loop forever (for each episode):
            1. Initialise S (ﬁrst state of episode) 
            2. Loop while S is not terminal
                1. Select A using policy π
                2. Take action A, observe S′, R
                3. δ ← R + ̂ v(S′, w) − v(S, w)
                4. w ← w + αw δ ∇ v(S, w)
                5. θ ← θ + α θ δ ∇lnπ(A|S, θ)
                6. S ← S′
        """

        episode = self.memory.all()

        rewards = episode["rewards"]

        print("")
        print(rewards)
        print("")

    def test(self):
        pass
