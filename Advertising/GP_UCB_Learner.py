import numpy as np
from Advertising.GP_Learner import *


class GP_UCB_Learner(GPTS_Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.t = 0

    def pull_arms(self):
        return np.argmax(self.means + self.sigmas)

    # def update(self, pull_arm, reward):
    #     super().update(pull_arm, reward)
    #     self.t += 1
    # #     self.means[pull_arm] = (self.means[pull_arm] * (self.t - 1) + reward) / self.t
    # #     print(self.means)
    #     for a in range(self.n_arms):
    #         n_samples = len(self.collected_rewards)
    #         self.confidence[a] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf
