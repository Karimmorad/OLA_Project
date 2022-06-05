from Advertising.Environment import *
from Advertising.setting_manager import *
from Advertising.GP_Learner import *
from Advertising.dp_optimization import *
import numpy as np
import matplotlib.pyplot as plt

opt_super_arm_reward = 0


def run_dp(products, budgets, features, click_functions,
           alphas, sigma):
    dp_opt = Environment(products, budgets, features, click_functions, alphas, sigma)

    # for feature_label in self.feature_labels:
    #     opt_env.add_subcampaign(label=feature_label, functions=click_functions[feature_label])

    real_values = dp_opt.round_all()
    opt_super_arm = knapsack_optimizer(real_values, budgets[0])

    opt_super_arm_reward = 0

    for (subc_id, pulled_arm) in enumerate(opt_super_arm):
        reward = dp_opt.campaigns[subc_id].round(pulled_arm)
        opt_super_arm_reward += reward

    return [get_dataframe(real_values, opt_super_arm, budgets[0]), opt_super_arm_reward]


env = setting_manager()
products = env.product
margins = env.margin
features = env.features
alphas = env.alphas
sigma = 5
click_functions = env.click_functions

max_budget = 50
n_arms = 6

budgets = []
for i in range(len(products)):
    budgets.append(np.linspace(0, max_budget / 5, n_arms))

real_values = None


# # All the parameters are known
# # results contain the table and the optimal rewards for the optimal arm
# results = run_dp(products, budgets, features, click_functions,
#                  alphas, sigma)
# df_table = results[0]
# opt_super_arm_reward = results[1]
# print(df_table)
# print(opt_super_arm_reward)

