from Advertising.Environment import *
from Advertising.setting_manager import *
from Advertising.GP_Learner import *
from Advertising.dp_optimization import *
import numpy as np
import matplotlib.pyplot as plt


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

    # return [get_dataframe(real_values, opt_super_arm, budgets[0]), opt_super_arm_reward]
    return [real_values, opt_super_arm_reward, opt_super_arm]


def plot_GP_graphs(subc_learners, budgets, real_values):

    x_pred = np.atleast_2d(budgets[0]).T
    for i, subc_learner in enumerate(subc_learners):
        y_pred = subc_learner.means
        sigma = subc_learner.sigmas
        X = np.atleast_2d(subc_learner.pulled_arms).T
        Y = subc_learner.collected_rewards.ravel()
        real_value = real_values[i]
        # title = subc_learner.label

        plt.plot(x_pred, real_value, 'r:', label=r'$click function$')
        plt.plot(X.ravel(), Y, 'ro', label=u'Observed Clicks')
        plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
        plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                 np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% conf interval')
        # plt.title(title)
        plt.xlabel('$budget$')
        plt.ylabel('$daily clicks$')
        plt.legend(loc='lower right')
        plt.show()


def run_experiment(env, nt):

    products = env.product
    margins = env.margin
    features = env.features
    alphas = env.alphas
    sigma = .5
    click_functions = env.click_functions
    max_budget = 50
    n_arms = 6

    budgets = []
    for i in range(len(products)):
        budgets.append(np.linspace(0, max_budget / 5, n_arms))

    env = Environment(products, budgets, features, click_functions, alphas, sigma)
    c_learners = []
    for c_id, product in enumerate(products):
        learner = GPTS_Learner(budgets[0])
        clicks = env.campaigns[c_id].round_all()
        # print(clicks)
        samples = [budgets[0], clicks]

        learner.learn_kernel_hyperparameters(samples)
        c_learners.append(learner)

    rewards = []
    for t in range(nt):

        estimations = []
        for c_learner in c_learners:
            estimate = c_learner.pull_arms()
            # estimate[0] = 0
            estimations.append(estimate)

        super_arm = knapsack_optimizer(estimations, budgets[0])
        super_arm_reward = 0
        for (c_id, pulled_arm) in enumerate(super_arm):
            arm_reward = env.campaigns[c_id].round(pulled_arm)
            super_arm_reward += arm_reward
            c_learners[c_id].update(pulled_arm, arm_reward)

        rewards.append(super_arm_reward)
    return rewards
    # plot_GP_graphs(c_learners, budgets, results[0])


env = setting_manager()
products = env.product
margins = env.margin
features = env.features
alphas = env.alphas
sigma = .5
click_functions = env.click_functions

max_budget = 50
n_arms = 6

budgets = []
for i in range(len(products)):
    budgets.append(np.linspace(0, max_budget / 5, n_arms))

real_values = None


# All the parameters are known
# results contain the table and the optimal rewards for the optimal arm
results = run_dp(products, budgets, features, click_functions,
                 alphas, sigma)

real_values = results[0]
opt_super_arm_reward = results[1]
super_arm_id = results[2]


horizon = 50
n_experiments = 40
gpts_rewards_per_experiment = []
opt_rewards_per_experiment = [opt_super_arm_reward] * horizon

for e in range(0, n_experiments):
    rewards = run_experiment(env, horizon)
    gpts_rewards_per_experiment.append(rewards)


plt.figure()
plt.ylabel("Number of Clicks")
plt.xlabel("t")

opt_exp = opt_rewards_per_experiment
mean_exp = np.mean(gpts_rewards_per_experiment, axis=0)

plt.plot(opt_exp, 'g', label='Optimal Reward')
plt.plot(mean_exp, 'b', label='Expected Reward')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.ylabel("Regret")
plt.xlabel("t")

mean_exp = np.mean(gpts_rewards_per_experiment, axis=0)
regret = np.cumsum(opt_super_arm_reward - mean_exp)

plt.plot(regret, 'r', label='Regret')
plt.legend(loc="upper left")
plt.show()
