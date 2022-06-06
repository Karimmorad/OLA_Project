import json
import numpy as np
import matplotlib.pyplot as plt


class setting_manager:
    def __init__(self):
        with open('config.json') as json_file:
            data = json.load(json_file)
        campaigns = data["campaigns"]

        # Product and margin settings
        self.product = []
        self.margin = []
        for i in range(len(campaigns)):
            self.product.append(campaigns[i]["product"])
            self.margin.append(campaigns[i]["margin"])

        # Class settings
        self.features = list(campaigns[0]["subcampaign"].keys())

        # Experiment settings
        self.click_functions = {}
        self.alphas = []
        for i, campaign in enumerate(campaigns):
            prod = campaign['product']
            self.click_functions[prod] = []
            _ = []
            for j, feature in enumerate(self.features):
                alpha = campaign['subcampaign'][feature]['alpha']
                _.append(alpha)
                self.click_functions[prod].append(lambda x, a=alpha: self.n(x, a))
            self.alphas.append(_)

    def n(self, x, a, max_clicks=150):
        return (1 - np.exp(-1.0 * x)) * a * max_clicks


# colors = ['r', 'b', 'black']
# env = setting_manager()
# features = env.features
# products = env.product
#
# budgets = []
# for i in range(len(products)):
#     budgets.append(np.linspace(0, 10, num=11))
#
# x = np.linspace(0, max(budgets[0]), num=550)
#
# fig, axs = plt.subplots(1, 5, figsize=(20, 8))
# for i, product in enumerate(products):
#     for j, label in enumerate(features):
#         y = env.click_functions[product][j](x)
#         scatters = env.click_functions[product][j](budgets[i])
#         axs[i].plot(x, y, color=colors[j], label=label)
#         axs[i].scatter(budgets[i], scatters, color=colors[j])
#         axs[i].set_title("product " + product + " click function")
#         axs[i].set_xlabel("Budget")
#         axs[i].set_ylabel("Number of Clicks")
#         axs[i].legend()
# plt.show()