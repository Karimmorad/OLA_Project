import json
import numpy as np
import matplotlib.pyplot as plt


class setting_manager:
    def __init__(self):
        with open('config.json') as json_file:
            data = json.load(json_file)
        campaigns = data["campaigns"]

        # Product settings
        self.product = []
        for i in range(len(campaigns)):
            self.product.append(campaigns[i]["product"])

        # Class settings
        self.features = list(campaigns[0]["subcampaign"].keys())

        # Experiment settings
        self.click_functions = {}
        for campaign in campaigns:
            prod = campaign['product']
            self.click_functions[prod] = []
            for feature in self.features:
                alpha_bar = campaign['subcampaign'][feature]['alpha_bar']
                self.click_functions[prod].append(lambda x, a=alpha_bar: self.n(x, a))

    def n(self, x, a, max_clicks=200):
        return (1 - np.exp(-5.0 * x)) * a * max_clicks


# colors = ['r', 'b', 'black']
# env = setting_manager()
#
# budgets = np.linspace(0, 10, num=11)
# x = np.linspace(0, max(budgets), num=550)
# features = env.features
# products = env.product
#
# fig, axs = plt.subplots(1, 5, figsize=(20, 8))
# for i, product in enumerate(products):
#     for j, label in enumerate(features):
#         y = env.click_functions[product][j](x)
#         scatters = env.click_functions[product][j](budgets)
#         axs[i].plot(x, y, color=colors[j], label=label)
#         axs[i].scatter(budgets, scatters, color=colors[j])
#         axs[i].set_title("product " + product + "click function")
#         axs[i].set_xlabel("Budget")
#         axs[i].set_ylabel("Number of Clicks")
#         axs[i].legend()
# plt.show()