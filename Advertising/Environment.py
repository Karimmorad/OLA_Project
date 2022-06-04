import numpy as np


class Environment:
    def __init__(self, products, budgets, features, function, alphas, sigma=0.0):
        self.n_products = len(products)
        self.sigma = sigma
        self.alphas = alphas
        self.features = features
        self.products = products
        self.campaign = [Campaign(budgets[i], products[i], features, function, alphas[i], sigma) for i in
                         range(len(self.products))]
        self.budgets = budgets

    # def add_campaign(self, product, function):
    #     self.campaign.append(
    #         Campaign(self.budget, product, self.features, functions)
    #     )

    def round(self, campaign_id, pulled_arm, feature=None):
        return self.campaign[campaign_id].round(pulled_arm, feature)

    def round_all(self, feature=None):
        table = []
        for campaign in self.campaigns:
            table.append(campaign.round_all(feature))
        return table


class Campaign:
    def __init__(self, budget, product, features, function, alphas, sigma):
        self.subcampaigns = [Subcampaign(budget, features[i], function, sigma) for i in range(len(self.features))]
        self.product = product
        self.budget = budget
        self.features = features
        self.alphas = alphas

    # def add_subcampaign(self, function, feature):
    #     self.subcampaigns.append(
    #         Subcampaign(feature, self.budget, feature, function)
    #     )

    # # round a specific arm
    # def round(self, subcampaign_id, pulled_arm, feature=None):
    #     return self.subcampaigns[subcampaign_id].round(pulled_arm, feature)
    #
    # # round all arms
    # def round_all(self, feature=None):
    #     table = []
    #     for subcampaign in self.subcampaigns:
    #         table.append(subcampaign.round_all(feature))
    #     return table

    # round a specific arm
    def round(self, pulled_arm, feature=None):
        # aggregate sample
        if feature is None:
            return sum(self.alphas[i] * self.subcampaigns[i].round(pulled_arm) for i in range(len(self.features)))
        # disaggregate sample
        else:
            return self.subcampaigns[feature].round(pulled_arm)

    # round all arms
    def round_all(self, feature=None):
        return [self.round(pulled_arm, feature) for pulled_arm in range(len(self.budget))]


class Subcampaign:
    # def __init__(self, budget, product, feature, function):
    #     self.product = product
    #     self.feature = feature
    #     self.budgets = budget
    #     self.means = function(budget)
    # self.features = [Subcampaign_feature(budgets, features[i], sigma) for i in range(self.n_features)]

    # round a specific arm
    # def round(self, pulled_arm, feature=None):
    #     # aggregate sample
    #     if feature is None:
    #         return sum(self.weights[i] * self.features[i].round(pulled_arm) for i in range(self.n_features))
    #     # disaggregate sample
    #     else:
    #         return self.features[feature].round(pulled_arm)
    #
    # # round all arms
    # def round_all(self, phase=None):
    #     return [self.round(pulled_arm, phase) for pulled_arm in range(len(self.budgets))]
    def __init__(self, budgets, feature, function, sigma):
        self.feature = feature
        self.means = function(budgets)
        self.sigmas = np.ones(len(budgets)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
