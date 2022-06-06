import numpy as np


class Environment:
    def __init__(self, products, budgets, features, function, alphas, sigma=0.0):
        self.n_products = len(products)
        self.sigma = sigma
        self.alphas = alphas
        self.features = features
        self.products = products
        self.campaigns = [Campaign(budgets[i], products[i], features, function[products[i]], alphas[i], sigma) for i in
                          range(len(self.products))]
        self.budgets = budgets

    # def add_campaign(self, product, function):
    #     self.campaign.append(
    #         Campaign(self.budget, product, self.features, functions)
    #     )

    def round(self, campaign_id, pulled_arm, feature=None):
        return self.campaigns[campaign_id].round(pulled_arm, feature)

    def round_all(self, feature=None):
        table = []
        for campaign in self.campaigns:
            table.append(campaign.round_all(feature))
        return table


class Campaign:
    def __init__(self, budget, product, features, function, alphas, sigma):
        self.product = product
        self.budget = budget
        self.features = features
        self.alphas = alphas
        self.subcampaigns = [Subcampaign(budget, features[i], i, function, sigma) for i in range(len(self.features))]

    # round a specific arm
    def round(self, pulled_arm, feature=None):
        # aggregate sample
        if feature is None:
            return sum(self.subcampaigns[i].round(pulled_arm) for i in range(len(self.features)))
        # disaggregate sample
        else:
            return self.subcampaigns[feature].round(pulled_arm)

    # round all arms
    def round_all(self, feature=None):
        return [self.round(pulled_arm, feature) for pulled_arm in range(len(self.budget))]


class Subcampaign:
    def __init__(self, budget, feature, feature_id, function, sigma):
        self.feature = feature
        self.means = function[feature_id](budget)
        self.sigmas = np.ones(len(budget)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
