""" Votes class """
from utils.losses import MrDist


class Vote:
    """ data generation and mask """

    def __init__(self, ratings, mask, voting_rights):
        self.ratings = ratings
        self.mask = mask
        self.voting_rights = voting_rights
        self.loss = MrDist()
        self.n_alternatives = len(ratings[0])
        self.n_voters = len(ratings)

    def run(self):
        return
