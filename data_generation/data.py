import numpy as np

def regularize_mask(mask, pair=(0, 1), rng=None):
    """ add a pair of commonly voted alternatives """
    n, m = mask.shape
    mask[:, pair] = np.ones((n, 2))
    for j in range(m):
        if np.all(mask[:, j] == 0):
            a = rng.integers(0, n)
            mask[a, j] = 1
    return mask

def gen_style_mask(
    n_unif, n_good, n_bad, n_alternatives, 
    n_byz=1, extreme=0.3, density=0.2, byz_density=1, regularize=True, rng=None
        ):
    """ generates a mask with 3 notation styles for honest voters:
        - uniform votes
        - voting for good videos only
        - voting for bad videos only
    """
    n_voters = n_unif + n_good + n_bad + n_byz
    top = int(extreme * n_alternatives)  # number of extreme videos
    mask = np.zeros((n_voters, n_alternatives), dtype=int)  # initialize mask
    # uniform
    mask[:n_unif, :] = rng.binomial(1, density, (n_unif, n_alternatives))
    # good
    mask[n_unif: n_unif + n_good, n_alternatives - top:] = rng.binomial(1, density, (n_good, top))
    # bad
    mask[n_unif + n_good: n_unif + n_good + n_bad, :top] = rng.binomial(1, density, (n_bad, top))
    # byzantine
    mask[n_voters - n_byz:, :] = rng.binomial(1, byz_density, (n_byz, n_alternatives))

    if regularize:
        return regularize_mask(mask)  # add common rated pair
    return mask


def generate_data(n_voters, n_extreme, n_alternatives, density, noise=0, byz_density=1, regularize=True, rng=None):
    """ generates random original preferences, ratings by voters and a mask """
    
    original_preferences = rng.standard_cauchy(n_alternatives)
    original_preferences = (original_preferences - original_preferences.min()) / (
                original_preferences.max() - original_preferences.min())
    original_preferences = np.sort(original_preferences)
    ratings = np.zeros((n_voters, n_alternatives))

    byzantine = n_voters - 1  # TODO parametrize nb byzantine
    for voter in range(n_voters):
        if voter == byzantine:
            ratings[voter] = rng.standard_cauchy(n_alternatives)
        else:
            scaling = rng.lognormal(1., 1.)
            translation = rng.normal(0., 10.)
            scaling = 1e-1 if scaling == 0 else scaling
            ratings[voter] = scaling * original_preferences + translation
            ratings[voter] += np.round(rng.normal(0, noise * scaling, (n_alternatives,)), 2)  # adding noise, TODO create dictionnary with names of transfo
            
    mask = gen_style_mask(
        n_voters - 2 * n_extreme - 1, n_extreme, n_extreme, n_alternatives, 1, 
        density=density, byz_density=byz_density, regularize=regularize, rng=rng
    )

    return ratings, original_preferences, mask