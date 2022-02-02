import json
import os
from scipy.stats import pearsonr
import shutil
from time import time

from data_generation.data import generate_data
from data_generation.voting_rights import generate_voting_rights, regularize_voting_rights
from votes.mehestan import Mehestan
from votes.basic_vote import BasicVote
from votes.maj_judgement import MajJudement
from plots.boxplot import draw_curves, range_boxplot
from numpy.random import default_rng
from tqdm import tqdm
import json


def comparative_runs(
        n_attempts=1, n_voters=30, n_extreme=0, n_alternatives=200,
        density=.01, noise=0, p_byzantine=.45, byz_density=1., byz_strat='random', voting_resilience=1.,
        transformation_name="min-max", regularize=True, pair_perc=1., sm3=0, sm4=0, n_proc=1, **kwargs
):
    """ comparing the voting algorithms on generated data """
    mj_corr, mj_p, bv_corr, bv_p, bv_noreg_corr, bv_noreg_p, mh_corr, mh_p = [], [], [], [], [], [], [], []
    for i in range(n_attempts):

        # data generation
        ratings, original_preferences, mask = generate_data(
            n_voters, n_extreme, n_alternatives, noise=noise,
            density=density, byz_density=byz_density, byz_strat=byz_strat,
            pair_perc=pair_perc, **kwargs
        )
        voting_rights = generate_voting_rights(n_voters, p_byzantine, **kwargs)
        voting_rights, mask = regularize_voting_rights(
            original_preferences, voting_rights, mask,
            voting_resilience=voting_resilience, sm3=sm3, sm4=sm4,
            n_extreme=n_extreme, **kwargs
        )

        # voting with MajJudgement
        mj = MajJudement(ratings, mask, voting_rights)
        out = mj.run()
        corr, pval = pearsonr(out, original_preferences)
        mj_corr.append(corr)
        mj_p.append(pval)

        # voting with BasicVote
        bv = BasicVote(
            ratings, mask, voting_rights,
            voting_resilience, transformation_name=transformation_name, n_proc=n_proc
        )
        out, out_noreg = bv.run()
        corr, pval = pearsonr(out, original_preferences)
        bv_corr.append(corr)
        bv_p.append(pval)

        # without regularisation
        corr, pval = pearsonr(out_noreg, original_preferences)
        bv_noreg_corr.append(corr)
        bv_noreg_p.append(pval)

        # voting with Mehestan
        mh = Mehestan(ratings, mask, voting_rights, voting_resilience, transformation_name=transformation_name, n_proc=n_proc)
        out = mh.run()
        corr, pval = pearsonr(out, original_preferences)
        mh_corr.append(corr)
        mh_p.append(pval)

    return mj_corr, mj_p, bv_corr, bv_p, bv_noreg_corr, bv_noreg_p, mh_corr, mh_p


def auto_run(defaults={}, seed=1, name='', params=[]):
    """ multiple runs of both algorithms with 1 parameter changing """
    l_mj_corr, l_mj_p, l_bv_corr, l_bv_p, l_bv_noreg_corr, l_bv_noreg_p, l_mh_corr, l_mh_p = [], [], [], [], [], [], [], []
    for param in tqdm(params):
        print(name, ':', param)
        defaults.pop(name, None)  # remove parameter default value
        rng = default_rng(seed)
        mj_corr, mj_p, bv_corr, bv_p, bv_noreg_corr, bv_noreg_p, mh_corr, mh_p = comparative_runs(
            **{name: param}, **defaults, rng=rng
        )
        l_mj_corr.append(mj_corr)
        l_mj_p.append(mj_p)
        l_bv_corr.append(bv_corr)
        l_bv_p.append(bv_p)
        l_bv_noreg_corr.append(bv_noreg_corr)
        l_bv_noreg_p.append(bv_noreg_p)
        l_mh_corr.append(mh_corr)
        l_mh_p.append(mh_p)
    return l_mj_corr, l_mj_p, l_bv_corr, l_bv_p, l_bv_noreg_corr, l_bv_noreg_p, l_mh_corr, l_mh_p


def write_params(params, path='params.json'):
    with open(path, 'w') as f:
        json.dump(params, f)


def run_plot(defaults={}, seed=1, folder='exp1', name='', params=[]):
    write_params(defaults, path=f'results/{folder}/params.json')
    l_mj_corr, _, l_bv_corr, _, l_bv_noreg_corr, _, l_mh_corr, _ = auto_run(
        seed=seed, defaults=defaults, name=name, params=params
    )
    draw_curves(
        l_mj_corr, l_bv_noreg_corr, l_bv_corr, l_mh_corr, params, 
        labels=('MajJudgement', 'BasicVote', 'BasicVote+QrMed', 'Mehestan'),
        folder=folder, x_name=name
    )
    range_boxplot(l_mj_corr, params, folder=folder, title='MajJudgement', x_name=name)
    range_boxplot(l_bv_corr, params, folder=folder, title='BasicVote+QrMed', x_name=name)
    range_boxplot(l_bv_noreg_corr, params, folder=folder, title='BasicVote', x_name=name)
    range_boxplot(l_mh_corr, params, folder=folder, title='Mehestan', x_name=name)
    print("++DONE++")


def multiple_experiments(experiments, defaults={}, seed=1):
    """ runs several experiments and saves results """
    shutil.rmtree('results', ignore_errors=True)  # clear results folder
    os.mkdir('results')
    i = 0
    for name, params in experiments.items():
        t_0 = time()
        i += 1
        print('experiment :', i)
        folder = f'exp{i}'
        os.mkdir(f'results/{folder}')
        run_plot(defaults=defaults, seed=seed, folder=folder, name=name, params=params)
        print('Experiment time :', time() - t_0)
