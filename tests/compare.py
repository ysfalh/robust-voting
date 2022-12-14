import json
import os
import copy
from scipy.stats import pearsonr
import shutil
import numpy as np
from tqdm import tqdm

from data_generation.data import generate_all_data, sparsify_mask
from votes.mehestan import Mehestan
from votes.basic_vote import BasicVote
from votes.maj_judgement import MajJudement
from plots.boxplot import draw_curves, range_boxplot
from numpy.random import default_rng



def comparative_runs(
        n_attempts=1, n_voters=30, n_extreme=0, n_alternatives=200,
        density=.01, noise_range=(0, 0), p_byzantine=.45, byz_density=1., byz_strat='random', voting_resilience=1.,
        transformation_name="min-max", delta=None, sm=0, n_proc=1, **kwargs
):
    """ comparing the voting algorithms on generated data """
    mj_corr, mj_p, bv_corr, bv_p, bv_noreg_corr, bv_noreg_p, mh_corr, mh_p, mh_noreg_corr, mh_noreg_p = [], [], [], [], [], [], [], [], [], []

    seeds = range(n_attempts)

    for seed in seeds:

        ratings, mask, voting_rights, original_preferences, deltas = generate_all_data(
            n_voters, n_extreme, n_alternatives, noise_range,
            density, byz_density, byz_strat, sm, delta, p_byzantine,
            seed, **kwargs
        )

        # voting with MajJudgement
        mj = MajJudement(
            np.copy(ratings), mask, voting_rights
        )
        out = mj.run()
        corr, pval = pearsonr(out, original_preferences)
        mj_corr.append(corr)
        mj_p.append(pval)


        # voting with BasicVote
        bv = BasicVote(
            np.copy(ratings), mask, voting_rights,
            voting_resilience, transformation_name=transformation_name, n_proc=n_proc, deltas=deltas
        )
        out, out_noreg = bv.run()

        # with regularization
        corr, pval = pearsonr(out, original_preferences)
        bv_corr.append(corr)
        bv_p.append(pval)

        # without regularisation
        corr, pval = pearsonr(out_noreg, original_preferences)
        bv_noreg_corr.append(corr)
        bv_noreg_p.append(pval)

        # voting with Mehestan
        # with regularization
        mh = Mehestan(
            np.copy(ratings), mask, voting_rights, voting_resilience,
            transformation_name=transformation_name, n_proc=n_proc, deltas=deltas
        )
        out = mh.run()
        corr, pval = pearsonr(out, original_preferences)
        mh_corr.append(corr)
        mh_p.append(pval)
        # print("mh", out[0], corr, pval)

        # without regularization
        mh_noreg = Mehestan(
            np.copy(ratings), mask, voting_rights, 0,
            transformation_name=transformation_name, n_proc=n_proc, deltas=deltas
        )
        out = mh_noreg.run()
        corr, pval = pearsonr(out, original_preferences)
        mh_noreg_corr.append(corr)
        mh_noreg_p.append(pval)
        # print("mhnoreg", out[0], corr, pval)

    return mj_corr, mj_p, bv_noreg_corr, bv_noreg_p, mh_corr, mh_p, mh_noreg_corr, mh_noreg_p


# def comparative_runs_sparsified(
#         ratings=None, mask=None, voting_rights=None,
#         density=0.8, n_extreme=0,
#         n_attempts=1,  delta=1e-10, voting_resilience=1., transformation_name="min-max", n_proc=1, **kwargs
#     ):
#     """ comparing the voting algorithms using existing data and its sparsified version """
#     mj_corr, mj_p, bv_corr, bv_p, bv_noreg_corr, bv_noreg_p, mh_corr, mh_p = [], [], [], [], [], [], [], []

#     # Computing -------- "ground truths" -----------------

#     origin_mask = mask

#     # voting with MajJudgement
#     mj = MajJudement(ratings, origin_mask, voting_rights)
#     gt_mj = mj.run()

#     # voting with BasicVote
#     bv = BasicVote(
#         np.copy(ratings), origin_mask, voting_rights,
#         voting_resilience, transformation_name=transformation_name, n_proc=n_proc, deltas=delta
#     )

#     gt_bv, gt_noreg = bv.run()

#     # voting with Mehestan
#     mh = Mehestan(
#         np.copy(ratings), origin_mask, voting_rights, voting_resilience,
#         transformation_name=transformation_name, n_proc=n_proc, deltas=delta
#     )
#     gt_mh = mh.run()

#     # --------- comparing to sparsified versions ----------------
#     seeds = range(n_attempts)

#     mj_corr, mj_p, bv_corr, bv_p, bv_noreg_corr, bv_noreg_p, mh_corr, mh_p = [], [], [], [], [], [], [], []

#     for seed in seeds:

#         rng = default_rng(seed)
#         np.random.seed(seed)
#         mask = sparsify_mask(origin_mask, density, n_extreme=n_extreme, rng=rng)

#            # voting with MajJudgement
#         mj = MajJudement(np.copy(ratings), mask, voting_rights)
#         out = mj.run()
#         corr, pval = pearsonr(out, gt_mj)
#         mj_corr.append(corr)
#         mj_p.append(pval)

#         # voting with BasicVote
#         bv = BasicVote(
#             np.copy(ratings), mask, voting_rights,
#             voting_resilience, transformation_name=transformation_name, n_proc=n_proc, deltas=delta
#         )
#         out, out_noreg = bv.run()
#         corr, pval = pearsonr(out, gt_bv)
#         bv_corr.append(corr)
#         bv_p.append(pval)

#         # without regularisation
#         corr, pval = pearsonr(out_noreg, gt_noreg)
#         bv_noreg_corr.append(corr)
#         bv_noreg_p.append(pval)

#         # voting with Mehestan
#         mh = Mehestan(
#             np.copy(ratings), mask, voting_rights, voting_resilience,
#             transformation_name=transformation_name, n_proc=n_proc, deltas=delta
#         )
#         out = mh.run()
#         corr, pval = pearsonr(out, gt_mh)
#         mh_corr.append(corr)
#         mh_p.append(pval)

#     return mj_corr, mj_p, bv_corr, bv_p, bv_noreg_corr, bv_noreg_p, mh_corr, mh_p


def auto_run(defaults={}, name='', params=[], data=None):
    """ multiple runs of all algorithms with 1 parameter changing """
    l_mj_corr, l_mj_p, l_bv_noreg_corr, l_bv_noreg_p, l_mh_corr, l_mh_p, l_mh_noreg_corr, l_mh_noreg_p = [], [], [], [], [], [], [], []
    for param in tqdm(params):
        defaults.pop(name, None)  # remove parameter default value
        if data is None:
            mj_corr, mj_p, bv_noreg_corr, bv_noreg_p, mh_corr, mh_p, mh_noreg_corr, mh_noreg_p = comparative_runs(
            **{name: param}, **defaults
        )
        else :
            pass  # DEPRECATED
            # mj_corr, mj_p, bv_corr, bv_p, bv_noreg_corr, bv_noreg_p, mh_corr, mh_p = comparative_runs_sparsified(
            # **{name: param}, **defaults, **data
        # )
        l_mj_corr.append(mj_corr)
        l_mj_p.append(mj_p)
        l_bv_noreg_corr.append(bv_noreg_corr)
        l_bv_noreg_p.append(bv_noreg_p)
        l_mh_corr.append(mh_corr)
        l_mh_p.append(mh_p)
        l_mh_noreg_corr.append(mh_noreg_corr)
        l_mh_noreg_p.append(mh_noreg_p)
    return l_mj_corr, l_mj_p, l_bv_noreg_corr, l_bv_noreg_p, l_mh_corr, l_mh_p, l_mh_noreg_corr, l_mh_noreg_p


def write_params(params, path='params.json'):
    with open(path, 'w') as f:
        json.dump(params, f)


def run_plot(defaults={}, folder='exp1', name='', params=[], data=None):
    write_params(defaults, path=f'results/{folder}/params.json')
    l_mj_corr, _, l_bv_noreg_corr, _, l_mh_corr, _, l_mh_noreg_corr, _,  = auto_run(
        defaults=defaults, name=name, params=params, data=data
    )
    draw_curves(
        l_mj_corr, l_bv_noreg_corr, l_mh_corr, l_mh_noreg_corr, params,
        labels=('Median', 'MinMax + Median', f'Mehestan W={defaults["voting_resilience"]}','Mehestan W=0'),
        folder=folder, x_name=name
    )
    range_boxplot(l_mj_corr, params, folder=folder, title='Median', x_name=name)
    range_boxplot(l_mh_noreg_corr, params, folder=folder, title='Mehestan W=0', x_name=name)
    range_boxplot(l_bv_noreg_corr, params, folder=folder, title='MinMax + Median', x_name=name)
    range_boxplot(l_mh_corr, params, folder=folder, title=f'Mehestan W={defaults["voting_resilience"]}', x_name=name)
    print("++DONE++")


def multiple_experiments(experiments, data=None):
    """ runs several experiments and saves results """
    shutil.rmtree('results', ignore_errors=True)  # clear results folder
    os.mkdir('results')
    i = 0
    for default, exp in experiments:
        i += 1
        folder = f'{i}-'+exp['name']
        os.mkdir(f'results/{folder}')
        run_plot(defaults=copy.deepcopy(default), folder=folder, **exp, data=data)
