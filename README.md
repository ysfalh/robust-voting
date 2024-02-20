# Robust Sparse Voting

This repository is the official code for paper Robust Sparse Voting, published at AISTATS 2024.

#### Abstract 
Many applications, such as content moderation and recommendation, require reviewing and scoring a large number of alternatives. Doing so robustly is however very challenging. Indeed, voters' inputs are inevitably sparse: most alternatives are only scored by a small fraction of voters. This sparsity amplifies the effects of biased voters introducing unfairness, and of malicious voters seeking to hack the voting process by reporting dishonest scores.\
We give a precise definition of the problem of robust sparse voting, highlight its underlying technical challenges, and present a novel voting mechanism addressing the problem. We prove that, using this mechanism, no voter can have more than a small parameterizable effect on each alternative's score; a property we call Lipschitz resilience. We also identify conditions of voters comparability under which any unanimous preferences can be recovered, even when each voter provides sparse scores, on a scale that is potentially very different from any other voter's score scale. Proving these properties required us to introduce, analyze and carefully compose novel aggregation primitives which could be of independent interest.

### Code

Use the following command line to run the experiments:

```bash
python3 main.py
```

The parameters of the experiments can be chosen inside the file `main.py`. 
Otherwise, the default parameters will be used.


### Specs

Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz \\
RAM : 8.00 Go 


### Library Requirements

math \\
os \\
json \\
copy \\
shutil \\
tqdm \\
multiprocess \\
numpy \\
itertools \\
pandas \\
matplotlib \\
scipy \\

### Citation

If you use this code, please cite the following (BibTex format):

```bash
@inproceedings{allouah2024robust,
  title        = {Robust Sparse Voting},
  author       = {Allouah, Youssef and Guerraoui, Rachid and Hoang, L{\^e}-Nguy{\^e}n and Villemaud, Oscar},
  booktitle    = {27th International Conference on Artificial Intelligence and Statistics},
  year         = {2024},
  organization = {PMLR}
}
```
