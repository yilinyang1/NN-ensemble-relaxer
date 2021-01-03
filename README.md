
## NN ensemble relaxer

This repo contains the main code for active learning based on NN ensemble to accelerate the geometry optimization for chemical structures. Details to use the code are shown in [demonstration.ipynb](https://github.com/yilinyang1/NN_ensemble_relaxer/blob/master/demonstration.ipynb). 

Besides the package to use the NN ensemble relaxer, the datasets for the examples like AuPd bare slab, CO/AuPd Icosahedron, Acrolein/AgPd slab relaxations are included as well. It also illustrate the code to conduct active learning based climbing NEB for the Pt heptamer rearrangement example using EMT.

The overall structure for this repo is:

    .
    |-- AuPd-nano-test          # Folder to contain the dataset in demonstration example
    |-- data for manuscript                    # Documentation files (alternatively `doc`)
    |-- src                     # Source files (alternatively `lib` or `app`)
    |-- test                    # Automated tests (alternatively `spec` or `tests`)
    |-- tools                   # Tools and utilities
    |-- LICENSE
    |
    |-- README.md
