
## NN ensemble relaxer

This repo contains the main code for active learning based on NN ensemble to accelerate the geometry optimization for chemical structures. Details to use the code are shown in [demonstration.ipynb](https://github.com/yilinyang1/NN_ensemble_relaxer/blob/master/demonstration.ipynb). 

Besides the package to use the NN ensemble relaxer, the datasets for the examples like AuPd bare slab, CO/AuPd Icosahedron, Acrolein/AgPd slab relaxations are included as well. It also illustrate the code to conduct active learning based climbing NEB for the Pt heptamer rearrangement example using EMT.

The overall structure for this repo is:

.
├── ...
├── test                    # Test files (alternatively `spec` or `tests`)
│   ├── benchmarks          # Load and stress tests
│   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
│   └── unit                # Unit tests
└── ...
