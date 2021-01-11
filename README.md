
## NN ensemble relaxer

This repo contains the main code for active learning based on NN ensemble to accelerate the geometry optimization for chemical structures. Details to use the code are shown in [demonstration.ipynb](https://github.com/yilinyang1/NN_ensemble_relaxer/blob/master/demonstration.ipynb). 

Besides the package to use the NN ensemble relaxer, the datasets for the examples like AuPd bare slab, CO/AuPd Icosahedron, Acrolein/AgPd slab relaxations are included as well. It also illustrate the code to conduct active learning based climbing NEB for the Pt heptamer rearrangement example using EMT.

The overall structure for this repo is:

    .
    ├── AuPd-nano-test           # Folder to store models and relaxation trajs in the demo example
    ├── data for manuscript     
    │   ├── Acetylele-hydrogenation-NEB              # Datasets for acetylenen hydrogenation NEB
    │   ├── Acrolein-AgPd-offline                    # Datasets for Acrolein/AgPd offline relaxation
    │   ├── Acrolein-AgPd-single-multiple-configs    # Datasets for active learning relaxation for Acrolein/AgPd with single, multiple configurations w/o warmup
    |   ├── Au-slabs-single-config                   # Datasets for active learning relaxation on Au slabs with single configuration
    │   ├── Pt-heptamer-rearrangement-NEB            # Demo code and dataset for Pt-heptamer-rearrangement NEB
    │   └── more-geometry-optimization-data          # Datasets for AuPd bare slab, more Acrolein/AgPd and AuPd icosahedron relaxation
    ├── utils                    # Utils files for NN ensemble relaxer, like NN ASE calculator, NN training, active learning relaxation files 
    ├── AuPd-ico-to-relax-10.db  # ASE database file generated in the demo example
    ├── README.md                
    ├── demonstration.ipynb      # Demostration to use the NN ensemble relaxer with an AuPd nanoparticle example
    └── nn_optimize.py           # NN ensemble relaxer class

The code to calculate the symmetry function is modified based on functions from https://github.com/MDIL-SNU/SIMPLE-NN
