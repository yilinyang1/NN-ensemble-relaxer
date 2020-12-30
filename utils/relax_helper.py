from .train_agent import BPNN
from .fp_calculator import set_sym
from .nn_calculator import NN_Calc_Ensemble
from ase.calculators.emt import EMT
import numpy as np
import sys
import os
from ase.db import connect
from .relax_optmizer import BFGS_Ensemble
from ase.io.trajectory import Trajectory
import torch



class Relaxer_Helper():
    def __init__(self, relax_db_path, train_db_path, step_model_path, fp_params, ensemble_size, elements, nn_params, alpha):
        layer_nodes = nn_params['layer_nodes']
        activations = nn_params['activations']
        n_fp = len(fp_params[elements[0]]['i'])
        model_list = []
        for m in range(ensemble_size):
            model = BPNN(n_fp, layer_nodes, activations, len(elements))
            model.load_state_dict(torch.load(os.path.join(step_model_path, f'model-{m}.sav'), 
                                    map_location=torch.device('cpu')))
            model_list.append(model)
        scale = torch.load(os.path.join(step_model_path, 'train_set_scale.sav'))
        self.nn_calc = NN_Calc_Ensemble(params_set=fp_params, models_list=model_list, 
                                    scale=scale, elements=elements)
        self.alpha = alpha
        self.threshold = self.__get_std_threshold(train_db_path)
        self.step_model_path = step_model_path
        self.relax_db_path = relax_db_path
        

    def __get_std_threshold(self, train_db_path):
        train_db = connect(train_db_path)
        train_nrg_stds = []
        for entry in train_db.select():
            atoms = entry.toatoms()
            atoms.set_calculator(self.nn_calc)
            atoms.get_potential_energy()
            train_nrg_stds.append(atoms.calc.results['energy_std'])
        return self.alpha * np.max(train_nrg_stds)

    
    def relax(self, n_step, fmax, to_cal_db_path):
        relax_dir_path = os.path.join(self.step_model_path, 'nn-relax-trajs')
        if not os.path.isdir(relax_dir_path):
            os.mkdir(relax_dir_path)
        
        to_relax_db = connect(self.relax_db_path)
        to_relax_images = [entry.toatoms() for entry in to_relax_db.select()]
        for c, atoms in enumerate(to_relax_images):
            relax_log_file = os.path.join(relax_dir_path, f'atoms-{c}-log.txt')
            relax_traj_file = os.path.join(relax_dir_path, f'atoms-{c}-traj.traj')
            if os.path.isfile(relax_log_file): 
                os.remove(relax_log_file)
            if os.path.isfile(relax_traj_file): 
                os.remove(relax_traj_file)

            atoms.set_calculator(self.nn_calc)
            if n_step <= 2:
                dyn = BFGS_Ensemble(atoms=atoms, logfile=relax_log_file, trajectory=relax_traj_file, 
                                    maxstep=0.01, nrg_std=self.threshold)
                dyn.run(fmax=fmax-0.02, steps=50)
            else:
                atoms.set_calculator(self.nn_calc)
                dyn = BFGS_Ensemble(atoms, logfile=relax_log_file, trajectory=relax_traj_file,
                                    maxstep=0.04, nrg_std=self.threshold)
                dyn.run(fmax=fmax-0.02, steps=50)

        
        to_cal_db = connect(to_cal_db_path)
        for c in range(len(to_relax_images)):
            relax_traj_file = os.path.join(relax_dir_path, f'atoms-{c}-traj.traj')
            traj = Trajectory(relax_traj_file)
            atoms = traj[-1]
            atoms.set_constraint(None)
            to_cal_db.write(atoms)
        
        return None