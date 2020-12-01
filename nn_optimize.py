from ase.data import atomic_numbers
import os
import shutil
from ase.db import connect
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from utils.ensemble_trainer import Ensemble_Trainer
from utils.fp_calculator import set_sym
from utils.relax_helper import Relaxer_Helper
import numpy as np


class Ensemble_Relaxer():
    def __init__(self, db, calculator, jobname, ensemble_size=10):
        self.job_name = jobname
        self.job_path = f'./{self.job_name}'
        self.model_path = os.path.join(self.job_path, 'models')
        self.traj_path = os.path.join(self.job_path, 'relax_trajs')
        self.__init_work_place(db)
        self.log_file = open(os.path.join(self.job_path, 'relax_log.txt'), 'w')

        self.elements = self.__get_elements(db)
        self.train_size = 50 if db.count() <= 10 else 5 * db.count()
        self.constraints = [entry.toatoms().constraints for entry in db.select()]   
        self.relaxed = [False] * db.count()

        self.gt_calculator = calculator  # groud truth calculator
        self.n_step = 0
        self.ensemble_size = ensemble_size
        self.fmax = 0.05  # initial set
        self.fp_params = self.__fp_setter()
        

    def __get_elements(self, db):
        """
        get all elements in order for the input database.
        """
        elements = set()
        for entry in db.select():
            elements.update(set(entry.symbols))
        el_num_pairs = [(el, atomic_numbers[el]) for el in elements]
        el_num_pairs.sort(key = lambda x: x[1])
        elements = [x[0] for x in el_num_pairs]

        return elements


    def __init_work_place(self, init_db):
        """
        Prepare local folder to store model, trajectories.
        """
        # prepare local folders
        if not os.path.isdir(self.job_path):
            os.mkdir(self.job_path)
            os.mkdir(self.traj_path)
            os.mkdir(self.model_path)  # if trained on arjuna, this folder is redundent
            init_db_path = os.path.join(self.traj_path, 'initial.db')
            init_db_cp = connect(init_db_path)
            for entry in init_db.select():
                init_db_cp.write(entry)

            new_db = connect(os.path.join(self.traj_path, 'to-cal-step0.db'))
            # set constraints to None for NN training
            for entry in init_db.select():
                atoms = entry.toatoms()
                atoms.set_constraint(None)
                new_db.write(atoms)
        
        return None


    def __fp_setter(self):
        Gs = [2, 4]
        g2_etas = [0.05, 4.0, 20.0] 
        g2_Rses = [0.0] 
        g4_etas = [0.005] 
        g4_zetas = [1.0, 4.0] 
        g4_lambdas = [-1.0, 1.0]
        cutoff = 6.0
        params_set = set_sym(self.elements, Gs, cutoff, g2_etas=g2_etas, g2_Rses=g2_Rses, 
                            g4_etas=g4_etas, g4_zetas=g4_zetas, g4_lambdas=g4_lambdas)
        return params_set
    

    def __get_ground_truth(self):
        """
        Calculate the groud truth energetics for uncertain configurations.
        """
        print(f'Step {self.n_step}: get groud truth data')
        self.log_file.write(f'Step {self.n_step}: get groud truth data \n')
        to_cal_db_path = os.path.join(self.traj_path, f'to-cal-step{self.n_step}.db')
        caled_db_path = os.path.join(self.traj_path, f'to-cal-step{self.n_step}-gt.db')
        to_cal_db = connect(to_cal_db_path)
        caled_db = connect(caled_db_path)
        for entry in to_cal_db.select():
            atoms = entry.toatoms()
            atoms.set_calculator(self.gt_calculator)
            nrg = atoms.get_potential_energy()
            frs = atoms.get_forces()
            atoms.set_calculator(SPC(atoms, energy=nrg, forces=frs))
            caled_db.write(atoms)

        print(f'Step {self.n_step}: groud truth data calculation done')
        self.log_file.write(f'Step {self.n_step}: groud truth data calculation done \n')
        return None


    def __update_training_data(self):
        """
        Add calcualted ground truth data into training file
        Check the convergence of atoms configurations
        """
        gt_path = os.path.join(self.traj_path, f'to-cal-step{self.n_step}-gt.db')
        gt_atoms = [entry.toatoms() for entry in connect(gt_path).select()]
        to_relax_path = os.path.join(self.traj_path, f'to-relax-step{self.n_step}.db')
        to_relax_db = connect(to_relax_path)
        max_frs = []
        to_added_atoms = []
        if self.n_step == 0:
            previous_data = []
        else:
            train_db_path = os.path.join(self.traj_path, f'train-set-step{self.n_step-1}.db')
            previous_db = connect(train_db_path)
            previous_data = [entry.toatoms() for entry in previous_db.select()]
        
        for i, atoms in enumerate(gt_atoms):
            if not self.relaxed[i]:
                to_added_atoms.append(atoms)  # for training
            dummy = atoms.copy()  # for relaxation
            dummy.set_constraint(self.constraints[i])
            dummy.set_calculator(SPC(dummy, energy=atoms.get_potential_energy(), forces=atoms.get_forces()))
            tmp_frs = atoms.get_forces()
            for c in self.constraints[i]:
                c.adjust_forces(None, tmp_frs)
            max_fr = np.linalg.norm(tmp_frs, axis=1).max()
            if max_fr <= self.fmax:
                self.relaxed[i] = True
            else:
                to_relax_db.write(dummy)
            max_frs.append(round(max_fr, 3))
        
        all_data = previous_data + to_added_atoms
        new_train_path = os.path.join(self.traj_path, f'train-set-step{self.n_step}.db')
        new_train_db = connect(new_train_path)
        for atoms in all_data[-self.train_size:]:
            new_train_db.write(atoms)
        
        self.constraints = [self.constraints[i] for i in range(len(self.relaxed)) if not self.relaxed[i]]
        self.relaxed = [entry for entry in self.relaxed if not entry]
        print('max force for each configuration: ')
        print(max_frs)
        self.log_file.write('max force for each configuration: \n')
        self.log_file.write(f'{max_frs} \n')
        self.log_file.flush()
        if not self.relaxed:
            return True
        else:
            return False


    def __train_NN(self):
        """
        Retrain NN ensemble using updated training data
        """
        print(f'Step {self.n_step}: start training')
        self.log_file.write(f'Step {self.n_step}: start training \n')
        train_db_path = os.path.join(self.traj_path, f'train-set-step{self.n_step}.db')
        step_model_path = os.path.join(self.model_path, f'model-step{self.n_step}')
        trainer = Ensemble_Trainer(train_db_path, step_model_path, self.fp_params, self.ensemble_size)
        trainer.calculate_fp()
        trainer.train_ensemble()
        print(f'Step {self.n_step}: training done')
        self.log_file.write(f'Step {self.n_step}: training done\n')
        return
    

    def __relax_NN(self):
        """
        Relax configurations using NN
        """
        print(f'Step {self.n_step}: start NN relaxation')
        self.log_file.write(f'Step {self.n_step}: start NN relaxation \n')
        to_relax_path = os.path.join(self.traj_path, f'to-relax-step{self.n_step}.db')
        train_db_path = os.path.join(self.traj_path, f'train-set-step{self.n_step}.db')
        step_model_path = os.path.join(self.model_path, f'model-step{self.n_step}')
        to_cal_db_path = os.path.join(self.traj_path, f'to-cal-step{self.n_step+1}.db')

        relaxer = Relaxer_Helper(to_relax_path, train_db_path, step_model_path, self.fp_params, 
                                self.ensemble_size, self.elements)
        relaxer.relax(self.n_step, self.fmax, to_cal_db_path)

        print(f'Step {self.n_step}: NN relaxation done')
        self.log_file.write(f'Step {self.n_step}: NN relaxation done \n')
        return

    
    def __collect_configs(self):
        """
        Collect final configurations after relaxation
        """
        init_db = connect(os.path.join(self.traj_path, 'initial.db'))
        init_cnstrt = [entry.toatoms().constraints for entry in init_db.select()]
        config_steps = [[] for _ in range(init_db.count())]
        config_convgs = [False] * init_db.count()

        for i in range(self.n_step):
            tmp_db = connect(os.path.join(self.traj_path, f'to-cal-step{i}-gt.db'))
            tmp_configs = [entry.toatoms() for entry in tmp_db.select()]
            tmp_frs = [atoms.get_forces() for atoms in tmp_configs]
            tmp_nrgs = [atoms.get_potential_energy() for atoms in tmp_configs]
            j = 0
            for k, convg in enumerate(config_convgs):
                if not convg:
                    f = tmp_frs[j]
                    for c in init_cnstrt[k]:
                        c.adjust_forces(None, f)
                    mf = np.linalg.norm(f, axis=1).max()
                    if mf <= self.fmax:
                        config_convgs[k] = True
                    dummy = tmp_configs[j].copy()
                    dummy.set_constraint(init_cnstrt[k])
                    dummy.set_calculator(SPC(dummy, energy=tmp_nrgs[j], forces=tmp_frs[j]))
                    config_steps[k].append(dummy)
                    j += 1
        
        final_db = connect(os.path.join(self.traj_path, 'final.db'))
        for traj in config_steps:
            final_db.write(traj[-1])
        return final_db


    def run(self, fmax=0.05, steps=None):
        self.fmax = fmax
        while steps and self.n_step <= steps:
            self.__get_ground_truth()
            if self.__update_training_data():
                self.n_step += 1
                break
            self.__train_NN()
            self.__relax_NN()
            self.n_step += 1

        return self.__collect_configs()
