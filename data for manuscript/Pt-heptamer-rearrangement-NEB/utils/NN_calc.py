from ase.calculators.calculator import (Calculator, all_changes, PropertyNotImplementedError)
from ase.calculators.emt import EMT
from .fp_calculator import set_sym, calculate_fp
import torch
import numpy as np

class NN_Calc(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy']

    def __init__(self, params_set, models_list, scale, elements, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.params_set = params_set
        self.models_list = models_list
        self.scale = scale
        self.elements = elements

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        temp_atoms = self.atoms.copy()
        temp_atoms.set_calculator(EMT())

        # calculate fingerprints and preprocessing
        image_data = calculate_fp(temp_atoms, self.elements, self.params_set)
        n_atoms = len(self.atoms)
        n_features = len(self.params_set[self.elements[0]]['i'])
        atom_idx = image_data['atom_idx']

        image_fp = torch.zeros([n_atoms, n_features])
        image_dfpdX = torch.zeros([n_atoms, n_features, n_atoms, 3])
        image_e_mask = torch.zeros([n_atoms, len(self.elements)])

        for ie in range(1, len(self.elements) + 1):
            el = self.elements[ie-1]
            image_fp[atom_idx == ie, :] = torch.FloatTensor(
                image_data['x'][el])
            image_dfpdX[atom_idx == ie, :, :, :] = torch.FloatTensor(
                image_data['dx'][el])
            image_e_mask[atom_idx == ie, ie-1] = 1

        image_fp = (image_fp - self.scale['fp_min']) / (self.scale['fp_max'] - self.scale['fp_min'] + 1e-10)
        image_dfpdX /= (self.scale['fp_max'] - self.scale['fp_min'] + 1e-10).view(1, -1, 1, 1)

        image_fp.requires_grad = True
        image_dfpdX = image_dfpdX.reshape(n_atoms*n_features, n_atoms*3)

        # calculate energy and forces
        nrg_preds = []
        force_preds = []

        for model in self.models_list:
            image_nrg_pre_raw = model(image_fp)  # [N_atoms, N_elements]
            image_nrg_pre_cluster = torch.sum(image_nrg_pre_raw * image_e_mask)  # [N_atoms]
            image_b_dnrg_dfp = torch.autograd.grad(image_nrg_pre_cluster, image_fp,
                                                   create_graph=True, retain_graph=True)[0].reshape(1, -1)

            image_force_pre = - torch.mm(image_b_dnrg_dfp, image_dfpdX).reshape(n_atoms, 3)

            nrg_preds.append(image_nrg_pre_cluster.detach().item())
            force_preds.append(image_force_pre.detach().numpy())

        self.energy = np.mean(nrg_preds)
        self.uncertainty = np.std(nrg_preds)
        self.forces = np.mean(force_preds, axis=0)
        self.results['energy'] = self.energy
        # use the key "free_energy"  to store uncertainty
        self.results['free_energy'] = self.uncertainty
        self.results['forces'] = self.forces

