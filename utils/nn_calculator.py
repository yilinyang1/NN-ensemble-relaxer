from ase.calculators.calculator import (Calculator, all_changes, PropertyNotImplementedError)
from .fp_calculator import set_sym, calculate_fp, db_to_fp
from ase.calculators.singlepoint import SinglePointCalculator as SPC
import torch
import numpy as np
from copy import deepcopy


def calculate_atoms(atoms, model, scale, params_set, elements, is_force=True):
    image_data = calculate_fp(atoms, elements, params_set)
    n_atoms, n_features = len(atoms), len(params_set[elements[0]]['i'])
    atom_idx = image_data['atom_idx']
    
    image_fp = torch.zeros([n_atoms, n_features])
    image_dfpdX = torch.zeros([n_atoms, n_features, n_atoms, 3])
    image_e_mask = torch.zeros([n_atoms, len(elements)])

    for ie in range(1, len(elements) + 1):
        el = elements[ie-1]
        image_fp[atom_idx == ie, :] = torch.FloatTensor(image_data['x'][el])
        image_dfpdX[atom_idx == ie, :, :, :] = torch.FloatTensor(image_data['dx'][el])
        image_e_mask[atom_idx == ie, ie-1] = 1

    image_fp = (image_fp - scale['fp_min']) / (scale['fp_max'] -  scale['fp_min'] + 1e-10)
    image_dfpdX /= (scale['fp_max'] - scale['fp_min'] + 1e-10).view(1, -1, 1, 1)
    
    image_fp.requires_grad = True    
    image_dfpdX = image_dfpdX.reshape(n_atoms*n_features, n_atoms*3)

    image_nrg_pre_raw = model(image_fp)
    image_nrg_pre_cluster = torch.sum(image_nrg_pre_raw * image_e_mask)
    if is_force:
        image_b_dnrg_dfp = torch.autograd.grad(image_nrg_pre_cluster, image_fp, 
                                                create_graph=True, retain_graph=True)[0].reshape(1, -1)
        image_force_pre = - torch.mm(image_b_dnrg_dfp, image_dfpdX).reshape(n_atoms, 3)
        nrg_pred = image_nrg_pre_cluster.detach().item()
        frs_pred = image_force_pre.detach().numpy()
        return (nrg_pred, frs_pred)
    else:
        nrg_pred = image_nrg_pre_cluster.detach().item()
        return (nrg_pred, None)


class NN_Calc_Ensemble(Calculator):
    implemented_properties = ['energy', 'forces', 'energy_std', 'forces_std', 'nn_rep']

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
        temp_atoms.set_calculator(SPC(temp_atoms, energy=0.0, forces=np.zeros([len(temp_atoms), 3])))
        
        # calculate energy and forces
        nrg_preds = []
        force_preds = []
        
        for model in self.models_list:
            nrg_pred, frs_pred = calculate_atoms(temp_atoms, model, self.scale, self.params_set, self.elements)
            nrg_preds.append(nrg_pred)
            force_preds.append(frs_pred)
        
        self.energy = np.mean(nrg_preds)
        self.uncertainty = np.std(nrg_preds)
        self.forces = np.mean(force_preds, axis=0)
        self.results['energy'] = self.energy
        self.results['energy_std'] = self.uncertainty  # use the key "free_energy"  to store uncertainty
        self.results['forces'] = self.forces
        self.results['forces_std'] = np.std(force_preds, axis=0)  # N_atom * 3


class NN_Calc_Lat_Dist(Calculator):
    implemented_properties = ['energy', 'forces', 'latent_dist', 'max_latent_dist', 'latent_rep', 'latent_rep_ww', 'latent_dist_all']

    def __init__(self, params_set, model, scale, elements, train_nbrs, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.params_set = params_set
        self.model = model
        self.scale = scale
        self.elements = elements
        self.train_nbrs = train_nbrs  # list of nearest neighbors of training set in 3 dims

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        temp_atoms = self.atoms.copy()
        temp_atoms.set_calculator(SPC(temp_atoms, energy=0.0, forces=np.zeros([len(temp_atoms), 3])))
        
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
            image_fp[atom_idx == ie, :] = torch.FloatTensor(image_data['x'][el])
            image_dfpdX[atom_idx == ie, :, :, :] = torch.FloatTensor(image_data['dx'][el])
            image_e_mask[atom_idx == ie, ie-1] = 1

        image_fp = (image_fp - self.scale['fp_min']) / (self.scale['fp_max'] -  self.scale['fp_min'] + 1e-10)
        image_dfpdX /= (self.scale['fp_max'] - self.scale['fp_min'] + 1e-10).view(1, -1, 1, 1)
        image_latent_rep, image_latent_rep_ww = latent_rep(image_fp, image_dfpdX, image_e_mask, self.model, self.elements)
        xyz_dists = []
        for dim in range(3):
            dim_dist, _ = self.train_nbrs[dim].kneighbors(image_latent_rep_ww[:, dim, :], n_neighbors=1)
            xyz_dists.append(dim_dist)
        latent_dist = np.mean(xyz_dists, axis=0)  # [N_atom * 1]
        latent_dist_all = xyz_dists

        image_fp.requires_grad = True    
        image_dfpdX = image_dfpdX.reshape(n_atoms*n_features, n_atoms*3)
        
        # calculate energy and forces
        image_nrg_pre_raw = self.model(image_fp)  # [N_atoms, N_elements]
        image_nrg_pre_cluster = torch.sum(image_nrg_pre_raw * image_e_mask)  # [N_atoms]
        image_b_dnrg_dfp = torch.autograd.grad(image_nrg_pre_cluster, image_fp, 
                                                create_graph=True, retain_graph=True)[0].reshape(1, -1)
        
        image_force_pre = - torch.mm(image_b_dnrg_dfp, image_dfpdX).reshape(n_atoms, 3)
        nrg_pred = image_nrg_pre_cluster.detach().item()
        frs_pred = image_force_pre.detach().numpy()

        self.energy = nrg_pred
        self.forces = frs_pred
        self.results['energy'] = nrg_pred
        self.results['forces'] = frs_pred
        self.results['latent_dist'] = latent_dist  # [N_atom, 1]
        self.results['max_latent_dist'] = latent_dist.max()  # float
        self.results['latent_rep'] = image_latent_rep  # [N_atom, 3, rep_dim]
        self.results['latent_rep_ww'] = image_latent_rep_ww
        self.results['latent_dist_all'] = latent_dist_all  # [N_atom, 1]


def latent_rep(cluster_fp, cluster_dfpdX, cluster_els, model, elements):
    sub_model = model.net[:-1]
    weights = model.net[-1].weight
    n_nodes = model.net[-1].in_features
    n_atoms = cluster_fp.size()[0]

    cluster_frs_rep = dict({_: torch.zeros(n_atoms, 3, n_nodes) for _ in elements})

    for atom_id in range(n_atoms):
        fp_i = cluster_fp[atom_id]
        el_i = cluster_els[atom_id].argmax().item()
        fp_i.requires_grad =True
        dpi_dfpi = torch.autograd.functional.jacobian(sub_model, fp_i, create_graph=True)
        dpi_dxi = torch.cat([(dpi_dfpi @ cluster_dfpdX[atom_id, :, :, k]).reshape([1, n_nodes, n_atoms]) 
                            for k in range(3)], dim=0).permute(2, 0, 1)
        cluster_frs_rep[elements[el_i]] -= dpi_dxi.detach()

    cluster_frs_rep = torch.cat([cluster_frs_rep[el] for el in elements], dim=2)
    weights_vector = torch.cat(list(weights)).detach()
    cluster_frs_rep_weighted = cluster_frs_rep * weights_vector
    return cluster_frs_rep.numpy(), cluster_frs_rep_weighted.numpy()


class NN_Calc_single_model(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, params_set, model, scale, elements, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.params_set = params_set
        self.model = model
        self.scale = scale
        self.elements = elements
        self.train_nbrs = train_nbrs  # list of nearest neighbors of training set in 3 dims


    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        temp_atoms = self.atoms.copy()
        temp_atoms.set_calculator(SPC(temp_atoms, energy=0.0, forces=np.zeros([len(temp_atoms), 3])))
        nrg_pred, frs_pred = calculate_atoms(temp_atoms, self.model, self.scale, self.parameters, self.elements)
        
        self.energy = nrg_pred
        self.forces = frs_pred
        self.results['energy'] = self.energy
        self.results['forces'] = self.forces

