import torch
from ase.db import connect
import os
import numpy as np


def get_scaling(train_data):
	train_fp = train_data['b_fp']
	fp_max = torch.max(train_fp.view(-1, train_fp.size(2)), dim=0)
	fp_min = torch.min(train_fp.view(-1, train_fp.size(2)), dim=0)
	train_nrg = train_data['b_e']
	nrg_max = torch.max(train_nrg)
	nrg_min = torch.min(train_nrg)

	scale = {'fp_max': fp_max[0], 'fp_min': fp_min[0], 'nrg_max': nrg_max, 'nrg_min': nrg_min}
	return scale


class BPNN(torch.nn.Module):
	def __init__(self, n_fp, layer_nodes, activations, bias=True):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		layer_nodes: list of int, number of nodes in each layer
		activation: str, "tanh", "sigmoid", "relu"
		"""
		super().__init__()
		acts = {'tanh': torch.nn.Tanh(), 'sigmoid': torch.nn.Sigmoid(), 'relu': torch.nn.ReLU()}
		layers = [torch.nn.Linear(n_fp, layer_nodes[0], bias=bias)]
		layers += [acts[activations[0]]]
		for i in range(len(layer_nodes)-1):
			layers += [torch.nn.Linear(layer_nodes[i], layer_nodes[i+1], bias=bias)]
			layers += [acts[activations[i+1]]]
		layers += [torch.nn.Linear(layer_nodes[-1], 1, bias=bias)]
		self.net = torch.nn.Sequential(*layers)

	def forward(self, x):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		return self.net(x)


class Agent(object):
	def __init__(self, train_data, valid_data, model_paths, test_data=None, layer_nodes=[10, 10], activation='tanh', 
				optim_params={'algo': 'L-BFGS', 'lr': 1, 'max_iter': 10, 'history_size': 100, 'line_search_fn': 'strong_wolfe'}, device=torch.device('cpu')):
		"""
		layer_nodes: list of int, # of nodes in the each layer
		activation: str, "tanh", "Sigmoid" or "relu"
		lr, max_iter and history_size: float, int, int， parameters for LBFGS optimization method in pytorch
		device: torch.device, cpu or cuda
		"""

		n_element = train_data['b_e_mask'].size(2)
		n_fp = train_data['b_fp'].size(2)

		self.train_data = train_data
		self.valid_data = valid_data
		self.test_data = test_data

		self.models = [BPNN(n_fp, layer_nodes, activation).to(device) for _ in range(n_element)]
		params = [param for model in self.models for param in list(model.parameters())]

		opt_type = optim_params['algo']
		if opt_type == 'L-BFGS':
			self.optimizer = torch.optim.LBFGS(params, lr=optim_params['lr'], max_iter=optim_params['max_iter'], 
								history_size=optim_params['history_size'], line_search_fn=optim_params['line_search_fn'])
		elif opt_type == 'Adam':
			self.optimizer = torch.optim.Adam(params, lr=optim_params['lr'])

		self.model_paths = model_paths


	def train(self, log_name, n_epoch=1000, interupt=True, val_interval=10, is_force=True, nrg_convg=1, force_convg=15, nrg_coef=1, force_coef=25, batch_size=None):
		"""
		interupt: bool, if interupt training process when the nrg_convg and force_convg criteria satisfied
		val_interval: int: interval steps to evaluate on the validation and test datasets
		is_force: bool, if training with forces
		nrg_coef, force_coef: float, coefficients for energy and force in loss function,
							  force_coef will be ignored automatically if is_force is False
		"""
		
		f = open(log_name, 'w')
		f.close()

		mse = torch.nn.MSELoss()
		mae = torch.nn.L1Loss()
		sum_l1 = torch.nn.L1Loss(reduction='sum')

		# preprocess 
		train_b_fp = self.train_data['b_fp']
		train_b_fp.requires_grad = True
		train_n_clusters, train_n_atoms, train_n_fp = train_b_fp.size(0), train_b_fp.size(1), train_b_fp.size(2)
		train_b_dfpdX = self.train_data['b_dfpdX'].reshape(train_n_clusters, train_n_atoms*train_n_fp, train_n_atoms*3)
		train_nrg_label_cluster = self.train_data['b_e']
		train_force_label = self.train_data['b_f']
		train_actual_atoms = self.train_data['N_atoms'].squeeze()  # actual number of atoms in each cluster

		valid_b_fp = self.valid_data['b_fp']
		valid_b_fp.requires_grad = True
		valid_n_clusters, valid_n_atoms, valid_n_fp = valid_b_fp.size(0), valid_b_fp.size(1), valid_b_fp.size(2)
		valid_b_dfpdX = self.valid_data['b_dfpdX'].reshape(valid_n_clusters, valid_n_atoms*valid_n_fp, valid_n_atoms*3)
		valid_nrg_label_cluster = self.valid_data['b_e']
		valid_force_label = self.valid_data['b_f']
		valid_actual_atoms = self.valid_data['N_atoms'].squeeze()

		if self.test_data is not None:
			test_b_fp = self.test_data['b_fp']
			test_n_clusters, test_n_atoms, test_n_fp = test_b_fp.size(0), test_b_fp.size(1), test_b_fp.size(2) 
			test_b_fp.requires_grad = True
			test_b_dfpdX = self.test_data['b_dfpdX'].reshape(test_n_clusters, test_n_atoms*test_n_fp, test_n_atoms*3)
			test_nrg_label_cluster = self.test_data['b_e']
			test_force_label = self.test_data['b_f']
			test_actual_atoms = self.test_data['N_atoms'].squeeze()

		if not is_force:
			test_force_mae = 0

		# test before training
		if self.test_data is not None:				
			test_nrg_pre_raw = torch.cat([model(test_b_fp) for model in self.models], dim=2)
			test_nrg_pre_atom = torch.sum(test_nrg_pre_raw * self.test_data['b_e_mask'], dim=2)
			test_nrg_pre_cluster = torch.sum(test_nrg_pre_atom, dim=1)
			test_nrg_mae = mae(test_nrg_pre_cluster/test_actual_atoms, test_nrg_label_cluster/test_actual_atoms)

			if is_force:
				test_b_dnrg_dfp = torch.autograd.grad(test_nrg_pre_cluster, test_b_fp, grad_outputs=torch.ones_like(test_nrg_pre_cluster), 
											   create_graph=True, retain_graph=True)[0].reshape(test_n_clusters, 1, -1)
				test_force_pre = - torch.bmm(test_b_dnrg_dfp, test_b_dfpdX).reshape(test_n_clusters,test_n_atoms,3)
				test_force_mae = sum_l1(test_force_pre, test_force_label) / torch.sum(test_actual_atoms) / 3

			with open(log_name, 'a') as file:
				file.write(f'test: epoch: -1, nrg_mae: {test_nrg_mae*1000} meV/atom, force_mae: {test_force_mae*1000} meV/AA\r\n')

		for epo in range(n_epoch):
			total_train = train_b_fp.size(0)
			if not batch_size:
				batch_size = train_b_fp.size(0)
			n_batchs = total_train // batch_size
			indices = np.arange(total_train)
			np.random.shuffle(indices)
			for i_b in range(n_batchs):
				low = i_b * batch_size
				up = min((i_b+1)*batch_size, total_train)
				if (i_b + 2) * batch_size > total_train:
					up = total_train
				s_idx = indices[low:up]

				def closure():
					self.optimizer.zero_grad()
					batch_fp = train_b_fp[s_idx]
					train_nrg_pre_raw = torch.cat([model(batch_fp) for model in self.models], dim=2)
					train_nrg_pre_atom = torch.sum(train_nrg_pre_raw*self.train_data['b_e_mask'][s_idx], dim=2)
					train_nrg_pre_cluster = torch.sum(train_nrg_pre_atom, dim=1)
					train_loss = mse(train_nrg_pre_cluster/train_actual_atoms[s_idx], train_nrg_label_cluster[s_idx]/train_actual_atoms[s_idx]) * nrg_coef
					train_nrg_mae = mae(train_nrg_pre_cluster/train_actual_atoms[s_idx], train_nrg_label_cluster[s_idx]/train_actual_atoms[s_idx])

					if is_force:
						train_b_dnrg_dfp = torch.autograd.grad(train_nrg_pre_cluster, batch_fp, grad_outputs=torch.ones_like(train_nrg_pre_cluster), 
													create_graph=True, retain_graph=True)[0].reshape(len(s_idx), 1, -1)
						train_force_pre = - torch.bmm(train_b_dnrg_dfp, train_b_dfpdX[s_idx]).reshape(len(s_idx), train_n_atoms, 3)
						train_force_loss = mse(train_force_pre, train_force_label[s_idx]) * force_coef
						train_loss += train_force_loss
						train_force_mae = sum_l1(train_force_pre, train_force_label[s_idx]) / torch.sum(train_actual_atoms[s_idx]) / 3
					else:
						train_force_loss = 0
						train_force_mae = 0

					train_loss.backward(retain_graph=True)
					with open(log_name, 'a') as file:
						file.write(f'epoch: {epo}, nrg_mae: {train_nrg_mae*1000} meV/atom, force_mae: {train_force_mae*1000} meV/AA\r\n')
					return train_loss
				self.optimizer.step(closure)


			if epo % val_interval == 0:
				train_nrg_pre_raw = torch.cat([model(train_b_fp) for model in self.models], dim=2)
				train_nrg_pre_atom = torch.sum(train_nrg_pre_raw*self.train_data['b_e_mask'], dim=2)
				train_nrg_pre_cluster = torch.sum(train_nrg_pre_atom, dim=1)
				train_nrg_mae = mae(train_nrg_pre_cluster/train_actual_atoms, train_nrg_label_cluster/train_actual_atoms)

				if is_force:
					train_b_dnrg_dfp = torch.autograd.grad(train_nrg_pre_cluster, train_b_fp, grad_outputs=torch.ones_like(train_nrg_pre_cluster), 
												create_graph=True, retain_graph=True)[0].reshape(train_n_clusters, 1, -1)
					train_force_pre = - torch.bmm(train_b_dnrg_dfp, train_b_dfpdX).reshape(train_n_clusters, train_n_atoms, 3)
					train_force_mae = sum_l1(train_force_pre, train_force_label) / torch.sum(train_actual_atoms) / 3
				else:
					train_force_mae = 0
				print(f'train: epoch: {epo}, nrg_mae: {train_nrg_mae*1000} meV/atom, force_mae: {train_force_mae*1000} meV/AA')

				valid_nrg_pre_raw = torch.cat([model(valid_b_fp) for model in self.models], dim=2)
				valid_nrg_pre_atom = torch.sum(valid_nrg_pre_raw * self.valid_data['b_e_mask'], dim=2)
				valid_nrg_pre_cluster = torch.sum(valid_nrg_pre_atom, dim=1)
				valid_nrg_mae = mae(valid_nrg_pre_cluster/valid_actual_atoms, valid_nrg_label_cluster/valid_actual_atoms)
				if is_force:
					valid_b_dnrg_dfp = torch.autograd.grad(valid_nrg_pre_cluster, valid_b_fp, grad_outputs=torch.ones_like(valid_nrg_pre_cluster), 
									   create_graph=True, retain_graph=True)[0].reshape(valid_n_clusters, 1, -1)
					valid_force_pre = - torch.bmm(valid_b_dnrg_dfp, valid_b_dfpdX).reshape(valid_n_clusters,valid_n_atoms,3)
					valid_force_mae = sum_l1(valid_force_pre, valid_force_label) / torch.sum(valid_actual_atoms) / 3
					valid_force_max = torch.abs(valid_force_pre - valid_force_label).max()
				else:
					valid_force_mae = 0
				with open(log_name, 'a') as file:
					file.write(
						f'validation: epoch: {epo}, nrg_mae: {valid_nrg_mae*1000} meV/atom, force_mae: {valid_force_mae*1000} meV/AA, max fae: {valid_force_max*1000} meV/AA \r\n')
				print(f'validation: epoch: {epo}, nrg_mae: {valid_nrg_mae*1000} meV/atom, force_mae: {valid_force_mae*1000} meV/AA, max fae: {valid_force_max*1000} meV/AA')

				if self.test_data is not None:					
					test_nrg_pre_raw = torch.cat([model(test_b_fp) for model in self.models], dim=2)
					test_nrg_pre_atom = torch.sum(test_nrg_pre_raw * self.test_data['b_e_mask'], dim=2)
					test_nrg_pre_cluster = torch.sum(test_nrg_pre_atom, dim=1)
					test_nrg_mae = mae(test_nrg_pre_cluster/test_actual_atoms, test_nrg_label_cluster/test_actual_atoms)
					if is_force:
						test_b_dnrg_dfp = torch.autograd.grad(test_nrg_pre_cluster, test_b_fp, grad_outputs=torch.ones_like(test_nrg_pre_cluster), 
													   create_graph=True, retain_graph=True)[0].reshape(test_n_clusters, 1, -1)
						test_force_pre = - torch.bmm(test_b_dnrg_dfp, test_b_dfpdX).reshape(test_n_clusters,test_n_atoms,3)
						test_force_mae = sum_l1(test_force_pre, test_force_label) / torch.sum(test_actual_atoms) / 3
					else:
						test_force_mae = 0
					with open(log_name, 'a') as file:
						file.write(f'test: epoch: {epo}, nrg_mae: {test_nrg_mae*1000} meV/atom, force_mae: {test_force_mae*1000} meV/AA\r\n')

				self.save_model()
				if interupt and (valid_nrg_mae*1000 <= nrg_convg) and (valid_force_mae*1000 <= force_convg):
					print('condition satisfied\r\n')
					with open(log_name, 'a') as file:
						file.write('condition satisfied\r\n')
					break


	def save_model(self):
		for i in range(len(self.models)):
			torch.save(self.models[i].state_dict(), self.model_paths[i])


	def load_model(self):
		for i in range(len(self.models)):
			self.models[i].load_state_dict(torch.load(self.model_paths[i]))


	def evaluate(self, test_data, is_force=False):
		self.load_model()
		test_b_fp = test_data['b_fp']
		test_n_clusters, test_n_atoms, test_n_fp = test_b_fp.size(0), test_b_fp.size(1), test_b_fp.size(2) 
		test_nrg_label_cluster = test_data['b_e']
		test_actual_atoms = test_data['N_atoms'].squeeze()
		if is_force:
			test_b_fp.requires_grad = True
			test_n_clusters = test_b_fp.size(0)
			test_b_dfpdX = test_data['b_dfpdX'].reshape(test_n_clusters, test_n_atoms*test_n_fp, test_n_atoms*3)  # N_clusters * N_atoms * N_fp * N_atoms * 3
			test_force_label = test_data['b_f']
		
		mae = torch.nn.L1Loss()
		sum_l1 = torch.nn.L1Loss(reduction='sum')

		nrg_pre_raw = torch.cat([model(test_b_fp) for model in self.models], dim=2)
		nrg_pre_atom = torch.sum(nrg_pre_raw * test_data['b_e_mask'], dim=2)
		nrg_pre_cluster = torch.sum(nrg_pre_atom, dim=1)
		nrg_mae = mae(nrg_pre_cluster/test_actual_atoms, test_nrg_label_cluster/test_actual_atoms)

		if is_force:
			b_dnrg_dfp = torch.autograd.grad(nrg_pre_cluster, test_b_fp, grad_outputs=torch.ones_like(nrg_pre_cluster), 
										   create_graph=True, retain_graph=True)[0].reshape(test_n_clusters, 1, -1)
			force_pre = - torch.bmm(b_dnrg_dfp, test_b_dfpdX).reshape(test_n_clusters, test_n_atoms, 3)
			force_mae = sum_l1(force_pre, test_force_label) / torch.sum(test_actual_atoms) / 3
		else:
			force_pre = 0
			force_mae = 0

		return nrg_pre_cluster.detach().cpu(), force_pre.detach().cpu(), nrg_mae.detach().cpu(), force_mae.detach().cpu()
