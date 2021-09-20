from ase.io.trajectory import Trajectory
import numpy as np
from copy import deepcopy
from ase.optimize.optimize import Optimizer, Dynamics
from ase.optimize import BFGS
from math import sqrt
import time


class Gradient_Descent():
	"""
	Used to update positions according to DFT for just one step
	"""
	def __init__(self, atoms, lr=0.015, log_path=None, traj_path=None):
		self.lr = lr
		self.atoms = atoms
		self.log_path = log_path
		self.traj_path = traj_path
		
	def run(self, fmax=0.05, maxstep=0.001, steps=1):
		if not steps:
			return
		if self.log_path:
			log = open(self.log_path, 'w')
		if self.traj_path:
			traj = Trajectory(self.traj_path, 'w')

		for step in range(steps):
			r0 = deepcopy(self.atoms.positions)
			f = self.atoms.get_forces()  # -dE/dx
			m_f = np.linalg.norm(f, axis=1).max()
			nrg = self.atoms.get_potential_energy()

			if self.log_path:
				log.write(f'step {step}, nrg: {nrg}, max_f: {m_f}\n')
			else:
				print(f'step {step}, nrg: {nrg}, max_f: {m_f}')
			if self.traj_path:
				traj.write(self.atoms)
			if m_f < fmax:
				break
			dr = self.lr * f
			print(np.max(dr))
			# check the max length of dr
			max_step_length = np.linalg.norm(dr, axis=1).max()
			if max_step_length > maxstep:
				scale = maxstep / max_step_length
				dr = dr * scale
			r = r0 + dr
			self.atoms.set_positions(r)

		if self.traj_path:
			traj.write(self.atoms)
		if self.log_path:
			log.close()
		if self.traj_path:
			traj.close()


class Gradient_Descent_Ensemble():
	"""
	Gradient descent with monitoring latent distance,
	should work with NN calculator with ensemble
	"""
	def __init__(self, atoms, threshold=0.03, lr=1, log_path=None, traj_path=None):
		self.lr = lr
		self.atoms = atoms
		self.log_path = log_path
		self.traj_path = traj_path
		self.threshold = threshold
		self.consts = atoms.constraints

	def run(self, fmax=0.035, maxstep=0.04, steps=None):
		if not steps:
			return
		if self.log_path:
			log = open(self.log_path, 'w')
			log.write(f'force threshold: {fmax}, std threshold {self.threshold} \n')
		else:
			print(f'force threshold: {fmax}, std threshold {self.threshold} \n')
		if self.traj_path:
			traj = Trajectory(self.traj_path, 'w')

		is_certain = True
		for step in range(steps):
			r0 = deepcopy(self.atoms.positions)
			rf = self.atoms.get_forces()
			for const in self.consts:
				const.adjust_forces(None, rf)
			m_f = np.linalg.norm(rf, axis=1).max()
			nrg = self.atoms.get_potential_energy()
			nrg_std = self.atoms.calc.results['energy_std']
			frs_stds = self.atoms.calc.results['forces_std']  # N_atom * 3
			for const in self.consts:
				const.adjust_forces(None, frs_stds)
			max_mean_frs_std = frs_stds.mean(axis=1).max()
			if nrg_std > self.threshold:
				if step > 1:  # when step == 1, it is more possible that our threshold is too low
					is_certain = False
				break
			if m_f < fmax:
				break
			if self.log_path:
				log.write(f'step {step}, nrg: {nrg}, max_f: {m_f}, nrg_std: {nrg_std}, m_f_m_std: {max_mean_frs_std}\n')
			else:
				print(f'step {step}, nrg: {nrg}, max_f: {m_f}, nrg_std: {nrg_std}, m_f_m_std: {max_mean_frs_std}')
			if self.traj_path:
				traj.write(self.atoms)

			# update
			dr = self.lr * rf
			# check the max length of dr
			max_step_length = np.linalg.norm(dr, axis=1).max()
			if max_step_length > maxstep:
				scale = maxstep / max_step_length
				dr = dr * scale
			r = r0 + dr
			self.atoms.set_positions(r)

		if is_certain:  # if certain about this step
			if self.log_path:
				log.write(f'step {step}, nrg: {nrg}, max_f: {m_f} nrg_std: {nrg_std}, m_f_m_std: {max_mean_frs_std}\n')
				log.write(f'relaxed with certainty \n')
			else:
				print(f'step {step}, nrg: {nrg}, max_f: {m_f} nrg_std: {nrg_std}, m_f_m_std: {max_mean_frs_std}')
				print(f'relaxed with certainty \n')
			if self.traj_path:
				traj.write(self.atoms)
		else:
			if self.log_path:
				log.write(f'end with uncertain configuration\n')
			else:
				print(f'end with uncertain configuration')

		if self.log_path:
			log.close()
		if self.traj_path:
			traj.close()


class BFGS_Ensemble(BFGS):
	def __init__(self, atoms, restart=None, logfile='-', trajectory=None, maxstep=0.04, 
				master=None, alpha=None, nrg_std=None):
		super().__init__(atoms, restart, logfile, trajectory, maxstep, master)
		self.nrg_std = nrg_std

	def converged(self, forces=None):
		if forces is None:
			forces = self.atoms.get_forces()
		if hasattr(self.atoms, "get_curvature"):
			return (forces ** 2).sum(axis=1).max() < self.fmax ** 2 \
				and self.atoms.get_curvature() < 0.0
		condition1 = (forces ** 2).sum(axis=1).max() < self.fmax ** 2
		condition2 = self.atoms.calc.results['energy_std'] > self.nrg_std
		return condition1 or condition2

	def log(self, forces=None):
		if forces is None:
			forces = self.atoms.get_forces()
		fmax = sqrt((forces ** 2).sum(axis=1).max())
		e = self.atoms.get_potential_energy(
			force_consistent=self.force_consistent
		)

		nrg_std = round(self.atoms.calc.results['energy_std'], 4)
		T = time.localtime()
		if self.logfile is not None:
			name = self.__class__.__name__
			if self.nsteps == 0:
				args = (" " * len(name), "Step", "Time", "Energy", "fmax", "nrg_std")
				msg = "%s  %4s %8s %15s %12s %12s \n" % args
				self.logfile.write(msg)

				if self.force_consistent:
					msg = "*Force-consistent energies used in optimization.\n"
					self.logfile.write(msg)

			ast = {1: "*", 0: ""}[self.force_consistent]
			args = (name, self.nsteps, T[3], T[4], T[5], e, ast, fmax, nrg_std)
			msg = "%s:  %3d %02d:%02d:%02d %15.6f%1s %12.4f %12.4f \n" % args
			self.logfile.write(msg)

			self.logfile.flush()