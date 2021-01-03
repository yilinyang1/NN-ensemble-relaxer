from ._libsymf import lib, ffi
from .gen_ffi import _gen_2Darray_for_ffi
import numpy as np
from .mpiclass import DummyMPI, MPI4PY
import torch
from time import time

def _read_params(filename):
	params_i = list()
	params_d = list()
	with open(filename, 'r') as fil:
		for line in fil:
			tmp = line.split()
			params_i += [list(map(int,   tmp[:3]))]
			params_d += [list(map(float, tmp[3:]))]

	params_i = np.asarray(params_i, dtype=np.intc, order='C')
	params_d = np.asarray(params_d, dtype=np.float64, order='C')

	return params_i, params_d

def calculate_fp(atoms, elements, params_set):
	"""
		atoms: ase Atoms class
		symbols: list of unique elements in atoms
	"""
	is_mpi = False

	try:
		import mpi4py
	except ImportError:
		comm = DummyMPI()
	else:
		if is_mpi:
			comm = MPI4PY()
		else:
			comm = DummyMPI()

	cart = np.copy(atoms.get_positions(wrap=True), order='C')
	scale = np.copy(atoms.get_scaled_positions(), order='C')
	cell = np.copy(atoms.cell, order='C')

	cart_p  = _gen_2Darray_for_ffi(cart, ffi)
	scale_p = _gen_2Darray_for_ffi(scale, ffi)
	cell_p  = _gen_2Darray_for_ffi(cell, ffi)

	atom_num = len(atoms.positions)
	symbols = np.array(atoms.get_chemical_symbols())
	atom_i = np.zeros([len(symbols)], dtype=np.intc, order='C')
	type_num = dict()
	type_idx = dict()
	for j,jtem in enumerate(elements):
		tmp = symbols==jtem
		atom_i[tmp] = j+1
		type_num[jtem] = np.sum(tmp).astype(np.int64)
		# if atom indexs are sorted by atom type,
		# indexs are sorted in this part.
		# if not, it could generate bug in training process for force training
		type_idx[jtem] = np.arange(atom_num)[tmp]
	atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

	res = dict()
	res['x'] = dict()
	res['dx'] = dict()
	res['params'] = dict()
	res['N'] = type_num
	res['tot_num'] = np.sum(list(type_num.values()))
	res['partition'] = np.ones([res['tot_num']]).astype(np.int32)
	res['E'] = atoms.get_total_energy()
	res['F'] = atoms.get_forces()
	res['atom_idx'] = atom_i

	for j,jtem in enumerate(elements):
		q = type_num[jtem] // comm.size
		r = type_num[jtem] %  comm.size

		begin = comm.rank * q + min(comm.rank, r)
		end = begin + q
		if r > comm.rank:
			end += 1

		cal_atoms = np.asarray(type_idx[jtem][begin:end], dtype=np.intc, order='C')
		cal_num = len(cal_atoms)
		cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

		x = np.zeros([cal_num, params_set[jtem]['num']], dtype=np.float64, order='C')
		dx = np.zeros([cal_num, atom_num * params_set[jtem]['num'] * 3], dtype=np.float64, order='C')

		x_p = _gen_2Darray_for_ffi(x, ffi)
		dx_p = _gen_2Darray_for_ffi(dx, ffi)
		errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
						 atom_i_p, atom_num, cal_atoms_p, cal_num, \
						 params_set[jtem]['ip'], params_set[jtem]['dp'], params_set[jtem]['num'], \
						 x_p, dx_p)
		comm.barrier()
		errnos = comm.gather(errno)
		errnos = comm.bcast(errnos)

		if isinstance(errnos, int):
			errnos = [errno]

		for errno in errnos:
			if errno == 1:
				err = "Not implemented symmetry function type."
				raise NotImplementedError(err)
			elif errno == 2:
				err = "Zeta in G4/G5 must be greater or equal to 1.0."
				raise ValueError(err)
			else:
				assert errno == 0

		
		if type_num[jtem] != 0:
			res['x'][jtem] = np.array(comm.gather(x, root=0))
			res['dx'][jtem] = np.array(comm.gather(dx, root=0))
			
			if comm.rank == 0:
				res['x'][jtem] = np.concatenate(res['x'][jtem], axis=0).reshape([type_num[jtem], params_set[jtem]['num']])
				res['dx'][jtem] = np.concatenate(res['dx'][jtem], axis=0).\
									reshape([type_num[jtem], params_set[jtem]['num'], atom_num, 3])
				res['partition_'+jtem] = np.ones([type_num[jtem]]).astype(np.int32)
		else:
			res['x'][jtem] = np.zeros([0, params_set[jtem]['num']])
			res['dx'][jtem] = np.zeros([0, params_set[jtem]['num'], atom_num, 3])
			res['partition_'+jtem] = np.ones([0]).astype(np.int32)
		res['params'][jtem] = params_set[jtem]['total']
	return res



def set_sym(elements, Gs, cutoff, g2_etas=None, g2_Rses=None, g4_etas=None, g4_zetas=None, g4_lambdas=None):
	"""
	specify symmetry function parameters for each element
	parameters for each element contain:
	integer parameters: [which sym func, surrounding element 1, surrounding element 1]
						surrouding element starts from 1. For G2 sym func, the third 
						element is 0. For G4 and G5, the order of the second and the
						third element does not matter.
	double parameters:  [cutoff radius, 3 sym func parameters]
						for G2: eta, Rs, dummy
						for G4 and G5: eta, zeta, lambda
	"""

	# specify all elements in the system
	params_set = dict()
	ratio = 36.0  # difference ratio from the AMP parameters
	for item in elements:
		params_set[item] = dict()
		int_params = []
		double_params = []
		for G in Gs:
			if G == 2:
				int_params += [[G, el1, 0] for el1 in range(1, len(elements)+1) 
										   for g2_eta in g2_etas
										   for g2_Rs in g2_Rses]
				double_params += [[cutoff, g2_eta/ratio, g2_Rs, 0] for el1 in range(1, len(elements)+1)
																   for g2_eta in g2_etas
																   for g2_Rs in g2_Rses]
			else:
				int_params += [[G, el1, el2] for el1 in range(1, len(elements)+1)
											 for el2 in range(el1, len(elements)+1)
											 for g4_eta in g4_etas
											 for g4_zeta in g4_zetas
											 for g4_lambda in g4_lambdas]
				double_params += [[cutoff, g4_eta/ratio, g4_zeta, g4_lambda] 
											 for el1 in range(1, len(elements)+1)
											 for el2 in range(el1, len(elements)+1)
											 for g4_eta in g4_etas
											 for g4_zeta in g4_zetas
											 for g4_lambda in g4_lambdas]


		params_set[item]['i'] = np.array(int_params, dtype=np.intc)
		params_set[item]['d'] = np.array(double_params, dtype=np.float64)
		params_set[item]['ip'] = _gen_2Darray_for_ffi(params_set[item]['i'], ffi, "int")
		params_set[item]['dp'] = _gen_2Darray_for_ffi(params_set[item]['d'], ffi)
		params_set[item]['total'] = np.concatenate((params_set[item]['i'], params_set[item]['d']), axis=1)
		params_set[item]['num'] = len(params_set[item]['total'])

	return params_set


def db_to_fp(db, params_set):
	N_max = 0
	N = db.count()
	elements = [key for key in params_set.keys()]
	n_features = len(params_set[elements[0]]['total'])

	for entry in db.select():
		if len(entry.symbols) > N_max:
			N_max = len(entry.symbols)

	N_atoms = torch.zeros(N)
	b_fp = torch.zeros((N, N_max, n_features))
	b_dfpdX = torch.zeros((N, N_max, n_features, N_max, 3))
	b_e_mask = torch.zeros((N, N_max, len(elements)))
	b_e = torch.zeros(N)
	b_f = torch.zeros((N, N_max, 3))

	idx = 0

	for entry in db.select():
		image = entry.toatoms()
		N_atoms[idx] = len(image)
		data = calculate_fp(image, elements, params_set)
		atom_idx = np.zeros(N_max)
		atom_idx[:len(image)] = data['atom_idx']
		for ie in range(1, len(elements) + 1):
			el = elements[ie-1]
			b_fp[idx, atom_idx == ie, :] = torch.FloatTensor(data['x'][el])
			b_dfpdX[idx, atom_idx == ie, :, :len(image), :] = torch.FloatTensor(data['dx'][el])
			b_e_mask[idx, atom_idx == ie, ie-1] = 1

		b_e[idx] = image.get_potential_energy()
		b_f[idx][:len(image), :] = torch.FloatTensor(image.get_forces())
		idx += 1


	data = {'N_atoms': N_atoms, 'b_fp': b_fp, 'b_dfpdX': b_dfpdX, 
			'b_e': b_e, 'b_f': b_f, 'b_e_mask': b_e_mask}

	return data


def cal_fp_only(atoms, elements, params_set):
	"""
		atoms: ase Atoms class
		symbols: list of unique elements in atoms
	"""
	is_mpi = False

	try:
		import mpi4py
	except ImportError:
		comm = DummyMPI()
	else:
		if is_mpi:
			comm = MPI4PY()
		else:
			comm = DummyMPI()

	# print(comm.size)
	cart = np.copy(atoms.get_positions(wrap=True), order='C')  # positions of atoms
	scale = np.copy(atoms.get_scaled_positions(), order='C')
	cell = np.copy(atoms.cell, order='C')

	cart_p  = _gen_2Darray_for_ffi(cart, ffi)  # pointers
	scale_p = _gen_2Darray_for_ffi(scale, ffi)
	cell_p  = _gen_2Darray_for_ffi(cell, ffi)

	atom_num = len(atoms.positions)
	symbols = np.array(atoms.get_chemical_symbols())
	atom_i = np.zeros([len(symbols)], dtype=np.intc, order='C')
	type_num = dict()
	type_idx = dict()
	for j,jtem in enumerate(elements):
		tmp = symbols==jtem
		atom_i[tmp] = j+1  # indicate the symbol type of each atom
		type_num[jtem] = np.sum(tmp).astype(np.int64)
		# if atom indexs are sorted by atom type,
		# indexs are sorted in this part.
		# if not, it could generate bug in training process for force training
		type_idx[jtem] = np.arange(atom_num)[tmp]

	atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

	res = dict()
	res['x'] = dict()
	res['dx'] = dict()

	for j,jtem in enumerate(elements):
		q = type_num[jtem] // comm.size
		r = type_num[jtem] %  comm.size

		begin = comm.rank * q + min(comm.rank, r)
		end = begin + q
		if r > comm.rank:
			end += 1

		cal_atoms = np.asarray(type_idx[jtem][begin:end], dtype=np.intc, order='C')

		cal_num = len(cal_atoms)
		cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

		x = np.zeros([cal_num, params_set[jtem]['num']], dtype=np.float64, order='C')
		dx = np.zeros([cal_num, atom_num * params_set[jtem]['num'] * 3], dtype=np.float64, order='C')

		x_p = _gen_2Darray_for_ffi(x, ffi)
		dx_p = _gen_2Darray_for_ffi(dx, ffi)
		errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
						 atom_i_p, atom_num, cal_atoms_p, cal_num, \
						 params_set[jtem]['ip'], params_set[jtem]['dp'], params_set[jtem]['num'], \
						 x_p, dx_p)
		comm.barrier()
		errnos = comm.gather(errno)
		errnos = comm.bcast(errnos)

		if isinstance(errnos, int):
			errnos = [errno]

		for errno in errnos:
			if errno == 1:
				err = "Not implemented symmetry function type."
				raise NotImplementedError(err)
			elif errno == 2:
				err = "Zeta in G4/G5 must be greater or equal to 1.0."
				raise ValueError(err)
			else:
				assert errno == 0

		if type_num[jtem] != 0:
			res['x'][jtem] = np.array(comm.gather(x, root=0))
			# res['dx'][jtem] = np.array(comm.gather(dx, root=0))
			
			if comm.rank == 0:
				res['x'][jtem] = np.concatenate(res['x'][jtem], axis=0).reshape([type_num[jtem], params_set[jtem]['num']])
				# res['dx'][jtem] = np.concatenate(res['dx'][jtem], axis=0).\
				# 					reshape([type_num[jtem], params_set[jtem]['num'], atom_num, 3])
		else:
			res['x'][jtem] = np.zeros([0, params_set[jtem]['num']])
			# res['dx'][jtem] = np.zeros([0, params_set[jtem]['num'], atom_num, 3])

	return res



def batch_to_fp(batch, params_set):
	N_max = 0
	N = len(batch)
	elements = [key for key in params_set.keys()]
	n_features = len(params_set[elements[0]]['total'])

	for image in batch:
		if len(image) > N_max:
			N_max = len(image)

	N_atoms = torch.zeros(N)
	b_fp = torch.zeros((N, N_max, n_features))
	b_dfpdX = torch.zeros((N, N_max, n_features, N_max, 3))
	b_e_mask = torch.zeros((N, N_max, len(elements)))
	b_e = torch.zeros(N)
	b_f = torch.zeros((N, N_max, 3))

	idx = 0

	for image in batch:
		N_atoms[idx] = len(image)
		data = calculate_fp(image, elements, params_set)
		atom_idx = np.zeros(N_max)
		atom_idx[:len(image)] = data['atom_idx']
		for ie in range(1, len(elements) + 1):
			el = elements[ie-1]
			b_fp[idx, atom_idx == ie, :] = torch.FloatTensor(data['x'][el])
			b_dfpdX[idx, atom_idx == ie, :, :len(image), :] = torch.FloatTensor(data['dx'][el])
			b_e_mask[idx, atom_idx == ie, ie-1] = 1

		b_e[idx] = image.get_potential_energy()
		b_f[idx][:len(image), :] = torch.FloatTensor(image.get_forces())
		idx += 1


	data = {'N_atoms': N_atoms, 'b_fp': b_fp, 'b_dfpdX': b_dfpdX, 
			'b_e': b_e, 'b_f': b_f, 'b_e_mask': b_e_mask}

	return data



def conditional_cal_fp_only(atoms, elements, params_set, conditions):
	"""
		atoms: ase Atoms class
		symbols: list of unique elements in atoms
		condition: dict: {element: array of bools}, indicating if to cal the fp for each atom for each element
	"""
	is_mpi = False

	try:
		import mpi4py
	except ImportError:
		comm = DummyMPI()
	else:
		if is_mpi:
			comm = MPI4PY()
		else:
			comm = DummyMPI()

	cart = np.copy(atoms.get_positions(wrap=True), order='C')  # positions of atoms
	scale = np.copy(atoms.get_scaled_positions(), order='C')
	cell = np.copy(atoms.cell, order='C')

	cart_p  = _gen_2Darray_for_ffi(cart, ffi)  # pointers
	scale_p = _gen_2Darray_for_ffi(scale, ffi)
	cell_p  = _gen_2Darray_for_ffi(cell, ffi)

	atom_num = len(atoms.positions)
	symbols = np.array(atoms.get_chemical_symbols())
	atom_i = np.zeros([len(symbols)], dtype=np.intc, order='C')
	type_num = dict()
	type_idx = dict()
	for j,jtem in enumerate(elements):
		tmp = (symbols==jtem)
		atom_i[tmp] = j+1  # indicate the symbol type of each atom
		type_num[jtem] = np.sum(tmp & conditions).astype(np.int64)
		# if atom indexs are sorted by atom type,
		# indexs are sorted in this part.
		# if not, it could generate bug in training process for force training
		type_idx[jtem] = np.arange(atom_num)[tmp & conditions]

	atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

	res = dict()
	res['x'] = dict()
	res['dx'] = dict()

	for j,jtem in enumerate(elements):
		q = type_num[jtem] // comm.size
		r = type_num[jtem] %  comm.size

		begin = comm.rank * q + min(comm.rank, r)
		end = begin + q
		if r > comm.rank:
			end += 1

		cal_atoms = np.asarray(type_idx[jtem][begin:end], dtype=np.intc, order='C')

		cal_num = len(cal_atoms)
		cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

		x = np.zeros([cal_num, params_set[jtem]['num']], dtype=np.float64, order='C')
		dx = np.zeros([cal_num, atom_num * params_set[jtem]['num'] * 3], dtype=np.float64, order='C')

		x_p = _gen_2Darray_for_ffi(x, ffi)
		dx_p = _gen_2Darray_for_ffi(dx, ffi)
		errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
						 atom_i_p, atom_num, cal_atoms_p, cal_num, \
						 params_set[jtem]['ip'], params_set[jtem]['dp'], params_set[jtem]['num'], \
						 x_p, dx_p)
		comm.barrier()
		errnos = comm.gather(errno)
		errnos = comm.bcast(errnos)

		if isinstance(errnos, int):
			errnos = [errno]

		for errno in errnos:
			if errno == 1:
				err = "Not implemented symmetry function type."
				raise NotImplementedError(err)
			elif errno == 2:
				err = "Zeta in G4/G5 must be greater or equal to 1.0."
				raise ValueError(err)
			else:
				assert errno == 0

		if type_num[jtem] != 0:
			res['x'][jtem] = np.array(comm.gather(x, root=0))
			# res['dx'][jtem] = np.array(comm.gather(dx, root=0))
			
			if comm.rank == 0:
				res['x'][jtem] = np.concatenate(res['x'][jtem], axis=0).reshape([type_num[jtem], params_set[jtem]['num']])
				# res['dx'][jtem] = np.concatenate(res['dx'][jtem], axis=0).\
				# 					reshape([type_num[jtem], params_set[jtem]['num'], atom_num, 3])
		else:
			res['x'][jtem] = np.zeros([0, params_set[jtem]['num']])
			# res['dx'][jtem] = np.zeros([0, params_set[jtem]['num'], atom_num, 3])

	return res

