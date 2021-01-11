"""Frankenstein is an open source code that uses a Gaussian process to reconstruct the 1D radial brightness profile of a disc non-parametrically.

The Frank_Plotter class contains all tools necessary to read, model, and plot visibilities.
"""
import os

import numpy as np
from numpy import ndarray

import random

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D

from pathlib import Path

from copy import copy

from .utils import estimate_baseline_dependent_weight, readvis

from frank.radial_fitters import FrankFitter
from frank.geometry import FitGeometryGaussian, FixedGeometry
from frank.fit import load_data
from frank.utilities import UVDataBinner
from frank.utilities import draw_bootstrap_sample

class vis:
	"""Visibility data object.

	Visibility data exported in CASA from a .ms using ExportMS in reductiontools.py
	or exported from any other reduction software e.g. miriad.

	Examples
	--------
	Use plot to generate a passable axes for plotting.

	something something ... good example
	"""

	def __init__(self, filename=None, wle=None, **kwargs):
		#Read visibility data
		self.filename = filename
		self.wle = wle
		self._read(**kwargs)

	def _read(self):
		"""Load visibility data.

		Parameters
		----------
		filename
			The path to the file.
		"""
		# Set file_path
		file_path = Path(self.filename).expanduser()
		if not file_path.is_file():
			raise FileNotFoundError('Cannot find visibility data file')
		
		#Read visibilitiy data
		self.u, self.v, self.vis, self.wgt = readvis(filename=str(file_path), wle=self.wle)

	def binned_vis(self,
		inc: float = None,
		PA: float = None,
		dRA: float = 0.0,
		dDec: float = 0.0,
		bin_width: float = 0.0,
		deproject: bool = True, 
		):
		"""Bins the visibility data.

		Parameters
		----------
		inc:
			Inclination of the disc in degrees.
		PA:
			Position angle of the disc in degrees.
		dRA
			Right accession offsets in arcseconds.
		dDec
			Declination offsets in arcseconds.
		bins
			Spacing for binnings the visibilities

		Returns
		-------


		"""
		# convert keywords into relevant units
		incr = np.radians(inc)
		PAr = 0.5 * np.pi - np.radians(PA)
		dRA *= -np.pi / (180 * 3600)
		dDec *= -np.pi / (180 * 3600)

		# change to a deprojected, rotated coordinate system
		if deproject:
			uprime = (self.u * np.cos(PAr) + self.v * np.sin(PAr))
			vprime = (-self.u * np.sin(PAr) + self.v * np.cos(PAr)) * np.cos(incr)
		else:
			uprime = self.u
			vprime = self.v

		rhop = np.sqrt(uprime**2 + vprime**2)

		#phase shifts to account for offsets
		shifts = np.exp(-2 * np.pi * 1.0j * (self.u*-dRA + self.v*-dDec))
		visp = self.vis * shifts
		realp = visp.real
		imagp = visp.imag
		wgt = self.wgt



		# if requested, return a binned (averaged) representation
		if (bin_width > 0):
			max_bin=int(np.nanmax(rhop))
			bins = np.arange(0, max_bin+bin_width*1e3, bin_width*1e3)
			avbins = bins       # scale to lambda units (input in klambda)
			bwid = 0.5 * (avbins[1] - avbins[0])
			bvis = np.zeros_like(avbins, dtype='complex')
			berr_std = np.zeros_like(avbins, dtype='complex')
			berr_scat = np.zeros_like(avbins, dtype='complex')
			n_in_bin = np.zeros_like(avbins, dtype='int')
			for ib in np.arange(len(avbins)):
				inb = np.where((rhop >= avbins[ib] - bwid) & (rhop < avbins[ib] + bwid))
				if (len(inb[0]) >= 5):
					bRe, eRemu = np.average(realp[inb], weights=wgt[inb], returned=True)
					eRese = np.std(realp[inb])
					bIm, eImmu = np.average(imagp[inb], weights=wgt[inb], returned=True)
					eImse = np.std(imagp[inb])
					bvis[ib] = bRe + 1j*bIm
					berr_scat[ib] = eRese + 1j*eImse
					berr_std[ib] = 1 / np.sqrt(eRemu) + 1j / np.sqrt(eImmu)
					n_in_bin[ib] = np.size(bRe)
				else:
					bvis[ib] = 0 + 1j*0
					berr_scat[ib] = 0 + 1j*0
					berr_std[ib] = 0 + 1j*0
					n_in_bin[ib] = 0
			parser = np.where(berr_std.real != 0)
			output = bvis[parser], avbins[parser], berr_std[parser], berr_scat[parser], n_in_bin[parser]

			return output

		output = realp + 1j*imagp, rhop, 1 / np.sqrt(wgt), 1 / np.sqrt(wgt), np.zeros_like(rhop)
		return output

	def plot(self,
		inc: float = None,
		PA: float = None,
		dRA: float = 0.0,
		dDec: float = 0.0,
		bin_width: float = 0.0,
		ax: ndarray = None,
		ax_kwargs={},
		):
		"""Plot visibility data.

		Parameters
		----------
		inc:
			Inclination of the disc in degrees.
		PA:
			Position angle of the disc in degrees.
		dRA
			Right accession offsets in arcseconds.
		dDec
			Declination offsets in arcseconds.
		bins
			Spacing for binnings the visibilities
		ax
			A matplotlib Axes handle.
		ax_kwargs
			Keyword arguments to pass to contour map matplotlib Axes.
		**kwargs
			Additional keyword arguments to pass to interpolation
			and matplotlib functions.

		Returns
		-------
		ax
			The matplotlib Axes object.

		Notes
		-----
		Any notes?

		Examples
		--------
		Put good example here.

		"""

		if ax is None:
			fig, ax = plt.subplots()
		else:
			fig = ax.figure

		#deproject and bin visibilities
		vis, rhop, err_std, err_scat, bins = self.binned_vis(inc, PA, dRA, dDec, bin_width)

		ax.scatter(rhop/1e3, vis.real, **ax_kwargs)
		ax.axhline(0, linewidth=1, alpha=1, color="k", ls='--')

		#errors
		err = np.sqrt(err_std**2+err_scat**2)/2

		ax.errorbar(rhop/1e3, vis.real, yerr=err, fmt='none')

		return ax

	def frank_model(self,
		Rmax: float = 3.0,
		N: float = 250,
		dRA: float = 0.0,
		dDec: float = 0.0,
		inc: float = None,
		PA: float = None,
		alpha: ndarray = None, 		
		ws: ndarray = None, 
		bin_width: float = None, 
		est_weights: bool = False,
		save_model: bool = False,
		):
		"""Calculates Frankenstein models.

		Parameters
		----------
		R

		N

		dRA
			Right accession offsets in arcseconds.
		dDec
			Declination offsets in arcseconds.
		inc:
			Inclination of the disc in degrees.
		PA:
			Position angle of the disc in degrees.
		alpha

		ws

		bin_width
			Spacing for binnings the visibilities
		est_weights

		save_model


		Returns
		-------
		binned_vis

		sol

		model_grid
		"""

		if est_weights == True:
			baselines = (self.u**2 + self.v**2)**.5
			weights = estimate_baseline_dependent_weight(baselines, self.vis, bin_width)
		else:
			weights = self.wgt

		disk_geometry = FixedGeometry(float(inc), float(PA), dRA=dRA, dDec=dDec)
		FF = FrankFitter(Rmax=Rmax, N=N, geometry=disk_geometry, alpha=alpha, weights_smooth=ws)
		print('Calculating Frankenstin model')
		sol = FF.fit(self.u, self.v, self.vis, weights)

		#Deprojecting
		u_deproj, v_deproj, vis_deproj = sol.geometry.apply_correction(self.u, self.v, self.vis)
		baselines = (u_deproj**2 + v_deproj**2)**.5

		#Model grid
		model_grid = np.logspace(np.log10(min(baselines.min(), sol.q[0])), np.log10(max(baselines.max(), sol.q[-1])), 10**4)

		#Visibilities
		binned_vis = UVDataBinner(baselines, vis_deproj, weights, bin_width)
		
		return binned_vis, sol, model_grid

	def frank_plot(self,
		Rmax: float = 3.0,
		N: float = 250,
		dRA: float = 0.0,
		dDec: float = 0.0,
		inc: float = None,
		PA: float = None,
		alpha: ndarray = None, 		
		ws: ndarray = None, 
		bin_width: float = None, 
		est_weights: bool = False,
		normalise: bool = True,
		save_model: bool = False,
		imaginary: bool = False,
		ax: ndarray = None,
		ax0_kwargs={},
		ax1_kwargs={},
		ax2_kwargs={},
		kwargs={},
		):
		"""Plots Frankenstin models.

		Parameters
		----------
		R

		N

		dRA
			Right accession offsets in arcseconds.
		dDec
			Declination offsets in arcseconds.
		inc:
			Inclination of the disc in degrees.
		PA:
			Position angle of the disc in degrees.
		alpha

		ws

		bin_width
			Spacing for binnings the visibilities
		est_weights

		normalise

		save_model

		ax
			A matplotlib Axes handle.
		ax_kwargs
			Keyword arguments to pass to contour map matplotlib Axes.
		**kwargs
			Additional keyword arguments to pass to interpolation
			and matplotlib functions.

		Returns
		-------
		ax
			The matplotlib Axes object.
		"""

		if ax is None:
			fig, ax = plt.subplots()
		# else:
		# 	fig = ax.figure
		_kwargs0 = copy(ax0_kwargs)

		_kwargs1 = copy(ax2_kwargs)

		binned_vis, sol, model_grid = self.frank_model(Rmax=Rmax, N=N, dRA=dRA, dDec=dDec,
			inc=inc, PA=PA, alpha=alpha, ws=ws, bin_width=bin_width, est_weights=est_weights,
			save_model=save_model)

		#Solving for non-negative brightness profiles
		I_nn = sol.solve_non_negative()
		vis_model = sol.predict_deprojected(model_grid, I=I_nn).real

		if normalise:
			real_data = binned_vis.V.real.data
			factor = real_data[~np.isnan(real_data)]
			factor = factor[factor != 0]
			print('kl0 value: ', factor[0])
			real = binned_vis.V.real/factor[0]
			real_err = binned_vis.error.real/factor[0]
			img = binned_vis.V.imag/factor[0]
			img_err = binned_vis.error.imag/factor[0]
			vis_model *= 1/factor[0]
			I_nn *= 1/max(np.abs(I_nn))
			self.kl0 = factor[0]
		else:
			real = binned_vis.V.real
			real_err = binned_vis.error.real
			img = binned_vis.V.imag
			img_err = binned_vis.error.imag

		if imaginary:
			lw = _kwargs0.pop('lw', 4)
			ls = _kwargs0.pop('ls', '-')	
			c = _kwargs0.pop('c', 'black')
			s = _kwargs0.pop('s', 150)

			ax0_kwargs.pop('lw', None)
			ax0_kwargs.pop('ls', None)
			ax0_kwargs.pop('c', None)
			ax0_kwargs.pop('s', None)

			ax[0].axhline(0, linewidth=lw, alpha=1, color="k", ls='--')	
			ax[0].errorbar(binned_vis.uv/1e3, real, yerr=real_err, ecolor='black', 
				fmt='none', capsize=0, zorder=1, elinewidth=3, **ax0_kwargs)
			ax[0].scatter(binned_vis.uv/1e3, real, s=s, c=c, zorder=2, **ax0_kwargs)

			ax[1].axhline(0, linewidth=lw, alpha=1, color="k", ls='--')	
			ax[1].errorbar(binned_vis.uv/1e3, img, yerr=img_err, ecolor='black', 
				fmt='none', capsize=0, zorder=1, elinewidth=3, **ax1_kwargs)
			ax[1].scatter(binned_vis.uv/1e3, img, s=s, c=c, zorder=2, **ax1_kwargs)

			c = _kwargs1.pop('c', 'purple')
			ax[0].plot(model_grid/1e3, vis_model, ls=ls, lw=lw, zorder=3, c=c, **ax0_kwargs)

			lw = _kwargs1.pop('lw', 4)
			ls = _kwargs1.pop('ls', '-')
			ax2_kwargs.pop('lw', None)
			ax2_kwargs.pop('ls', None)
			ax2_kwargs.pop('c', None)
			ax[2].plot(sol.r, I_nn, lw=lw, ls=ls, c=c, zorder=2, **ax2_kwargs)
			return ax[0], ax[1], ax[2]
		else:
			lw = _kwargs0.pop('lw', 4)
			ls = _kwargs0.pop('ls', '-')	
			c = _kwargs0.pop('c', 'black')
			s = _kwargs0.pop('s', 150)

			ax0_kwargs.pop('lw', None)
			ax0_kwargs.pop('ls', None)
			ax0_kwargs.pop('c', None)
			ax0_kwargs.pop('s', None)

			ax[0].axhline(0, linewidth=lw, alpha=1, color="k", ls='--')	
			ax[0].errorbar(binned_vis.uv/1e3, real, yerr=real_err, ecolor='black', 
				fmt='none', capsize=0, zorder=1, elinewidth=3, **ax0_kwargs)
			ax[0].scatter(binned_vis.uv/1e3, real, s=s, c=c, zorder=2, **ax0_kwargs)

			c = _kwargs1.pop('c', 'purple')
			ax[0].plot(model_grid/1e3, vis_model, ls=ls, lw=lw, zorder=3, c=c, **ax0_kwargs)

			lw = _kwargs1.pop('lw', 4)
			ls = _kwargs1.pop('ls', '-')
			ax2_kwargs.pop('lw', None)
			ax2_kwargs.pop('ls', None)
			ax2_kwargs.pop('c', None)

			ax[1].plot(sol.r, I_nn, lw=lw, ls=ls, c=c, zorder=2, **ax2_kwargs)
			return ax[0], ax[1]

	def frank_param_explore(self,
		Rmax: float = 3,
		N: float = 250,
		dRA: float = 0.0,
		dDec: float = 0.0,
		inc: float = None,
		PA: float = None,
		alpha: ndarray = None, 		
		ws: ndarray = None, 
		bin_width: float = None, 
		est_weights: bool = False,
		normalise: bool = True,
		save_model: bool = False,
		ax: ndarray = None,
		ax0_kwargs={},
		ax1_kwargs={},
		kwargs={},
		):
		"""Calculates and plots Frankenstin models.

		Parameters
		----------
		R

		N

		dRA
			Right accession offsets in arcseconds.
		dDec
			Declination offsets in arcseconds.
		inc:
			Inclination of the disc in degrees.
		PA:
			Position angle of the disc in degrees.
		alpha

		ws

		bin_width
			Spacing for binnings the visibilities
		est_weights

		normalise

		save_model

		ax
			A matplotlib Axes handle.
		ax_kwargs
			Keyword arguments to pass to contour map matplotlib Axes.
		**kwargs
			Additional keyword arguments to pass to interpolation
			and matplotlib functions.

		Returns
		-------
		ax
			The matplotlib Axes object.
		"""
		
		ax0_kwargs = copy(ax0_kwargs)

		number_of_colors = len(alpha)*len(ws)

		color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
		for i in range(number_of_colors)]

		#setting up legend
		symbols = []
		labels = []
		c_counter = 0
		for a in range(0,len(alpha)):
			print('alpha = ', alpha[a])
			for w in range(0,len(ws)):
				print('wsmooth = ', ws[w])

				binned_vis, sol, model_grid = self.frank_model(Rmax=Rmax, N=N, dRA=dRA, dDec=dDec,
					inc=inc, PA=PA, alpha=alpha[a], ws=ws[w], bin_width=bin_width, est_weights=est_weights,
					save_model=save_model)

				#Solving for non-negative brightness profiles
				I_nn = sol.solve_non_negative()
				vis_model = sol.predict_deprojected(model_grid, I=I_nn).real
				vis_model_realbins = sol.predict_deprojected(binned_vis.uv, I=I_nn).real

				if normalise:
					real_data = binned_vis.V.real.data
					factor = real_data[~np.isnan(real_data)]
					factor = factor[factor != 0]
					print('kl0 value: ', factor[0])
					real = binned_vis.V.real/factor[0]
					real_err = binned_vis.error.real/factor[0]
					img = binned_vis.V.imag/factor[0]
					img_err = binned_vis.error.imag/factor[0]
					vis_model *= 1/factor[0]
					I_nn *= 1/max(np.abs(I_nn))
					self.kl0 = factor[0]
				else:
					real = binned_vis.V.real
					real_err = binned_vis.error.real
					img = binned_vis.V.imag
					img_err = binned_vis.error.imag

				log = np.sum((real - vis_model_realbins)**2/real_err)

				if not log:
					print('Model failed, increase Rmax or maybe N')
					log=1e3

				lw = ax0_kwargs.pop('lw', 4)
				ax[0].plot(model_grid/1e3, vis_model, ls='--', lw=lw, zorder=3, color=color[c_counter], alpha=0.5)
				ax[1].plot(sol.r, I_nn, zorder=2, color=color[c_counter], lw=lw, ls='--', alpha=0.5)


				symbols.append(mlines.Line2D([0], [0], color=color[c_counter], linewidth=lw, linestyle='-'))
				labels.append(r'$\alpha$ ='+str(alpha[a])+' ws ='+str(ws[w])+r' $\chi^2$ ='+str(float("{:.3f}".format(log))))
				
				c_counter = c_counter+1
		font = ax0_kwargs.pop('font', 20)
		s = ax0_kwargs.pop('s', 150)
		ax0_kwargs.pop('s', None)
		ax[0].axhline(0, linewidth=lw, alpha=1, color="k", ls='--')		
		ax[0].errorbar(binned_vis.uv/1e3, real, yerr=real_err, ecolor='black', 
			fmt='none', capsize=0, zorder=1, elinewidth=3, **ax0_kwargs)
		ax[0].scatter(binned_vis.uv/1e3, real, zorder=1, c='black', s=s, **ax0_kwargs)
		ax[1].legend(handles=symbols, labels=labels, loc='upper right', fontsize=font)
		ax[1].plot(**ax1_kwargs)

		return ax[0], ax[1]

	def frank_bootstrap(self,
		iters: int = 100,
		Rmax: float = 3.0,
		N: float = 300,
		dRA: float = 0.0,
		dDec: float = 0.0,
		inc: float = None,
		PA: float = None,
		alpha: ndarray = None, 		
		ws: ndarray = None, 
		bin_width: float = None, 
		est_weights: bool = False,
		save_model: bool = False,
		source: str = None,
		):
		"""Calculates Frankenstein models.

		Parameters
		----------
		R

		N

		dRA
			Right accession offsets in arcseconds.
		dDec
			Declination offsets in arcseconds.
		inc:
			Inclination of the disc in degrees.
		PA:
			Position angle of the disc in degrees.
		alpha

		ws

		bin_width
			Spacing for binnings the visibilities
		est_weights

		save_model


		Returns
		-------
		binned_vis

		sol

		model_grid
		"""

		#Calculating intitial model
		if est_weights == True:
			baselines = (self.u**2 + self.v**2)**.5
			weights = estimate_baseline_dependent_weight(baselines, self.vis, bin_width)
		else:
			weights = self.wgt

		disk_geometry = FixedGeometry(float(inc), float(PA), dRA=dRA, dDec=dDec)
		FF = FrankFitter(Rmax=Rmax, N=N, geometry=disk_geometry, alpha=alpha, weights_smooth=ws)

		print('Calculating initial Frankenstin model')
		sol = FF.fit(self.u, self.v, self.vis, weights)
		I_nn = sol.solve_non_negative()

		model_peak = sol.r[I_nn == I_nn.max()]

		#bootstrap
		peak_pos_boot = []
		bootstrap_vis_model = []
		bootstrap_bp_model = []
		bootstrap_vis_modelgrid = []
		bootstrap_bp_model_grid = []
		modelmean_diff = []



		print('Bootstrap model: ')
		for i in range(0,iters):
			print(i)

			u_boot, v_boot, vis_boot, wgt_boot = draw_bootstrap_sample(self.u, self.v, self.vis, weights)
			try:
				sol_boot = FF.fit(u_boot, v_boot, vis_boot, wgt_boot)
			except ValueError:
				continue

			#Deprojecting
			u_deproj, v_deproj, vis_deproj = sol_boot.geometry.apply_correction(u_boot, v_boot, vis_boot)
			baselines = (u_deproj**2 + v_deproj**2)**.5

			#Model grid
			model_grid_boot = np.logspace(np.log10(min(baselines.min(), sol_boot.q[0])), np.log10(max(baselines.max(), sol_boot.q[-1])), 10**4)

			I_nn_boot = sol_boot.solve_non_negative()

			vis_model_boot = sol_boot.predict_deprojected(model_grid_boot, I=I_nn_boot).real

			#Visibilities
			binned_vis = UVDataBinner(baselines, vis_deproj, wgt_boot, bin_width)


			peak_pos_boot.append(sol_boot.r[I_nn_boot == I_nn_boot.max()])

			bootstrap_bp_model.append(I_nn_boot)
			bootstrap_bp_model_grid.append(sol_boot.r)
			bootstrap_vis_model.append(vis_model_boot)
			bootstrap_vis_modelgrid.append(model_grid_boot)
			modelmean_diff.append(np.abs(I_nn_boot-I_nn))

		idx = np.argsort(np.mean(modelmean_diff, 1))
		bootstrap_bp_model = np.asarray(bootstrap_bp_model)
		bootstrap_bp_model = bootstrap_bp_model[idx]

		bootstrap_bp_model_grid = np.asarray(bootstrap_bp_model_grid)
		bootstrap_bp_model_grid = bootstrap_bp_model_grid[idx]

		bootstrap_vis_model = np.asarray(bootstrap_vis_model)
		bootstrap_vis_model = bootstrap_vis_model[idx]

		bootstrap_vis_modelgrid = np.asarray(bootstrap_vis_modelgrid)
		bootstrap_vis_modelgrid = bootstrap_vis_modelgrid[idx]

		if not os.path.isdir('bootstrap_models'):
			os.makedirs('bootstrap_models')

		if not source:
			print('WARNING bootstrap model is saved under a generic name and may be overwritten')
			source = 'frank'

		print(str(source))
		np.save('bootstrap_models/'+str(source)+'_bootstrap_bp_'+str(iters)+'_models', bootstrap_bp_model)
		np.save('bootstrap_models/'+str(source)+'_bootstrap_vis_'+str(iters)+'_models', bootstrap_vis_model)

		np.save('bootstrap_models/'+str(source)+'_bootstrap_bp_'+str(iters)+'_model_grid', bootstrap_bp_model_grid)
		np.save('bootstrap_models/'+str(source)+'_bootstrap_vis_'+str(iters)+'_model_grid', bootstrap_vis_modelgrid)

		model_peak_std = np.std(peak_pos_boot)
		string = str(source) + ' model peak: '+ str(model_peak) + ', STD: ' + str(model_peak_std)

		print(string)

		return string





























