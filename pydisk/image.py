"""Imaging tool for plotting fits image files

The Image class contains all tools necessary to read and plot fits image files.
"""

import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
import cmasher as cmr
from matplotlib.colors import TwoSlopeNorm

from scipy import interpolate
from scipy import constants

from pathlib import Path

from astropy.wcs import WCS
from astropy.visualization import (AsinhStretch, LogStretch, LinearStretch, ImageNormalize)

from copy import copy

from .utils import readfits, getdeg, Jybeam_to_Tb

class image:
	"""Image FITS object.

	Image FITS files contain a cleaned image outputted as 
	a FITS file from software such as CASA as well as a 
	detailed FITS header ... if the softwatre is any good.

	Examples
	--------
	Use plot to generate a passable axes for plotting.

	something something ... good example
	"""

	def __init__(self, filename=None, **kwargs):
		#Read FITS image file
		self.filename = filename
		self._read(**kwargs)

	def _read(self):
		"""Load image FITS file.

		Parameters
		----------
		filename
			The path to the file.
		"""
		# Set file_path
		file_path = Path(self.filename).expanduser()
		if not file_path.is_file():
			raise FileNotFoundError('Cannot find image FITS')
		
		#Read image FITS
		self.im, self.he = readfits(file_path)


	def contour_map(self,
		dRA: float = 0.0,
		dDec: float = 0.0,
		):
		"""Makes a contour map.

		Parameters
		----------
		dRA
			Right accession offsets in arcseconds.
		dDec
			Declination offsets in arcseconds.

		Returns
		-------
		im
			The contourmap.
		x
			The x component of the contourmap.
		y
			The y component of the contourmap.
		"""
		#reading in fits
		# self._read()

		im = self.im
		im[np.isnan(im)]=0.
		contmap=np.squeeze(im)

		he = self.he

		nx, ny = he['NAXIS1'], he['NAXIS2']

		xr = 3600 * he['CDELT1'] * (np.arange(nx) - (he['CRPIX1'] - 1)) - dRA
		yr = 3600 * he['CDELT2'] * (np.arange(ny) - (he['CRPIX2'] - 1)) - dDec

		if (xr.shape[0]!=contmap.shape[1]):
			xr = xr[0:contmap.shape[1]]
			print('xr array corrected')
		if (yr.shape[0]!=contmap.shape[0]):
			yr = yr[0:contmap.shape[0]]
			print('yr array corrected')

		x, y = np.meshgrid(xr,yr)

		rr = np.sqrt(x**2+y**2)

		#Reshaping if x!=y
		if (rr.shape!=contmap.shape):
			rr = rr[0:contmap.shape[0],0:contmap.shape[1]]

		#Ideally the map is larger than the source ... this may be bad
		radius = 0.5*rr.max()

		w = np.where(rr>radius)
		rms = np.std(contmap[w])

		return contmap, x, y, rms

	def contour_map_bad_fits(self,
		dRA: float = 0.0,
		dDec: float = 0.0,
		pixel_size: float = None,
		):
		"""Makes a contour map.

		Parameters
		----------
		dRA
			Right accession offsets in arcseconds.
		dDec
			Declination offsets in arcseconds.
		pixel_size
			Pixel scale given in arcsecond/pixel.

		Returns
		-------
		im
			The contourmap.
		x
			The x component of the contourmap.
		y
			The y component of the contourmap.
		"""
		#reading in fits
		# self._read()

		im = self.im
		im[np.isnan(im)]=0.
		contmap=np.squeeze(im)

		he = self.he

		nx, ny = he['NAXIS1'], he['NAXIS2']

		xr = pixel_size * (np.arange(nx) - ((nx/2) - 1)) - dRA
		yr = pixel_size * (np.arange(ny) - ((ny/2) - 1)) - dDec

		if (xr.shape[0]!=contmap.shape[1]):
			xr = xr[0:contmap.shape[1]]
			print('xr array corrected')
		if (yr.shape[0]!=contmap.shape[0]):
			yr = yr[0:contmap.shape[0]]
			print('yr array corrected')

		x, y = np.meshgrid(-xr,yr)


		return contmap, x, y

	def radial_profile(self,
		inc: float = None,
		PA: float = None,
		dRA: float = 0.0,
		dDec: float = 0.0,
		rbins: ndarray = None,
		tbins: ndarray = None,
		rotate_map: float = None,
		):
		"""Calculates the radial profile of a contour map.

		Parameters
		----------
		inc
			Inclination of the disc.
		PA
			Position angle of the disc.
		dRA
			Right accession offsets in arcseconds.
		dDec
			Declination offsets in arcseconds.
		rbins
			The radial points at which to calculate an annulus as a numpy array.
		tbins
			The azimuthal points at which to calculate the profile as a numpy array.

		Returns
		-------
		rbins
			Radial bins of the profile.
		SBr
			Brightness of the contour map as Jy/area.
		err_SBr
			Standard deviation of each averaged annulus.
		rtmap
			Averaged contour map.

		"""
		#reading in fits

		image, x, y, rms = self.contour_map(dRA, dDec)

		# convert these to radius
		incr, PAr = np.radians(inc), np.radians(PA)

		# deproject and rotate to new coordinate frame
		xp = (x * np.cos(PAr) - y * np.sin(PAr)) / np.cos(incr)
		yp = (x * np.sin(PAr) + y * np.cos(PAr))

		if rotate_map:
			ang = np.radians(rotate_map)
			x_rot = xp
			y_rot = yp
			xp =  x_rot* np.cos(ang) - y_rot * np.sin(ang)
			yp =  x_rot * np.sin(ang) + y_rot * np.cos(ang)

		# now convert to polar coordinates (r in arcseconds, theta in degrees)
		# note that theta starts along the minor axis (theta = 0), and rotates clockwise in the sky plane)
		r = np.sqrt(xp**2 + yp**2)
		theta = np.degrees(np.arctan2(yp, xp))

		# radius and azimuth bin centers (and their widths)
		if rbins is None:
			rbins = np.linspace(0.0005, 1.5, 3000)	# in arcseconds
		if tbins is None:
			tbins = np.linspace(-180, 180, 361)     # in degrees

		dr = np.abs(rbins[1] - rbins[0])
		dt = np.abs(tbins[1] - tbins[0])

		# initialize the (r, az)-map and radial profile
		rtmap = np.empty((len(tbins), len(rbins)))
		SBr, err_SBr = np.empty(len(rbins)), np.empty(len(rbins))

		# loop through the bins to populate the (r, az)-map and radial profile
		for i in range(len(rbins)):
			# identify pixels that correspond to the radial bin (i.e., in this annulus)
			in_annulus = ((r >= (rbins[i] - 0.5 * dr)) & (r < (rbins[i] + 0.5 * dr)))

			# accumulate the azimuth values and surface brightness values in this annulus
			az_annulus = theta[in_annulus]
			SB_annulus = image[in_annulus]

			# average the intensities (and their scatter) in the annulus
			SBr[i], err_SBr[i] = np.average(SB_annulus), np.std(SB_annulus)

			SBr[i] = np.average(SB_annulus)
			err_SBr[i] = np.std(SB_annulus)
			# populate the azimuthal bins for the (r, az)-map at this radius
			for j in range(len(tbins)):
				# identify pixels that correspond to the azimuthal bin
				in_wedge = ((az_annulus >= (tbins[j] - 0.5 * dt)) & (az_annulus < (tbins[j] + 0.5 * dt)))

				# if there are pixels in that bin, average the corresponding intensities
				if (len(SB_annulus[in_wedge]) > 0):
					rtmap[j,i] = np.average(SB_annulus[in_wedge])
				else:
					rtmap[j,i] = -1e10 #place 
		#custom counter for bad columns
		j = 0
		#fixing placeholder
		for i in range(len(rbins)):
			#j=i
			# extract an azimuthal slice of the (r, az)-map
			az_slice = rtmap[:,j]
			# identify if there's missing information in an az bin along that slice:
			# if so, fill it in with linear interpolation along the slice
			if np.any(az_slice < -1e5):
				# extract non-problematic bins in the slice
				x_slice, y_slice = tbins[az_slice >= -1e5], az_slice[az_slice >= -1e5]
				# if np.array(x_slice).any() and np.array(y_slice).any():

				# pad the arrays to make sure they span a full circle in azimuth
				x_slice_ext = np.pad(x_slice, 1, mode='constant')
				x_slice_ext[0] -= 360.
				x_slice_ext[-1] += 360.
				y_slice_ext = np.pad(y_slice, 1, mode='constant')

				# define the interpolation function
				raz_func = interpolate.interp1d(x_slice_ext, y_slice_ext, bounds_error=True)

				# interpolate and replace those bins in the (r, az)-map
				fixed_slice = raz_func(tbins)
				rtmap[:,j] = fixed_slice
				j = j+1
				# else:
				# 	rtmap = np.delete(rtmap, i, axis=1)
				# 	rbins = np.delete(rbins, i)
				# 	j = j-1


		return rbins, tbins, SBr, err_SBr, rtmap

	def plot_map(self,
		plot_star: bool = False,
		plot_beam: bool = True,
		contour_overlay: str = None,
		overlay_dRA: float = 0.0,
		overlay_dDec: float = 0.0,
		contours: ndarray = None,
		contour_overlay2: str = None,
		overlay2_dRA: float = 0.0,
		overlay2_dDec: float = 0.0,
		contours2: ndarray = None,
		dRA: float = 0.0,
		dDec: float = 0.0,
		custom_rms: float = None,
		ax: ndarray = None,
		int_flux = True,
		Tb: bool = False,
		map_kwargs={},
		contour_kwargs={},
		contour_kwargs2={},
		colorbar_kwargs={},
		star_kwargs={},
		beam_kwargs={},
		kwargs={},
		):
		"""Plot image as a contour map.

		Parameters
		----------
		contours
			Levels at which contour lines should be plotted 
			on top of the continuum map.
		dRA
			Right accession offsets in arcseconds.
		dDec
			Declination offsets in arcseconds.
		ax
			A matplotlib Axes handle.
		map_kwargs
			Keyword arguments to pass to contour map matplotlib Axes.
		contour_kwargs
			Keyword arguments to pass to contour line matplotlib Axes.
		colorbar_kwargs
			Keyword arguments to pass to matplotlib Colorbar.
		star_kwargs
			Keyword arguments to pass to star symbol.
		beam_kwargs
			Keyword arguments to pass to beam symbol.
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

		Other Parameters
		----------------
		show_colorbar : bool
			Whether or no to display a colorbar. efault is True.

		Examples
		--------
		Put good example here.
		"""
		__kwargs = copy(kwargs)

		show_colorbar = __kwargs.pop('show_colorbar', 'True')

		if ax is None:
			fig, ax = plt.subplots()
		else:
			fig = ax.figure

		#produce contour map from fits file
		contmap, x, y, rms = self.contour_map(dRA, dDec)

		he = self.he

		if int_flux:
			obj = he['OBJECT']
			freq = he['CRVAL3']/1e9
			rr = np.sqrt(x**2+y**2)
			w=np.where(contmap>3*rms)			
			beampix=he['BMAJ']*he['BMIN']/(abs(he['CDELT1'])**2)*np.pi/(4*np.log(2.))
			flux=np.nansum(contmap[w])/beampix
			print('Source: ', obj)
			print('Intergrated flux: ', flux)
			print('rms: ', rms)
			print('Freq: ', freq,'GHz - ',constants.c/(freq*1e6), 'mm')

		_kwargs = copy(map_kwargs)
		cmap = _kwargs.pop('cmap', cmr.heat)
		levels = _kwargs.pop('levels', 1000)

		if Tb:
			print('Converting scale to temperature')
			contmap = Jybeam_to_Tb(Fnu=contmap, nu=he['CRVAL3'], bmaj=he['BMAJ']/0.000277778, bmin=he['BMIN']/0.000277778)
			contf = ax.contourf(x, y, contmap, levels=levels, cmap=cmap, zorder= 1, **_kwargs)
		else:
			contf = ax.contourf(x, y, contmap*10**3, levels=levels, cmap=cmap, zorder= 1, **_kwargs)

		if contour_overlay:
			im, he = readfits(contour_overlay)
			im[np.isnan(im)]=0.
			contmap=np.squeeze(im)

			nx, ny = he['NAXIS1'], he['NAXIS2']

			xr = 3600 * he['CDELT1'] * (np.arange(nx) - (he['CRPIX1'] - 1)) - overlay_dRA
			yr = 3600 * he['CDELT2'] * (np.arange(ny) - (he['CRPIX2'] - 1)) - overlay_dDec

			if (xr.shape[0]!=contmap.shape[1]):
				xr = xr[0:contmap.shape[1]]
				print('xr array corrected')
			if (yr.shape[0]!=contmap.shape[0]):
				yr = yr[0:contmap.shape[0]]
				print('yr array corrected')

			x, y = np.meshgrid(xr,yr)

			rr = np.sqrt(x**2+y**2)

			#Reshaping if x!=y
			if (rr.shape!=contmap.shape):
				rr = rr[0:contmap.shape[0],0:contmap.shape[1]]

			#Ideally the map is larger than the source ... this may be bad
			radius = 0.5*rr.max()

			w = np.where(rr>radius)
			rms = np.std(contmap[w])

			_kwargs = copy(contour_kwargs)
			alpha = _kwargs.pop('alpha', 1)
			color = _kwargs.pop('color', 'w')
			lw = _kwargs.pop('linewidth', 1)
			ls = _kwargs.pop('linestyle', '-')

			contour_array = np.array(contours)
			contours_neg = -1*contour_array
			if custom_rms:
				rms_map = float(custom_rms)
			else:
				rms_map = rms
			levels = np.sort(np.append(contours_neg,contour_array))*rms
			ax.contour(x,y,contmap,levels, alpha=alpha, linewidths=lw, 
				colors=color, linestyles=ls, zorder=2)

			if contour_overlay2:

				im, he = readfits(contour_overlay2)
				im[np.isnan(im)]=0.
				contmap=np.squeeze(im)

				nx, ny = he['NAXIS1'], he['NAXIS2']

				xr = 3600 * he['CDELT1'] * (np.arange(nx) - (he['CRPIX1'] - 1)) - overlay2_dRA
				yr = 3600 * he['CDELT2'] * (np.arange(ny) - (he['CRPIX2'] - 1)) - overlay2_dDec

				if (xr.shape[0]!=contmap.shape[1]):
					xr = xr[0:contmap.shape[1]]
					print('xr array corrected')
				if (yr.shape[0]!=contmap.shape[0]):
					yr = yr[0:contmap.shape[0]]
					print('yr array corrected')

				x, y = np.meshgrid(xr,yr)

				rr = np.sqrt(x**2+y**2)

				#Reshaping if x!=y
				if (rr.shape!=contmap.shape):
					rr = rr[0:contmap.shape[0],0:contmap.shape[1]]

				#Ideally the map is larger than the source ... this may be bad
				radius = 0.5*rr.max()

				w = np.where(rr>radius)
				rms = np.std(contmap[w])

				_kwargs = copy(contour_kwargs2)
				alpha = _kwargs.pop('alpha', 1)
				color = _kwargs.pop('color', 'w')
				lw = _kwargs.pop('linewidth', 1)
				ls = _kwargs.pop('linestyle', '-')

				contour_array = np.array(contours2)
				contours_neg = -1*contour_array
				levels = np.sort(np.append(contours_neg,contour_array))*rms
				ax.contour(x,y,contmap,levels, alpha=alpha, linewidths=lw, 
					colors=color, linestyles=ls, zorder=2)


		else:
			if np.array(contours).any():
				_kwargs = copy(contour_kwargs)
				alpha = _kwargs.pop('alpha', 1)
				color = _kwargs.pop('color', 'w')
				lw = _kwargs.pop('linewidth', 1)

				contour_array = np.array(contours)
				contours_neg = -1*contour_array
				levels = np.sort(np.append(contours_neg,contour_array))
				ax.contour(x,y,contmap,levels, alpha=alpha, linewidths=lw, 
					colors=color, zorder=2)

		#colorbar
		if show_colorbar:
			_kwargs = copy(colorbar_kwargs)
			if Tb:
				label = _kwargs.pop('label', r'$T_b (K)$')
			else:
				label = _kwargs.pop('label', 'mJy/beam')
			
			position = _kwargs.pop('position', 'right')
			size = _kwargs.pop('size', '5%')
			pad = _kwargs.pop('pad', '2%')
			if position in ('top', 'bottom'):
				_kwargs.update({'orientation': 'horizontal'})
				pad = _kwargs.pop('pad', '6%')

			divider = make_axes_locatable(ax)
			cax = divider.append_axes(position=position, size=size, pad=pad)
			cbar = plt.colorbar(contf, cax, **_kwargs)
			if position in ('top', 'bottom'):
				cbar.ax.tick_params(axis='x', top=True, bottom=False, 
					labelbottom=False, labeltop=True)
				cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
			else:
				cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

			fontsize = __kwargs.pop('cbar_fontsize', '12')
			fontweight = __kwargs.pop('cbar_fontweight', 'bold')
			cbar.set_label(label, fontsize=fontsize, fontweight=fontweight)
			font_size = __kwargs.pop('tick_labelsize', '12')
			cbar.ax.tick_params(labelsize=font_size)
			no_ticks = __kwargs.pop('tick_number', '5')

			tick_locator = ticker.MaxNLocator(nbins=no_ticks)
			cbar.locator = tick_locator
			cbar.update_ticks()

		if plot_star:
			_kwargs = copy(star_kwargs)
			size = _kwargs.pop('size', '120')
			alpha = _kwargs.pop('alpha', '0.7')
			color = _kwargs.pop('color', 'w')
			ecolor = _kwargs.pop('edgecolor', 'k')
			marker = _kwargs.pop('marker', '*')
			ax.scatter(0,0, s=size, alpha=alpha, c=color, 
				edgecolor=ecolor, marker=marker, zorder=3, 
				rasterized=True)

		if plot_beam:
			bmpa=90.-he['BPA']
			bmj = he['BMAJ']
			bmn = he['BMIN']
			print('Beam dim: bmj=',bmj/0.000277778,'as, bmn=',bmn/0.000277778,'as, pa=', he['BPA'],' deg')
			_kwargs = copy(beam_kwargs)
			lw = _kwargs.pop('linewidth', '2')
			clr = _kwargs.pop('edgecolor', 'w')
			beam_pos = _kwargs.pop('beam_pos', '2')

			ell=Ellipse(xy=(0.7*float(beam_pos),-0.7*float(beam_pos)), 
				width=bmj*3600., height=bmn*3600., angle=bmpa, 
				edgecolor=clr, hatch='\\\\\\', fc='None',linewidth=lw, 
				zorder=5)
			ell1=Ellipse(xy=(0.7*float(beam_pos),-0.7*float(beam_pos)), 
				width=bmj*3600., height=bmn*3600., angle=bmpa, 
				edgecolor=clr, hatch='////', fc='None',linewidth=lw, 
				zorder=5)
			ax.add_patch(ell)
			ax.add_patch(ell1)

		return ax

	def plot_1Dradialprofile(self,
		plot_beam: bool = True,
		inc: float = None,
		PA: float = None,
		dRA: float = 0.0,
		dDec: float = 0.0,
		sr: bool = False,
		rbins: ndarray = None,
		tbins: ndarray = None,
		ax: ndarray = None,
		ax_kwargs={},
		beam_kwargs={},
		kwargs={},
		):
		"""Plot image as a contour map.

		Parameters
		----------
		contours
			Levels at which contour lines should be plotted 
			on top of the continuum map.
		inc
			Inclination of the disc in degrees.
		PA
			Position angle of the disc in degrees.
		dRA
			Right accession offsets in arcseconds.
		dDec
			Declination offsets in arcseconds.
		rbins
			The radial points at which to calculate an annulus as a numpy array.
		tbins
			The azimuthal points at which to calculate the profile as a numpy array.
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


		#produce contour map from fits file
		rbins, tbins, SBr, err_SBr, rtmap = self.radial_profile(inc, PA, dRA, dDec, rbins, tbins)

		# SBr = SBr*4.25e10
		bmj = 3600 *self.he['BMAJ']
		bmn = 3600 *self.he['BMIN']

		beam_area = (np.pi * bmj * bmn/ (4 * np.log(2))) / (3600 * 180 / np.pi)**2
		SBr_0 = SBr / (beam_area * (3600 * 180 / np.pi)**2)
		err_SBr_0 = err_SBr / (beam_area * (3600 * 180 / np.pi)**2)

		rbins = rbins[SBr_0>0]
		err_SBr_0 = err_SBr_0[SBr_0>0]
		SBr_0 = SBr_0[SBr_0>0]
		if sr:
			SBr_0 = SBr_0*4.25e10
			err_SBr_0 = err_SBr_0*4.25e10

		ax.scatter(rbins, SBr_0, **ax_kwargs)
		ax.errorbar(rbins, SBr_0, yerr=err_SBr_0, ecolor='black', 
			fmt='none', capsize=0, zorder=1, elinewidth=1)

		#plt.fill_between(rbins, SBr_0-err_SBr_0, SBr_0+err_SBr_0, alpha=0.5, **ax_kwargs)

		if plot_beam:
			bmpa=90.-self.he['BPA']
			bmj = self.he['BMAJ']
			bmn = self.he['BMIN']
			_kwargs = copy(beam_kwargs)
			lw = _kwargs.pop('linewidth', '2')
			clr = _kwargs.pop('edgecolor', 'w')
			beam_pos = _kwargs.pop('beam_pos', '2')

			ell=Ellipse(xy=(0.7*float(beam_pos),-0.7*float(beam_pos)), 
				width=bmj*3600., height=bmn*3600., angle=bmpa, 
				edgecolor=clr, hatch='\\\\\\', fc='None',linewidth=lw, 
				zorder=5)
			ell1=Ellipse(xy=(0.7*float(beam_pos),-0.7*float(beam_pos)), 
				width=bmj*3600., height=bmn*3600., angle=bmpa, 
				edgecolor=clr, hatch='////', fc='None',linewidth=lw, 
				zorder=5)
			ax.add_patch(ell)
			ax.add_patch(ell1)

		return ax

	def plot_rphi_map(self,
		contours: ndarray = None,
		inc: float = None,
		PA: float = None,
		dRA: float = 0.0,
		dDec: float = 0.0,
		ax: ndarray = None,
		Tb: bool = False,
		asinh_scale: bool= False,
		log_scale: bool= False,
		rbins: ndarray = None,
		tbins: ndarray = None,
		sr: bool = False,
		radial_xaxis: bool = True,
		rotate_map: float = None,
		map_kwargs={},
		contour_kwargs={},
		colorbar_kwargs={},
		beam_kwargs={},
		norm_kwargs ={},
		kwargs={},
		):

		if ax is None:
			fig, ax = plt.subplots()
		else:
			fig = ax.figure

		__kwargs = copy(kwargs)

		_kwargs = copy(map_kwargs)
		cmap = _kwargs.pop('cmap', cmr.heat)
		levels = _kwargs.pop('levels', 1000)

		show_colorbar = __kwargs.pop('show_colorbar', False)

		rbins, tbins, SBr, err_SBr, rtmap = self.radial_profile(inc, PA, dRA, dDec, rbins, tbins, rotate_map)

		rtmap_bounds = (rbins.min(), rbins.max(), tbins.min(), tbins.max())

		vmin = _kwargs.pop('vmin', rtmap.min())

		vmax = _kwargs.pop('vmax', rtmap.max())

		if radial_xaxis:
			x=rbins
			y=tbins
		else:
			x=tbins
			y=rbins
			rtmap = np.transpose(rtmap)

		if Tb:
			print('Converting scale to temperature')
			rtmap = Jybeam_to_Tb(Fnu=rtmap, nu=self.he['CRVAL3'], bmaj=self.he['BMAJ']/0.000277778, bmin=self.he['BMIN']/0.000277778)
			contf = ax.contourf(x, y, rtmap, cmap=cmap, levels=levels, aspect='auto', **_kwargs)
		else:
			contf = ax.contourf(x, y, 1e3*rtmap, cmap=cmap, levels=levels, aspect='auto', **_kwargs)


		

		if contours:
			_kwargs = copy(contour_kwargs)
			alpha = _kwargs.pop('alpha', 0.7)
			color = _kwargs.pop('color', 'w')
			lw = _kwargs.pop('linewidth', 1)

			contour_array = np.array(contours)
			contours_neg = -1*contour_array
			levels = np.sort(np.append(contours_neg,contour_array))
			ax.contour(x,y,rtmap,levels, alpha=alpha, linewidths=lw, 
				colors=color, zorder=2, **_kwargs)

		# #colorbar
		# if show_colorbar:
		# 	cb = plt.colorbar(im, ax=ax, pad=0.05)
		# 	cb.set_label('surface brightness [mJy / beam]', rotation=270, labelpad=17)

		if show_colorbar:
			_kwargs = copy(colorbar_kwargs)
			if Tb:
				label = _kwargs.pop('label', r'$T_K$')
			else:
				label = _kwargs.pop('label', 'mJy/beam')
			
			position = _kwargs.pop('position', 'right')
			size = _kwargs.pop('size', '5%')
			pad = _kwargs.pop('pad', '2%')
			aspect = _kwargs.pop('aspect', 12)
			if position in ('top', 'bottom'):
				_kwargs.update({'orientation': 'horizontal'})
				pad = _kwargs.pop('pad', '6%')

			divider = make_axes_locatable(ax)
			cax = divider.append_axes(position=position, size=size, pad=pad, aspect=aspect)
			cbar = plt.colorbar(contf, cax, **_kwargs)
			if position in ('top', 'bottom'):
				cbar.ax.tick_params(axis='x', top=True, bottom=False, 
					labelbottom=False, labeltop=True)
				cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
			else:
				cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

			fontsize = __kwargs.pop('cbar_fontsize', '12')
			fontweight = __kwargs.pop('cbar_fontweight', 'bold')
			rotation = __kwargs.pop('rotation', 270)
			labelpad = __kwargs.pop('labelpad', 5)
			cbar.set_label(label, fontsize=fontsize, fontweight=fontweight, rotation=rotation, labelpad=labelpad)
			font_size = __kwargs.pop('tick_labelsize', '12')
			cbar.ax.tick_params(labelsize=font_size)
			no_ticks = __kwargs.pop('tick_number', '5')

			tick_locator = ticker.MaxNLocator(nbins=no_ticks)
			cbar.locator = tick_locator
			cbar.update_ticks()

		# if plot_beam:
		# 	bmpa=90.-he['BPA']
		# 	bmj = he['BMAJ']
		# 	bmn = he['BMIN']
		# 	_kwargs = copy(beam_kwargs)
		# 	lw = _kwargs.pop('linewidth', '2')
		# 	clr = _kwargs.pop('edgecolor', 'w')
		# 	beam_pos = _kwargs.pop('beam_pos', '2')

		# 	ell=Ellipse(xy=(0.7*float(beam_pos),-0.7*float(beam_pos)), 
		# 		width=bmj*3600., height=bmn*3600., angle=bmpa, 
		# 		edgecolor=clr, hatch='\\\\\\', fc='None',linewidth=lw, 
		# 		zorder=5)
		# 	ell1=Ellipse(xy=(0.7*float(beam_pos),-0.7*float(beam_pos)), 
		# 		width=bmj*3600., height=bmn*3600., angle=bmpa, 
		# 		edgecolor=clr, hatch='////', fc='None',linewidth=lw, 
		# 		zorder=5)
		# 	ax.add_patch(ell)
		# 	ax.add_patch(ell1)







