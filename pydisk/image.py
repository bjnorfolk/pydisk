"""Imaging tool for plotting fits image files

The Image class contains all tools necessary to read and plot fits image files.
"""

import numpy as np

from .utils import readfits, getdeg

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

	def __init__(self):
		#Read FITS image file
		self._read()

	def _read(self):
		"""Load image FITS file.

		Parameters
		----------
		filename
			The path to the file.
		"""
		# Set file_path
		file_path = Path(filename).expanduser()
		if not file_path.is_file():
			raise FileNotFoundError('Cannot fine image FITS')
		
		#Read image FITS
		self.im, self.he = readfits(file_path)

	def _contour_map(self,
		stra: str = None,
		stdec: str = None,
		):
		"""Makes contour map.

		Parameters
		----------
		stra
			Right accession of the source in hms.
			(If not specified in the FITS header)
		stdec
			Declination of the source in dms.
			(If not specified in the FITS header)

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
		_read(self)

		im = self.im
		im[np.isnan(im)]=0.
		im=np.squeeze(im)


		he = self.he

		if not he['CRVAL1'] and he['CRVAL2']:
			print('Converting RA and DEC to degrees')
			print('Hopefully you gaia corrected')

			targetra_ang, targetdec_ang = getdeg(stra, stdec)
		else:
			targetra_ang = he['CRVAL1']
			targetdec_ang = he['CRVAL2']

		xsize = he['NAXIS1']
		ysize = he['NAXIS2']
		x=np.arange(0,xsize,1)
		y=np.arange(0,ysize,1)
		import warnings
		warnings.filterwarnings("ignore")
		w=WCS(filename)
		warnings.resetwarnings()

		try: #should work in most cases
			try:
				lon,lat,p1,p2=w.wcs_pix2world(x,y,1,1,1)
			except ValueError: #other dimensions in data
				lon,lat=w.wcs_pix2world(x,y,1)
			if(lon[1]<0.):
				lon=np.array(lon)+360.
			xr=(lon-targetra_ang)*3600.*np.cos(targetdec_ang*DEGTORAD)
			yr=(lat-targetdec_ang)*3600.
		except TypeError:
			xpscale=he['CDELT1']*3600.
			ypscale=he['CDELT2']*3600.
			cxpx=he['CRPIX1']-1
			cypx=he['CRPIX2']-1
			cxval=he['CRVAL1']
			cyval=he['CRVAL2']
			xr=np.arange(-1*xpscale*cxpx,xpscale*(xsize-cxpx),xpscale)+(cxval-targetra_ang)*3600.
			yr=np.arange(-1*ypscale*cypx,ypscale*(ysize-cypx),ypscale)+(cyval-targetdec_ang)*3600.

		if (xr.shape!=im.shape[1]):
			xr = xr[0:im.shape[1]]
			print('xr array corrected')
		if (yr.shape!=im.shape[0]):
			yr = yr[0:im.shape[0]]
			print('yr array corrected')

	return im, x, y

	def plot(self,
		contours: ndarray = None,
		stra: str = None,
		stdec: str = None,
		shift_map: ndarray = None,
		plot_star: bool = False,
		plot_beam: bool = True,
		ax: Any,
		map_kwargs={},
		contour_kwargs={},
		colorbar_kwargs={},
		star_kwargs={},
		beam_kwargs={},
		**kwargs,
		) -> Any:
		"""Plot image as a contour map.

		Parameters
		----------
		contours
			Levels at which contour lines should be plotted 
			on top of the continuum map.
		stra
			Right accession of the source in hms.
			(If not specified in the FITS header)
		stdec
			Declination of the source in dms.
			(If not specified in the FITS header)
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

		show_colorbar = kwargs.pop('show_colorbar', kind == 'image')

		if ax is None:
			fig, ax = plt.subplots()
		else:
			fig = ax.figure

		#read fits file in
		_contour_map(self)

		ax.contourf(x, y, im, rasterized=True, **map_kwargs)

		if contours:
			_kwargs = copy(contour_kwargs)
			alpha = _kwargs.pop('alpha', '0.7')
			color = _kwargs.pop('color', 'w')
			lw = _kwargs.pop('linewidth', '2')
			ax.contour(x,y,im,contours, alpha=alpha, linewidths=lw, 
				colors=color, zorder=2, rasterized=True)

		#colorbar
		if show_colorbar:
			divider = make_axes_locatable(ax)
			_kwargs = copy(colorbar_kwargs)
			position = _kwargs.pop('position', 'right')
			size = _kwargs.pop('size', '5%')
			pad = _kwargs.pop('pad', '2%')
			if position in ('top', 'bottom'):
				_kwargs.update({'orientation': 'horizontal'})
			cax = divider.append_axes(position=position, size=size, pad=pad)
			cbar = fig.colorbar(im*10**3, cax, **_kwargs)

			label = _kwargs('label', 'mJy/beam')
			cbar.set_label(label)

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
			sz = np.array(ax.get_xlim())[0]
			_kwargs = copy(beam_kwargs)
			lw = _kwargs.pop('linewidth', '2')
			clr = _kwargs.pop('edgecolor', 'k')

			ell=Ellipse(xy=(0.7*sz,-0.7*sz), width=bmj*3600.,
				height=bmn*3600., angle=bmpa, edgecolor=clr, 
				hatch='\\\\\\', fc='None',linewidth=lw)
			ell1=Ellipse(xy=(0.7*sz,-0.7*sz), width=bmj*3600.,
				height=bmn*3600., angle=bmpa, edgecolor=clr, 
				hatch='////', fc='None',linewidth=lw)
			ax.add_patch(ell)
			ax.add_patch(ell1)











