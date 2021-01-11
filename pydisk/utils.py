import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.io import ascii

from frank.utilities import UVDataBinner

default_cmap = "inferno"

sigma_to_FWHM = 2.0 * np.sqrt(2.0 * np.log(2))
FWHM_to_sigma = 1.0 / sigma_to_FWHM
arcsec = np.pi / 648000
c = 299792458

def Wm2_to_Jy(nuFnu, nu):
    '''
    Convert from W.m-2 to Jy
    nu [Hz]
    '''
    return 1e26 * nuFnu / nu

def Jy_to_Wm2(Fnu, nu):
    '''
    Convert from Jy to W.m-2
    nu [Hz]
    '''
    return 1e-26 * Fnu * nu

def Jybeam_to_Tb(Fnu, nu, bmaj, bmin):
    '''
     Convert Flux density in Jy/beam to brightness temperature [K]
     Flux [Jy]
     nu [Hz]
     bmaj, bmin in [arcsec]
     T [K]
    '''
    beam_area = bmin * bmaj * arcsec ** 2 * np.pi / (4.0 * np.log(2.0))
    exp_m1 = 1e26 * beam_area * 2.0 * sc.h / sc.c ** 2 * nu ** 3 / Fnu
    hnu_kT = np.log1p(np.maximum(exp_m1, 1e-10))

    Tb = sc.h * nu / (hnu_kT * sc.k)

    return Tb

def Jy_to_Tb(Fnu, nu, pixelscale):
    '''
     Convert Flux density in Jy/pixel to brightness temperature [K]
     Flux [Jy]
     nu [Hz]
     bmaj, bmin in [arcsec]
     T [K]
    '''
    pixel_area = (pixelscale * arcsec) ** 2
    exp_m1 = 1e16 * pixel_area * 2.0 * sc.h / sc.c ** 2 * nu ** 3 / Fnu
    hnu_kT = np.log1p(exp_m1 + 1e-10)

    Tb = sc.h * nu / (hnu_kT * sc.k)

    return Tb

def Wm2_to_Tb(nuFnu, nu, pixelscale):
    """Convert flux converted from Wm2/pixel to K using full Planck law.
        Convert Flux density in Jy/beam to brightness temperature [K]
        Flux [W.m-2/pixel]
        nu [Hz]
        bmaj, bmin, pixelscale in [arcsec]
        """
    pixel_area = (pixelscale * arcsec) ** 2
    exp_m1 = pixel_area * 2.0 * sc.h * nu ** 4 / (sc.c ** 2 * nuFnu)
    hnu_kT = np.log1p(exp_m1)

    Tb = sc.h * nu / (sc.k * hnu_kT)

    return Tb

def uvdump2ascii(vis, out):
	uvdump_file = Table.read(vis, format='ascii')
	u = uvdump_file['col1']
	v = uvdump_file['col2']
	w = uvdump_file['col3']
	real = uvdump_file['col4']
	imag = uvdump_file['col5']
	wgt = 1/uvdump_file['col6']
	freq = uvdump_file['col7']

	uv_data = Table([u, v, w, real, imag, wgt, freq], 
		names=['u', 'v', 'w', 'real', 'imag', 'wgt', 'freq'])

	if not out:
		raise FileNotFoundError('Need to specify an output file')

	ascii.write(uv_data, out, overwrite=True)

	return uv_data

def vis_shift_min(filename, Rmax=2, dtheta=0.01, xlim=None):
	'''
	Chi squared minimization to find the dRA and dDec offsets
	N [arcseconds], physical area in the sky to iterate over
	dtheta [arcsec], resolution
	xlim [klambda], restricts the maximum baseline in the minimization
	'''
	#reading in vis file
	u, v, vis, wgt = readvis(filename)
	real=vis.real
	imag=vis.imag
	amp = np.sqrt(real**2 + imag**2)
	phase = np.arctan2(imag,real)

	#Masking max baseline
	if xlim:
		uvdist = np.sqrt(u**2+v**2)
		u=u[(uvdist/1e3)<xlim]
		v=v[(uvdist/1e3)<xlim]
		real=real[(uvdist/1e3)<xlim]
		imag=imag[(uvdist/1e3)<xlim]
		amp = np.sqrt(real**2 + imag**2)
		phase = np.arctan2(imag,real)

	#Constructing parameter space
	shift_arcsec = (np.arange(Rmax*1e2) - (Rmax*1e2)/2) * dtheta
	shift_rad =  (np.pi/180.)*(shift_arcsec/3600.)

	img_scatter=np.zeros((int(Rmax*1e2),int(Rmax*1e2)))

	for i in range(0,len(shift_rad)):
		dra = shift_rad[i]
		for j in range(0,len(shift_rad)):
			ddec = shift_rad[j]
			temp_img = amp*np.sin(phase + 2.*np.pi*(u*dra+v*ddec))
			img_scatter[i,j] = np.std(temp_img)

	min_err = img_scatter.min()
	minloc = np.where( img_scatter == min_err )

	#Finding best shift
	dra_best = -shift_arcsec[minloc[0][0]]
	ddec_best = -shift_arcsec[minloc[1][0]]

	print('RA shift:',dra_best)
	print('DEC shift:',ddec_best)

	#returns imaginary scatter plot
	return(img_scatter)

def readfits(filename):
	'''
	Reads image FITS data and header.
	'''
	#Reading FITS data and header
	im, he = fits.getdata(filename, header=True)

	return im,he

def readvis(filename, wle=None):
	'''
	Reads visibility data.
	'''
	if not filename.endswith(('.txt','.npz','.dat')):
		raise ValueError('Data must be either in a ascii txt file or numpy npz file')

	if filename.endswith('.txt'):
		if wle:
			uv_data = Table.read(filename, format='ascii')
			u=uv_data['col1']/wle
			v=uv_data['col2']/wle
			#w=uv_data['w']
			wgt=uv_data['col5']
			real=uv_data['col3']
			imag=uv_data['col4']
			vis = real + imag * 1j			
		else:
			uv_data = Table.read(filename, format='ascii')
			u=uv_data['u']
			v=uv_data['v']
			#w=uv_data['w']
			wgt=uv_data['wgt']
			real=uv_data['real']
			imag=uv_data['imag']
			vis = real + imag * 1j
			# print('array length: ', len(u))
			# print('Mean Re: ', np.mean(vis.real))
			# print('Mean Imag: ', np.mean(vis.imag))
			# print('Mean weight: ', np.mean(wgt))
			# print('Min weight: ', wgt.min())
		
		return u, v, vis, wgt

	if filename.endswith('.npz'):
		dat = np.load(filename)
		u, v, w, vis, wgt = dat['u'], dat['v'], dat['w'], dat['Vis'], dat['Wgt']
		# print('array length: ', len(u))
		# print('Mean Re: ', np.mean(vis.real))
		# print('Mean Imag: ', np.mean(vis.imag))
		# print('Mean weight: ', np.mean(wgt))
		# print('Min weight: ', wgt.min())
		return u, v, vis, wgt

	if filename.endswith('.dat'):
		uv_data = Table.read(filename, format='ascii')
		u=uv_data['u']
		v=uv_data['v']
		#w=uv_data['w']
		wgt=uv_data['weights']
		real=uv_data['real']
		imag=uv_data['imag']
		vis = real + imag * 1j
		return u, v, vis, wgt

def getdeg(stra: str,
	stdec: str,
	):
	"""Converts RA and DEC into angular values

	Parameters
	----------
	stra
		Right accession of the source in hms.

	stdec
		Declination of the source in dms.

	"""
	if(len(stra.split(' '))==1):
		H, M, S = [float(i) for i in stra.split(':')]
		D, Md, Sd = [float(i) for i in stdec.split(':')]
	if(len(stra.split(' '))==3):
		H, M, S = [float(i) for i in stra.split(' ')]
		D, Md, Sd = [float(i) for i in stdec.split(' ')]
	targetra_ang = (H*15.) + (M/4.) + (S/240.)
	ds = 1.
	if str(D)[0] == '-':
		ds, D = -1, abs(D)
	targetdec_ang = ds*(D + (Md/60.) + (Sd/3600.))
	return targetra_ang,targetdec_ang

def estimate_baseline_dependent_weight(q, V, bin_width):

	print('Re-weighting')
	uvBin = UVDataBinner(q, V, np.ones_like(q), bin_width)
	var = 0.5*(uvBin.error.real**2 + uvBin.error.imag**2) * uvBin.bin_counts

	weights = np.full_like(q, np.nan)
	left, right = uvBin.bin_edges
	for i, [l,r] in enumerate(zip(left, right)):
		idx = (q >= l) & (q < r)
		weights[idx] = 1/var[i]

	assert np.all(~np.isnan(weights)), "Weights needed for all data points"
	return weights

def fwhm_calc(bmaj,dist,re_au=None):
	sigma = np.sqrt(-bmaj**2/(2*np.log(0.003)))
	fwhm = 2.355*sigma
	print('fwhm error: ', 0.5*fwhm,'arcseconds - ',0.5*fwhm*dist,' AU')
	if re_au:
		return 0.5*fwhm*dist


