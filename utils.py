import numpy as np

default_cmap = "inferno"

sigma_to_FWHM = 2.0 * np.sqrt(2.0 * np.log(2))
FWHM_to_sigma = 1.0 / sigma_to_FWHM
arcsec = np.pi / 648000


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

def miriad2txt(vis, freq, out=None):
	'''
	Converts miriad uvfits exported from the fits function to a .txt file
	Freq [GHz]
	'''
	if freq is None:
		raise ValueError('A observing frequency is required to scale the raw uvfits')
	
	freq_factor = freq*10**9

	#Reading in the fits file
	uv_fits = Table.read(vis)

	#Converting to a riendly format
	U = uv_fits['UU']*freq_factor
	V = uv_fits['VV']*freq_factor
	DATA = uv_fits['DATA']

	#Setting empty arrays for loop
	Re = []
	Img = []
	weight = []
	#Extracting the Re, Img, and weight variables from the DATA
	for i in range(len(DATA)):

		data = DATA[i,0,0,:,0,:] 
		mask = data[:,2]>0

		if (mask).any():
			data = data[mask]

		Re.append(np.ma.average(data[:,0], weights=data[:,2],axis=0))
		Img.append(np.ma.average(data[:,1], weights=data[:,2],axis=0))
		weight.append(np.ma.sum(data[:,2]))


	#Constructing uv data table
	uv_data = Table([U, V, Re, Img, weight], names=['u', 'v', 'Re', 'Im', 'weights'])

	if out:
		ascii.write(uv_data, out, overwrite=True)
	else:
		ascii.write(uv_data, 'miriad_uvdata.txt', overwrite=True)

	return(uv_data)

def alma2txt(vis,freq,out=None):
	'''
	Converts casa uvfits exported from the exportuvfits function to a .txt file
	Freq [GHz]
	'''
	if freq is None:
		raise ValueError('A observing frequency is required to scale the raw uvfits')
	
	freq_factor = freq*10**9

	#Reading in the fits file
	uv_fits = Table.read(vis)

	#Converting to a friendly format
	U = uv_fits['UU']*freq_factor
	V = uv_fits['VV']*freq_factor
	DATA = uv_fits['DATA']

	#Setting empty arrays for loop
	Re = []
	Img = []
	weight = []
	u = []
	v = []
	#Extracting the u, v, Re, Img, and weight variables from the DATA
	for i in range(len(DATA)):

		data = DATA[i,0,0,0,0,:,:] 
		mask = data[:,2]>0

		if (mask).any():
			data = data[mask]

		if np.ma.sum(data[:,2])>0:
			Re.append(np.ma.average(data[:,0], weights=data[:,2],axis=0))
			Img.append(np.ma.average(data[:,1], weights=data[:,2],axis=0))
			weight.append(np.ma.sum(data[:,2]))
			u.append(U[i])
			v.append(V[i])

	uv_data = Table([u, v, Re, Img, weight], names=['u', 'v', 'Re', 'Im', 'weights'])

	if out:
		ascii.write(uv_data, out, overwrite=True)
	else:
		ascii.write(uv_data, 'casa_uvdata.txt', overwrite=True)

	return(uv_data)



