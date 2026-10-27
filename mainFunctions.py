import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.table import Table
import json, os
from scipy.optimize import brentq

def convolveFilterWithSED(sedX, sedY, tc, redshift = 0):
    transX, transY = tc[:,0]*(1+redshift), tc[:,1]
    sedX = sedX*10 # AA
    ind = np.where((sedX > np.min(transX)) & (sedX < np.max(transX)))[0]
    xnew = sedX[ind]
    fluxNew = sedY[ind]

    ynew = np.interp(xnew,transX, transY)

    F = fluxNew * ynew/xnew

    yFlux = np.trapezoid(F, xnew)
    norm = np.trapezoid(ynew/xnew, xnew)
    xmean = transX[transY == np.max(transY)]
    realY = yFlux / norm
    return xmean, realY
def cigale_filters(v=25):
    """
    This function loads the json database
    :param v: CIGALE version. Right now only 25 available.
    :return: List of names of bands available in the database.
    """
    path = f"CIGALE{v}_filters_parsed.json"
    with open(path, "r") as f:
        loaded = json.load(f)
    return np.array(list(loaded.keys()))
def loadTransmissionCurves(bandList):
    """
    This function loads all bands from the database.
    :param bandList: List of names of the bands.
    :return: Dictionary of numpy.array transmission curves listed by the name of bands.
    """
    loaded = {}
    for bOG in bandList:
        b = bOG.split('.')
        tc = "Bands/"
        for instrument in b[0:-1]:
            tc+="%s/"%(instrument)

        try:
            fileList = os.listdir(tc)
        except:
            continue

        for f in fileList:
            indxStart = f.find(b[-1])
            if f.find(b[-1])>=0 and f[indxStart::][-5::] == b[-1][-1] + '.dat':
                loaded[bOG] = np.loadtxt(tc + f)
    for bOG in bandList:
        if not np.isin(bOG, list(loaded.keys())):
            print('I could not find %s is not in the database, it will be skipped'%bOG)
            print('Check the spelling or download it from SVO')
    return loaded
def findBandByName(nameElement, database):
    """
    This function helps you find bands of your interest in the database.
    It supports recursive quarrying (using results of this function as the variable in the next run).
    :param nameElement: string, name you are looking for, e.g. hst
    :param database: list of elements in database.
    :return: list of names with the nameElement in them.
    """
    indices = np.array([i for i, s in enumerate(database) if s.find(nameElement)>=0])
    return database[indices]
def predictFnu(fnu1, fnu2, filter1, filter2, filter3, redshift):
    """
    This function interpolates the rest-frame flux according to two observations (one bluer and one redder).
    It assumes that the underlying SED has a form of a powerlaw:
    Flam = lambda**beta.
    This is true for UV part of SED as well as the optical-NIR part.
    We have to take caution close to balmer break.
    It deconvolves the SED from the transmission curves and fit Beta with emitted fluxes.
    At the end, it convolves the powerlaw with transmission curve of target band.
    For the sake of mental health, it uses the pivot wavelength (decrease the risk of numerical issues).
    :param fnu1: float, flux of bluer band in mJy.
    :param fnu2: float, flux of redder band in mJy.
    :param filter1: numpy.array, two column variable of transmission curve of bluer band.
    :param filter2: numpy.array, two column variable of transmission curve of redder band.
    :param filter3: numpy.array, two column variable of transmission curve of target band.
    :param redshift: redshift of the source.
    :return: flux in mJy of the target band and Beta.
    """
    lam0 = 5000.0  # Reference wavelength, in Angstroms
    def band_flux(beta, filt, z = 0):
        """
        This function estimates the flux acording to different betas for the fitting algorithm.
        :param beta: float
        :param filt: numpy array
        :param z: redhist
        :return: flux
        """
        lam, T = filt[:,0]*(1+z), filt[:,1]
        idx = np.argsort(lam)
        lam, T = lam[idx], T[idx]
        return np.trapezoid((lam / lam0)**(beta + 2) * T / lam, lam) / np.trapezoid(T / lam, lam)

    def equation(beta):
        """
        Function for fitting, it evaluates how far is the given point from a solution.
        :param beta: beta
        :return: distance
        """
        return (band_flux(beta, filter1) / band_flux(beta, filter2) - fnu1 / fnu2)

    beta = brentq(equation, -10, 10)
    A = fnu1 / band_flux(beta, filter1)
    fnu3 = A * band_flux(beta, filter3, z = redshift)
    return fnu3, beta
def predictFnu_MC(fnu1, fnu2, err1, err2,
                   filter1, filter2, filter3, redhsift,
                   n_iter=1000):
    """
    Evaluates the results of predicted Fnu with Monte Carlo approach to find the uncertainties.
    :param fnu1: float, flux of bluer band in mJy.
    :param fnu2: float, flux of redder band in mJy.
    :param err1: float, flux error of bluer band in mJy.
    :param err2: float, flux error of redder band in mJy.
    :param filter1: numpy.array, two column variable of transmission curve of bluer band.
    :param filter2: numpy.array, two column variable of transmission curve of redder band.
    :param filter3: numpy.array, two column variable of transmission curve of target band.
    :param redhsift: redshift.
    :param n_iter: number of itterations for MC.
    :return: Flux and its uncertainties (16th and 84th percentile) as low and high distance from flux.
    """
    try:
        fnu3, beta = predictFnu(fnu1, fnu2, filter1, filter2, filter3, redhsift)
    except:
        return -99, (-99, -99)
    fnu3_samples = []
    beta_samples = []
    for _ in range(n_iter):
        f1 = np.random.normal(fnu1, err1)
        f2 = np.random.normal(fnu2, err2)
        # Fluxes must be positive for our power-law model
        if f1 <= 0 or f2 <= 0:
            continue
        try:
            f3, b = predictFnu(f1, f2, filter1, filter2, filter3, redhsift)
            fnu3_samples.append(f3)
            beta_samples.append(b)
        except ValueError:
            continue
    fnu3_samples = np.asarray(fnu3_samples)
    # 16th and 84th percentiles
    p16, p84 = np.percentile(fnu3_samples, [16, 84])
    err_low = fnu3 - p16
    err_high = p84 - fnu3
    return fnu3, (err_low, err_high)
def GNUAstroImplementation(bandsAvailablePaths,
                           toInterpolate,
                           redshift,
                           fluxColumn = 'SIGCLIP_MEAN',
                           fluxColumnErr = 'SIGCLIP_STD',
                           outputFile = 'results.fits',
                           MC_iter = 1000):
    """
    This function interpolates all rest-frame bands you need according to other available observations.
    It takes as arguments GNUAstro SB profiles and produces a similar file in the end (the same shape).
    :param bandsAvailablePaths: Dictionary of bands and paths to the radial profiles.
    :param toInterpolate: List of bands to interpolate
    :param redshift: redshift
    :param fluxColumn: Name of column you want to use as flux, default SIGCLIP_MEAN
    :param fluxColumnErr: Name of column you want to use as flux error, default SIGCLIP_STD
    :param outputFile: Path to result file
    :param MC_iter: Number of Monte Carlo iterations for the error estimation.
    :return: Astropy.table of the same length as input radial profiles.
    """
    bandsAvailable = loadTransmissionCurves(list(bandsAvailablePaths.keys()))
    toInterpolate = loadTransmissionCurves(toInterpolate)

    wavelengthAvailable = {}
    for b in bandsAvailable:
        tc = bandsAvailable[b]
        w = np.average(tc[:, 0], weights=tc[:, 1])
        wavelengthAvailable[b] = float(w / 10)

    short = {}
    long = {}
    wavelengthInterpolated = {}
    for b in toInterpolate:
        tc = toInterpolate[b]
        w = np.average(tc[:, 0], weights=tc[:, 1]) * (1 + redshift) / 10
        wavelengthInterpolated[b] = w
        #### Find shorter lambda ####
        indx = np.where(list(wavelengthAvailable.values()) < w)[0]
        shorter = np.array(list(wavelengthAvailable.keys()))[indx]
        indx = np.where(np.array(list(wavelengthAvailable.values()))[indx] ==
                        np.max(np.array(list(wavelengthAvailable.values()))[indx]))[0]
        shorter = shorter[indx][0]
        tc = bandsAvailable[shorter]
        #### Find longer lambda ####
        indx = np.where(list(wavelengthAvailable.values()) > w)[0]
        longer = np.array(list(wavelengthAvailable.keys()))[indx]
        indx = np.where(np.array(list(wavelengthAvailable.values()))[indx] ==
                        np.min(np.array(list(wavelengthAvailable.values()))[indx]))[0]
        longer = longer[indx][0]
        tc = bandsAvailable[longer]
        short[b] = shorter
        long[b] = longer

    names = []
    for b in toInterpolate:
        shortFile = open(bandsAvailablePaths[short[b]]).readlines()
        found = 0
        for l in range(len(shortFile)):
            if shortFile[l].find(fluxColumn) > 0:
                colFShort = l
                found += 1
            if shortFile[l].find(fluxColumnErr) > 0:
                colFerrShort = l
                found += 1
            if found > 1: break
        shortFile = open(bandsAvailablePaths[long[b]]).readlines()
        found = 0
        for l in range(len(shortFile)):
            if shortFile[l].find(fluxColumn) > 0:
                colFLong = l
                found += 1
            if shortFile[l].find(fluxColumnErr) > 0:
                colFerrLong = l
                found += 1
            if found > 1: break

        sbProfileShort = ascii.read(bandsAvailablePaths[short[b]])
        fluxShort = sbProfileShort['col' + str(colFShort + 1)]
        fluxErrShort = sbProfileShort['col' + str(colFerrShort + 1)]

        sbProfileLong = ascii.read(bandsAvailablePaths[long[b]])
        fluxLong = sbProfileShort['col' + str(colFLong + 1)]
        fluxErrLong = sbProfileShort['col' + str(colFerrLong + 1)]

        if len(fluxShort) != len(fluxLong):
            print('These tables are not the same lenght. Check if the sampling is the same!')
            raise ValueError

        resultBand = []
        resultBandErr = []
        for radius in range(len(fluxShort)):
            yshort = fluxShort[radius]
            yshort_err = fluxErrShort[radius]
            ylong = fluxLong[radius]
            ylong_err = fluxErrLong[radius]

            fnu, fnuErr = predictFnu_MC(yshort, ylong, yshort_err, ylong_err,
                                        bandsAvailable[short[b]],
                                        bandsAvailable[long[b]],
                                        toInterpolate[b],
                                        redshift, n_iter=MC_iter)
            resultBand = np.append(resultBand, [fnu])
            resultBandErr = np.append(resultBandErr, np.mean(fnuErr))

        try:
            result = np.column_stack([result, resultBand, resultBandErr])
        except:
            try:
                if len(result) != len(resultBand):
                    print('These tables are not the same lenght. Check if the sampling is the same!')
                    raise ValueError
            except:
                result = np.column_stack([resultBand, resultBandErr])
        names.append('%s' % b)
        names.append('%s_err' % b)

    result = Table(rows=result, names=names)
    result.write(outputFile, overwrite=True)
    return result