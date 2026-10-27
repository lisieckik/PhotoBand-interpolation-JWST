import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import json, os
from scipy.optimize import brentq
from mainFunctions import*

### This file interpolates the SDSS bands according to CEERS observations.
# There are two paths, one to use observed data, the other to focus only at SEDs.
# I try both. Here I only produce the files.

databaseBands = cigale_filters()

SEDpath = '/home/lisieckik/Pulpit/Darko galaxie/SFH2025/v6/completeness/v2/seds/'
SEDlist = os.listdir(SEDpath) ### noMIRI delayed
cat = fits.open('Uncertainties/Lisiecki+25_HST_NIRCam_MIRI_catalogue.fits')[1].data
bandsAvailable = ['jwst.nircam.F444W','jwst.nircam.F410M','jwst.nircam.F356W',
                  'jwst.nircam.F277W','jwst.nircam.F200W','jwst.nircam.F150W',
                  'jwst.nircam.F115W','hst.acs.wfc.F814W','hst.acs.wfc.F606W']
toInterpolate = ['sloan.sdss.u','sloan.sdss.g','sloan.sdss.r','sloan.sdss.i','sloan.sdss.z']
MCinter = 300



bandsAvailable = loadTransmissionCurves(bandsAvailable)
toInterpolate = loadTransmissionCurves(toInterpolate)

#
# results = Table(names = ('id', 'modelFlux', 'modelFlux_err',
#                          'SEDFlux', 'SEDFluxShort','SEDFluxShort_err',
#                          'SEDFluxLong','SEDFluxLong_err'))
#
# for bInt in toInterpolate:
#     nn = 0
#     for galSED in SEDlist:
#         nn+=1
#         if nn%100 == 0:
#             print('%.2f'%(nn/len(SEDlist)*100) + '% done, ' +'%s band'%bInt)
#         catRow = np.where(cat['id'] == int(galSED.split('.')[0]))[0]
#         if len(catRow) == 0:continue
#         catRow = cat[catRow]
#         redshift = catRow['redshift'][0]
#
#         wavelengthAvailable = {}
#         for b in bandsAvailable:
#             if catRow[b] >0:
#
#                 tc = bandsAvailable[b]
#                 plt.plot(tc[:,0]/10,tc[:,1]*1e-4, c = 'k', alpha = 0.1)
#                 w = np.average(tc[:,0],weights=tc[:,1])
#                 wavelengthAvailable[b] = float(w/10)
#         if len(wavelengthAvailable)<=5:
#             continue
#
#         short = {}
#         long = {}
#         wavelengthInterpolated = {}
#         shorter = -1
#         longer = -1
#         for b in [bInt]:
#             tc = toInterpolate[b]
#             w = np.average(tc[:,0],weights=tc[:,1])*(1+redshift)/10
#             wavelengthInterpolated[b] = w
#             #### Find shorter lambda ####
#             indx = np.where(list(wavelengthAvailable.values()) < w)[0]
#             if len(indx) == 0:
#                 #print('There is no bluer band, continiue')
#                 continue
#             shorter = np.array(list(wavelengthAvailable.keys()))[indx]
#             indx = np.where(np.array(list(wavelengthAvailable.values()))[indx] ==
#                                np.max(np.array(list(wavelengthAvailable.values()))[indx]))[0]
#             shorter = shorter[indx][0]
#             tc = bandsAvailable[shorter]
#             #### Find longer lambda ####
#             indx = np.where(list(wavelengthAvailable.values()) > w)[0]
#             if len(indx) == 0:
#                 #print('There is no redder band, continiue')
#                 continue
#             longer = np.array(list(wavelengthAvailable.keys()))[indx]
#             indx = np.where(np.array(list(wavelengthAvailable.values()))[indx] ==
#                                np.min(np.array(list(wavelengthAvailable.values()))[indx]))[0]
#             longer = longer[indx][0]
#             tc = bandsAvailable[longer]
#             short[b] = shorter
#             long[b] = longer
#         if isinstance(shorter, int) or isinstance(longer, int):continue
#         SED = fits.open(SEDpath + galSED)[1].data
#
#         lam = SED['wavelength']
#         Fnu = SED['Fnu']
#
#         predicted = {}
#         for b in [bInt]:
#             _, yshortModel = convolveFilterWithSED(lam, Fnu, bandsAvailable[short[b]])
#             _, ylongModel = convolveFilterWithSED(lam, Fnu, bandsAvailable[long[b]])
#
#             yshortReal = yshortModel
#             ylongReal = ylongModel
#             yshortReal_err = np.median(cat[short[b]+'_err'][cat[short[b]]>0])
#             ylongReal_err = np.median(cat[long[b]+'_err'][cat[long[b]]>0])
#             try:
#                 fnu, fnuErr = predictFnu_MC(yshortReal, ylongReal, yshortReal_err, ylongReal_err,
#                                   bandsAvailable[short[b]],
#                                   bandsAvailable[long[b]],
#                                   toInterpolate[b],
#                                   redshift,
#                                   n_iter=MCinter)
#                 predicted[b] = fnu
#
#                 _,y = convolveFilterWithSED(lam, Fnu, toInterpolate[b], redshift =redshift)
#                 results.add_row([catRow['id'], fnu, np.mean(fnuErr),
#                                  y, yshortReal, yshortReal_err,
#                                  ylongReal, ylongReal_err])
#             except Exception as Exerr:
#                 print(Exerr)
#                 pass
#
#     results.write('Uncertainties/fromModelData_%s.fits'%bInt, overwrite = True)



results = Table(names = ('id', 'modelFlux', 'modelFlux_err',
                         'SEDFlux', 'SEDFluxShort','SEDFluxShort_err',
                         'SEDFluxLong','SEDFluxLong_err'))
for bInt in toInterpolate:
    nn = 0
    for galSED in SEDlist:
        nn+=1
        if nn%100 == 0:
            print('%.2f'%(nn/len(SEDlist)*100) + '% done')
        catRow = np.where(cat['id'] == int(galSED.split('.')[0]))[0]
        if len(catRow) == 0:continue
        catRow = cat[catRow]
        redshift = catRow['redshift'][0]

        wavelengthAvailable = {}
        for b in bandsAvailable:
            if catRow[b] >0:

                tc = bandsAvailable[b]
                plt.plot(tc[:,0]/10,tc[:,1]*1e-4, c = 'k', alpha = 0.1)
                w = np.average(tc[:,0],weights=tc[:,1])
                wavelengthAvailable[b] = float(w/10)
        if len(wavelengthAvailable)<=5:
            continue

        short = {}
        long = {}
        wavelengthInterpolated = {}
        shorter = -1
        longer = -1
        for b in [bInt]:
            tc = toInterpolate[b]
            w = np.average(tc[:,0],weights=tc[:,1])*(1+redshift)/10
            wavelengthInterpolated[b] = w
            #### Find shorter lambda ####
            indx = np.where(list(wavelengthAvailable.values()) < w)[0]
            if len(indx) == 0:
                #print('There is no bluer band, continiue')
                continue
            shorter = np.array(list(wavelengthAvailable.keys()))[indx]
            indx = np.where(np.array(list(wavelengthAvailable.values()))[indx] ==
                               np.max(np.array(list(wavelengthAvailable.values()))[indx]))[0]
            shorter = shorter[indx][0]
            tc = bandsAvailable[shorter]
            #### Find longer lambda ####
            indx = np.where(list(wavelengthAvailable.values()) > w)[0]
            if len(indx) == 0:
                #print('There is no redder band, continiue')
                continue
            longer = np.array(list(wavelengthAvailable.keys()))[indx]
            indx = np.where(np.array(list(wavelengthAvailable.values()))[indx] ==
                               np.min(np.array(list(wavelengthAvailable.values()))[indx]))[0]
            longer = longer[indx][0]
            tc = bandsAvailable[longer]
            short[b] = shorter
            long[b] = longer
        if isinstance(shorter, int) or isinstance(longer, int):continue
        SED = fits.open(SEDpath + galSED)[1].data

        lam = SED['wavelength']
        Fnu = SED['Fnu']


        for b in [bInt]:
            _, yshortModel = convolveFilterWithSED(lam, Fnu, bandsAvailable[short[b]])
            _, ylongModel = convolveFilterWithSED(lam, Fnu, bandsAvailable[long[b]])

            yshortReal = catRow[short[b]][0]
            ylongReal = catRow[long[b]][0]
            yshortReal_err = catRow[short[b]+'_err'][0]
            ylongReal_err = catRow[long[b]+'_err'][0]
            try:
                fnu, fnuErr = predictFnu_MC(yshortReal, ylongReal, yshortReal_err, ylongReal_err,
                                  bandsAvailable[short[b]],
                                  bandsAvailable[long[b]],
                                  toInterpolate[b],
                                  redshift,
                                  n_iter=MCinter)

                _,y = convolveFilterWithSED(lam, Fnu, toInterpolate[b], redshift =redshift)
                results.add_row([catRow['id'], fnu, np.mean(fnuErr),
                                 y, yshortReal, yshortReal_err,
                                 ylongReal, ylongReal_err])
            except Exception as Exerr:
                print(Exerr)
                pass

    results.write('Uncertainties/fromRealData_%s.fits'%bInt, overwrite = True)



