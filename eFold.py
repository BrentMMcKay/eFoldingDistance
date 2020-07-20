import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import scipy.optimize as optimization
import geopy.distance
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import sys
import warnings
import io
import pickle

####################################################
def binTime( time, values, binEdges, binFunc=np.nanmean ):
    '''
    Place a set of temperature values into user-defined bins based on times associated with each value.
    Return the median bin values and the result of a user-passed function over the contents of each bin.
 
    @param time: 1-dimensional array of times
    @param values: 1-dimensional array of times
    @param binEdges: 1-dimensional array containing edge values of each bin
    @param binFunc: Function used to calculate binY values. Default is np.nanmean()
 
    @return binnedTimes: 1-dimensional array containing the average of the bin lower bound and bin upper bound
    @return binnedTemps: 1-dimensional array containing the result of binFunc executed over the contents of each bin
   '''
    
    # binnedTemps will contain the mean (or binFunc) value of each bin
    numBins = len(binEdges)-1
    binnedTemps = np.repeat(np.NaN, numBins)

    # binnedTimes is an array containing the mean values of the time bin columns
    binnedTimes = np.mean(np.column_stack((binEdges[:-1],binEdges[1:])),axis=1)

    #print(binEdges)
    for i in range(0,numBins):
        # Get the bin edges in ascending order
        thisBinEdge = np.sort(binEdges[i:i+2])

        # Fill the bin
        q = (time>=thisBinEdge[0]) & (time<thisBinEdge[1])
        #print(q)
        #print(values[q])
        vq = values[q]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            binnedTemps[i] = binFunc( vq )

        #print('avg delta',thisBinEdge,binnedTemps[i])
        
    return binnedTimes, binnedTemps

###############################################
def bandPassFilter(times,temps,cutoff=10,btype='highpass'):
    '''
    Create and execute a filter over a set of temperatures/times, filtering out low or high frequencies.
 
    @param times: 1-dimensional array containing the time of each measurement
    @param temps: 1-dimensional array containing the temperatures
    @param cutoff: Cutoff value for the filter. The cutoff frequency is calculated as (1/(cutoff/step)), where
                   step is the median difference of all values in array 'times'. E.g., if the entries in 'times'
                   have a median difference of one year, setting cutoff=10 means the filter cutoff frequency would 
                   0.1 Hz
    @param btype: 'highpass' or 'lowpass'; defaults to 'highpass'
 
    @return filteredTemps: 1-dimensional array containing the filtered temperatures
   '''
    
    # Create a butterworth high pass filter to filter out the low frequency changes
    step = np.median(abs(np.diff(times)))
    buffB,buffA = butter(1,1/(cutoff/step),btype=btype)

    # create array of all non-NaN entries
    nonNans = ~np.isnan(temps)
    goodTemps = temps[nonNans]

    # create array to hold the filtered numbers; leave NaNs alone
    filteredTemps = np.full(len(temps),np.NaN)

    # Make sure there are some nonNan values
    if len(goodTemps) == 0:
        return filteredTemps

    # Set the padding for smaller sites
    elif len(goodTemps) < 7:
        pad = 0
    else:
        pad = 6

    # Run the filter on all non-NaNs
    filteredTemps[nonNans] = filtfilt(buffB,buffA,goodTemps,padlen=pad)

    #fig, ax1 = plt.subplots()
    #ax1.plot(goodTemps,color='black')
    #ax1.plot(filteredTemps[nonNans],color='red')
    #plt.show()

    return filteredTemps

###############################################
def binAndFilter( siteList, timeStart=1800,timeEnd=2017,timeStep=1,highBandPass=True,\
                  bandPassCutoff=10,replaceNaN=True,replaceNaNDivisor=12,outFid='' ):
    '''
    Bin and filter temperatures for a list of sites. 
    
    @param siteList: A list of sites. Each list entry is a dictionary with 
                     the following key/value pairs:
        sitename: Name of site
        temps: temperature readings for the site (numpy 1D array).
        times: time of each temperature reading; values are floats representing years (numpy 1D array)
    @param timeStart: Starting edge of first bin
    @param timeEnd: Trailing edge of last bin
    @param timeStep: Width of each bin
    @param highBandPass: Set to True if a high band pass filter should be used to filter out low frequency variations
    @param bandPassCutoff: Cutoff for the band pass filter; units is number of timesteps
    @param replaceNaN: Replace all missing temp readings. Defaults to True
    @param replaceNaNDivisor: Replace all missing temp readings with the mean value of readings with the same index modulus when index is divided by this value. Default is 12, which would replace each NaN with the mean of the measurements from the same month, if the readings are one month apart.
    @param outFid: If specified, filtered data and siteList will be saved in a pickle file with 
                   this filename.

'''
    
    # Create one column for each site
    ncols = len(siteList)
    
    # Define the bin edges
    timeBinEdges = np.arange(timeStart, timeEnd+2, step=timeStep)

    # Create array holding all the filtered temp data for each site and each year: 
    # row index is time window; column index is site
    filteredArray = np.empty( [len(timeBinEdges)-1,ncols] )
    
    # Do the binning for each site
    #for i in range(0,3):
    for i in range(0,len(siteList)):
        # Load the temperature values and the times into arrays
        temps = np.array(siteList[i]['temps'])
        times = np.array(siteList[i]['times'])
    
        if replaceNaN:
            # Replace all NaNs in the values array with the mean of its column
            temps2d = temps.reshape(-1,replaceNaNDivisor)
            column_means = np.nanmean(temps2d, axis=0)
            nanInds = np.where(np.isnan(temps2d))
            temps2d[nanInds] = np.take(column_means,nanInds[1])
            temps = temps2d.flatten()
    
        # Bin all the temps into time bins
        binnedTimes,binnedTemps = binTime( time=times, values=temps, binEdges=timeBinEdges )
    
        if highBandPass:
            # Run temps through a high pass filter to eliminate long term low frequency trends
            filteredTemps = bandPassFilter(binnedTimes,binnedTemps,cutoff=bandPassCutoff*timeStep) 
        else:
            filteredTemps = binnedTemps
    
        # Save the filtered temps for this site
        filteredArray[:,i] = filteredTemps 

    if outFid != '':
        # Store the filtered temperature data
        pickle.dump((siteList,filteredArray),open(outFid,'wb'))

    return filteredArray
    


###############################################
def eFoldingDistance( distArr,
                      corrArr,
                      minPts = 10, 
                      minClosePts = 3, 
                      weightPower = 2, 
                      distanceStabilizer=100, 
                      iterate = True,
                      iterationThreshold=100, 
                      iterationDistanceMultiplier = 3.0,
                      optionPlot=False,
                      optionPlotFilename='',
                      optionPlotTitle='',
                      optionPlotAllSites=False,
                    ):
    '''
    Calculate the e-Folding distance between a site and a group of other sites
 
    @param distArr: NxN matrix of distances between sites, where N is number of sites
    @param corrArr: NxN matrix of correlations of temperature values between sites, where N is number of sites
    @param minPts: Minimum number of nonNaN datapoints in corrArr. If threshold is not met, NaN is returned.
    @param minClosePts: Minimum number of close sites. If there aren't enough close points, NaN is returned.
    @param weightPower: Exponent used for the weights in the minimization function, which minimizes a in 
                        sum( ( e^(distances*a) - rvalues ) / (1/(distances+distanceStablizer)**weightPower) ).
    @param distanceStabilizer: Value in miles to add to distances to help prevent very near sites from having
                               outsized impact
    @param iterate: Iterate through curve fits until the change in the e-folding distance from the previous attempt
                    is less than iterationThreshold
    @param iterationThreshold: terminate loop when eFoldDistance has less than this change from previous iteration
    @param iterationDistanceMultiplier: On new iteration, set distance threshold for sites to 
                                        eFoldingDistance * iterationDistanceMultiplier:
    @param optionPlot: if True, produce a plot
    @param optionPlotFilename: Filename to save plot. If empty, plot is shown on screen and not saved
    @param optionPlotTitle: Title for the plot
    @param optionPlotAllSites: If True, plots all sites. Otherwise, plot only those sites used in the final curve fit.
 
   '''

    def f(distances, fitParameter):
        # Function for use in curve fitting. Returns e^(distances*fitParameter)
        return np.exp(fitParameter*distances)

    # Abort if there aren't enough nonNan correlation points
    if np.count_nonzero(~np.isnan(corrArr)) < minPts:
        return np.NaN,np.NaN,np.NaN

    # Make sure minClosePts >= 1
    minClosePts = max(1,minClosePts)

    # initial setting of distance threshold is the max distance to another site
    goodDist = np.nanmax(distArr)

    # Set up loop parameters
    delta = iterationThreshold+1
    eFoldOld = -iterationThreshold-1
    iterations = 0
    fitStart = -1/2000.0

    # Repeat until eFold distance changes by less than the threshold
    while delta >= iterationThreshold:
        # Find all indices with a nonNan correlation and distance under threshold
        good = np.logical_and(~np.isnan(corrArr),(distArr<=goodDist))

        # Return NaN if there aren't any sites that meet the threshold
        if np.sum(good) <= 1:
            return np.NaN, np.NaN,np.NaN

        # Get R values and distances for all sites within the threshold
        r = corrArr[good]
        d = distArr[good]

        # Create weight array
        weights = 1/(d+distanceStabilizer)**weightPower

        # Fit an exponentially decreasing curve to (d,r)
        fitCoefficienceArray,fitCovariance = \
                    optimization.curve_fit(f, d, r, fitStart, sigma=weights, absolute_sigma=False, maxfev=1000)
        fitCoef = fitCoefficienceArray[0]

        # Calculate the e-folding distance
        # D0 = ln(1)/a = 0
        # D1 = ln(1/e)/a = -1/a
        # eFoldDistance = D1-D0 = -1/a = -1/fitCoef
        eFoldDistance = -1/fitCoef

        # Prepare for next loop: update start value, max distance, cutoff threshold, and save current eFoldDistance
        fitStart = fitCoef
        goodDist = eFoldDistance*iterationDistanceMultiplier
        delta = abs(eFoldDistance-eFoldOld)
        eFoldOld = eFoldDistance

        # If number of close sites is less than the threshold, return NaN
        if eFoldDistance < np.sort(d)[minClosePts-1]:
            return np.NaN,np.NaN,np.NaN

        # Create a correlation coefficient matrix
        r2 = np.corrcoef(np.exp(fitCoef*d),r)[1,0]
        if iterations == 0:
            origR2 = r2
            origFitCoef = fitCoef

        # Break loop after 10 tries or if caller specified no iterations:
        iterations += 1
        if iterations >= 10 or not iterate:
            break
        
    # Create plot if requested
    if optionPlot:
        fig,ax1 = plt.subplots()
        if optionPlotAllSites:
            curveFullX = np.arange(0,max(distArr))
            curveFullY = np.exp(origFitCoef*curveFullX)
            ax1.plot(curveFullX,curveFullY,color='pink')
            ax1.scatter(distArr[~good],corrArr[~good],color='gray')
            ax1.text(0.65,0.85,r'$r^2(all)$={0:4.2f}'.format(origR2),transform=ax1.transAxes,color='gray')
        ax1.scatter(d,r,color='black')
        curveX = np.arange(0,max(distArr))
        curveY = np.exp(fitCoef*curveX)
        ax1.plot(curveX,curveY,color='red')
        ax1.axvline(x=eFoldDistance,color='green')
        ax1.axhline(y=0,color='blue')
        ax1.text(0.65,0.95,'e-fold distance {0:4.0f} km'.format(eFoldDistance),transform=ax1.transAxes,color='black')
        ax1.text(0.65,0.90,r'$r^2$={0:4.2f}'.format(r2),transform=ax1.transAxes,color='black')
        if optionPlotTitle != '':
            plt.title(optionPlotTitle)
        if optionPlotFilename == '':
            plt.show()
        else:
            plt.savefig(optionPlotFilename,format='pdf')

    return eFoldDistance, r2, origR2

###############################################

def calcEFold( siteList, filteredArray, minPeriods=8, distanceThreshold=10000 ):
    '''
    Calculate the eFolding distance and r^2 value for each site in siteList
    @param siteList: list of dictionaries, one for each site
    @filteredArray: Array containing preprocessed temperature readings for each site
    @minPeriods: Minimum number of corresponding timeframes between sites to consider
    @distanceThreshold: Maximimum distance between sites to consider
    '''
    # Calculate correlation coefficients between all samples and sites, leaving NaN if insufficient number of samples.
    corrDf = pd.DataFrame(filteredArray).corr(min_periods=minPeriods)
    corrArray = corrDf.values
    
    eFold = []
    
    # Loop thru all sites
    for i in range(0,len(siteList)):
        
        # Calculate distance from this site to all others
        coord1 = (siteList[i]['latitude'],siteList[i]['longitude'])
    
        distArray = np.arange(len(siteList),dtype=float).reshape(len(siteList),-1)
    
        for j in range(0,len(siteList)):
            coord2 = (siteList[j]['latitude'],siteList[j]['longitude'])
            dist = geopy.distance.distance( coord1,coord2 ).km
            distArray[j,0] = dist
        
        # Create distance and correlation arrays where distance is less than the cutoff
        goodInd = (distArray <= distanceThreshold)
        goodDistArray = distArray[goodInd].reshape(np.sum(goodInd),-1)
        goodCorrArray = corrArray[i].reshape(len(siteList),-1)[goodInd].reshape(np.sum(goodInd),-1)
        # Calculate the eFolding distances
        eFoldDistance, r2, r2all = eFoldingDistance( goodDistArray, 
                                                     goodCorrArray, 
                                                     minClosePts = 1,
                                                     weightPower = 1,
                                                     optionPlot=False,
                                                     optionPlotAllSites=True,
                                                     #optionPlotFilename='test'+str(i)+'.pdf', 
                                                     optionPlotTitle=siteList[i]['sitename']
                                                   )
    
        site = {'name':siteList[i]['sitename'],'lat':siteList[i]['latitude'],'lon':siteList[i]['longitude'],
                     'eFoldDistance':eFoldDistance,'r2':r2 }
        eFold.append(site)
    
    return eFold



###############################################
def plotMap( latitudes, longitudes, values, 
             map_crs=ccrs.Robinson(),
             data_crs=ccrs.PlateCarree(),
             colormapMin=0.01, colormapMax=0.01,norm=colors.LogNorm,
             plotTitle='', dataLabel='', dotSize=10, colorMap='RdPu',
             plotFilename='', plotFormat='png'):
    '''
    Plot a dataset onto a world map

    @param latitudes: list of latitudes for each datapoint
    @param longitudes: list of longitudes for each datapoint
    @param values: list of values for each datapoint
    @param plotTitle: Title of the plot
    @param dataLabel: Label for the plot legend
    @param dotSize: Size in pixels of each datapoint
    @param colorMap: A matplotlib colormap name. 
           See http://matplotlib.org/3.1.1/gallery/color/colormap_reference.html 
           for a list of choices.
    @param plotFilename: If not empty, save the plot to this filename rather than displaying it
    @param plotFormat: Format of the saved plot

    '''

    # Create a figure
    plt.figure()
    ax = plt.axes(projection=map_crs)

    # Plot the entire earth with country and coastline borders
    ax.set_global()
    ax.add_feature(cf.BORDERS)
    ax.add_feature(cf.COASTLINE)
    colors = values
    if colormapMin < 0.1:
        colormapMin = min(colors)
    if colormapMax < 0.1:
        colormapMax = max(colors)

    # Put the sites on the map with color based on eFoldDistance or r2
    ax.scatter(longitudes,latitudes,s=dotSize,c=colors,cmap=colorMap,transform=data_crs,norm=norm(colormapMin, colormapMax))

    # Create a colorbar legend
    sm = plt.cm.ScalarMappable(cmap=colorMap,norm=norm(colormapMin, colormapMax))
    sm._A = []
    cbar = plt.colorbar(sm,ax=ax)

    cbar.set_label(dataLabel)

    # Title the plot
    plt.title(plotTitle)

    # Save or show the plot
    if plotFilename == '':
        plt.show()
    else:
        plt.savefig(plotFilename,format=plotFormat)
