# Import modules

import subprocess
import sys
import math
import time
import tkinter as tk
from tkinter import filedialog
import copy
import pip
import warnings
from tqdm.notebook import tqdm
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import pprint as pp
import pywt
from statsmodels.robust import mad
from BaselineRemoval import BaselineRemoval

### Scipy

from scipy.ndimage.interpolation import shift
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve
import scipy.stats as stats
from scipy import interpolate
from scipy.sparse import csc_matrix, eye, diags

### Scikit-Learn

### Classification modles

### Discriminant analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
### Ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import HistGradientBoostingClassifier
### Gaussian process
from sklearn.gaussian_process import GaussianProcessClassifier
### Neural network
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
### SVM
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
### Linear model
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
### Neighbors
from sklearn import neighbors
### Tree
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import ExtraTreeClassifier

# Utilities
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Decomposition
from sklearn.decomposition import PCA
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD

#from sklearn import svm

import customerrors

#warnings.filterwarnings("ignore")

# Define all function used in the program

# Processing
def quickProcess(file_paths, sample_type, method='Basic'):
    if method == 'Basic':
        WN, array, sample_ID = readArrayFromFile(file_paths, sample_type)
        df = readArrayToDataFrame(array, 'Raw_array')
        df['Sample_type'] = sample_ID
        df = addColumnToDataFrame(df,
                                  smooth(df['Raw_array'],
                                         method = 'FFT',
                                         fourior_values = 300),
                                  'Smoothed_array')
        df = addColumnToDataFrame(df,
                                  normalise(df['Smoothed_array'],
                                            method = 'area',
                                            normalisation_indexs = [1000,1005],
                                            wavenumbers=WN),
                                  'Normalized_array')
        df = addColumnToDataFrame(df,
                                  baselineCorrection(df['Normalized_array'],
                                                     lam=10**4.5),
                                  'Baseline_corrected_array')
        df = addColumnToDataFrame(df,
                                  removeCosmicRaySpikes(df['Baseline_corrected_array'],
                                                        threshold = 5),
                                  'Despiked_array')
        df = addColumnToDataFrame(df,
                                  xAling(df['Despiked_array'],
                                  alingnemt_indexes = [1000,1005],
                                  wavenumbers=WN),
                                  'Baseline_corrected_alinged_array')
        df = addColumnToDataFrame(df,
                                  normalise(df['Baseline_corrected_alinged_array'],
                                            method = 'area',
                                            normalisation_indexs = [1000,1005],
                                            wavenumbers=WN),
                                  'Baseline_corrected_normalized_array')
    elif method == 'DNA_normlaisation':
        WN, array, sample_ID = readArrayFromFile(file_paths, sample_type)
        df = readArrayToDataFrame(array, 'Raw_array')
        df['Sample_type'] = sample_ID
        df = addColumnToDataFrame(df,
                                  smooth(df['Raw_array'],
                                         method = 'FFT',
                                         fourior_values = 300),
                                  'Smoothed_array')
        df = addColumnToDataFrame(df,
                                  normalise(df['Smoothed_array'],
                                            method = 'area',
                                            normalisation_indexs = [770,790],
                                            wavenumbers=WN),
                                  'Normalized_array')
        df = addColumnToDataFrame(df,
                                  baselineCorrection(df['Normalized_array'],
                                                     lam=10**4.5),
                                  'Baseline_corrected_array')
        df = addColumnToDataFrame(df,
                                  removeCosmicRaySpikes(df['Baseline_corrected_array'],
                                                        threshold = 5),
                                  'Despiked_array')
        df = addColumnToDataFrame(df,
                                  xAling(df['Despiked_array'],
                                  alingnemt_indexes = [775,785],
                                  wavenumbers=WN),
                                  'Baseline_corrected_alinged_array')
        df = addColumnToDataFrame(df,
                                  normalise(df['Baseline_corrected_alinged_array'],
                                            method = 'area',
                                            normalisation_indexs = [770,790],
                                            wavenumbers=WN),
                                  'Baseline_corrected_normalized_array')
    elif method == 'Heavy_normlaisation':
        WN, array, sample_ID = readArrayFromFile(file_paths, sample_type)
        df = readArrayToDataFrame(array, 'Raw_array')
        df['Sample_type'] = sample_ID
        df = addColumnToDataFrame(df,
                                  smooth(df['Raw_array'],
                                         method = 'FFT',
                                         fourior_values = 300),
                                  'Smoothed_array')
        df = addColumnToDataFrame(df,
                                  normalise(df['Smoothed_array'],
                                            method = 'area',
                                            normalisation_indexs = [990,1015],
                                            wavenumbers=WN),
                                  'Normalized_array')
        df = addColumnToDataFrame(df,
                                  baselineCorrection(df['Normalized_array'],
                                                     lam=10**4.5),
                                  'Baseline_corrected_array')
        df = addColumnToDataFrame(df,
                                  removeCosmicRaySpikes(df['Baseline_corrected_array'],
                                                        threshold = 5),
                                  'Despiked_array')
        df = addColumnToDataFrame(df,
                                  xAling(df['Despiked_array'],
                                  alingnemt_indexes = [1000,1005],
                                  wavenumbers=WN),
                                  'Baseline_corrected_alinged_array')
        df = addColumnToDataFrame(df,
                                  normalise(df['Baseline_corrected_alinged_array'],
                                            method = 'area',
                                            normalisation_indexs = [990,1015],
                                            wavenumbers=WN),
                                  'Baseline_corrected_normalized_array')
    elif method == 'Baseline_normalisation':
        WN, array, sample_ID = readArrayFromFile(file_paths, sample_type)
        df = readArrayToDataFrame(array, 'Raw_array')
        df['Sample_type'] = sample_ID
        df = addColumnToDataFrame(df,
                                  smooth(df['Raw_array'],
                                         method = 'FFT',
                                         fourior_values = 300),
                                  'Smoothed_array')
        array, baseline = baselineCorrection(df['Smoothed_array'], method = 'ALS', lam=10**4.5, return_baseline = True)
        df = addColumnToDataFrame(df, array, 'Baseline_corrected_array')
        baseline_norm_values = np.sum(baseline,axis=0)
        df = addColumnToDataFrame(df,
                                  normalise(df['Baseline_corrected_array'],
                                            method = 'custom_values',
                                            custom_values=baseline_norm_values),
                                  'Baseline_corrected_normalized_array')
        df = addColumnToDataFrame(df,
                                  removeCosmicRaySpikes(df['Baseline_corrected_normalized_array'],
                                                        threshold = 5),
                                  'Despiked_array')
        df = addColumnToDataFrame(df,
                                  xAling(df['Despiked_array'],
                                         alingnemt_indexes = [1000,1005],
                                         wavenumbers=WN),
                                  'Baseline_corrected_alinged_array')
    return df

def readArrayFromFile(file_path, sample_ID):
    """
    Reads the spectra from a WiER2 mapgrid file where there are four columns
    corrisponding to X, Y, Wavenumber, Intensity. Each repeat spectra is
    seperated into corrisponding rows in an array.

    ---------

    Arguments :

    file_path : file path(s) to the desiered file(s). If this is in the form of
                a list then the function will open each file on after another
                and append the results to a single array.
    sample_ID : takes a list of the same size as the file_path list and
                duplicates the entrys.

    ---------
    Returns   :

    WN        : a 1D array of a singel set of the wavenumbers for the
                corrisponding spectra.
    array     : a 2D array of the spectra contained in the supplied files.
    sample_ID : a 1D array of equal length to the number of spectra where each
                entry is the lable assinged to the file by the sample_ID
                argument
    """

    if type(file_path) != list:
        if type(file_path) != str:
            raise FilePathTypeError(file_path)

    if type(sample_ID) != list:
        if type(sample_ID) != str:
            raise LableTypeError(sample_ID)

    if type(file_path) == list:
        if len(file_path) != len(sample_ID):
            raise LabelSizeMissmachError(file_path, sample_ID)

    if type(file_path) == list:
        master_array = False
        master_sample_ID_list = []
        index = 0
        for file_active in file_path:
            with open(file_active) as filecontent:
                total_data = []
                for data in filecontent:
                    data = data.rstrip("\n")
                    data_list = data.split("\t")
                    total_data.append(data_list)
                wavenumbers, array, sample_ID_list = splitArray(np.array(total_data).astype(np.float), sample_ID[index])
            if type(master_array) == np.ndarray:
                master_array = np.hstack((master_array,array))
            else:
                master_array = array
            master_sample_ID_list.extend(sample_ID_list)
            index += 1
        return wavenumbers, np.transpose(master_array), master_sample_ID_list
    elif type(file_path) == str:
        with open(file_path) as filecontent:
            total_data = []
            for data in filecontent:
                data = data.rstrip("\n")
                data_list = data.split("\t")
                total_data.append(data_list)
            WN, array, sample_ID_list = splitArray(np.array(total_data).astype(np.float), sample_ID)
            return WN, np.transpose(array), sample_ID_list

def splitArray(array, sample_ID):
    # Format the array by the first X value (wavenumber) and splitting the array at each subsiquent
    # mach (indication of the start of a new measurment)
    raman_array = [np.array_split(array, x)
                   for x in np.where(array[:,-2] == array[0,-2])]
    raman_array = raman_array[0]
    del raman_array[0]
    array = np.dstack(raman_array)[:,-1,:]
    WN = np.dstack(raman_array)[:,-2,0]
    sample_ID_list = [sample_ID for i in range(np.shape(array)[1])]
    return WN, array, sample_ID_list

def waveletSmooth( x, wavelet="db4", level=1):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    # calculate a threshold
    sigma = mad( coeff[-level] )
    # changing this threshold also changes the behavior,
    # but I have not played with this very much
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec( coeff, wavelet, mode="per" )
    return y

def speyediff(N, d, format='csc'):
    """
    (utility function)
    Construct a d-th order sparse difference matrix based on
    an initial N x N identity matrix

    Final matrix (N-d) x N
    """

    assert not (d < 0), "d must be non negative"
    shape     = (N-d, N)
    diagonals = np.zeros(2*d + 1)
    diagonals[d] = 1.
    for i in range(d):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff
    offsets = np.arange(d+1)
    spmat = sparse.diags(diagonals, offsets, shape, format=format)
    return spmat


def whittaker_smooth(y, lmbd, d = 2):
    """
    Implementation of the Whittaker smoothing algorithm,
    based on the work by Eilers [1].
    [1] P. H. C. Eilers, "A perfect smoother", Anal. Chem. 2003, (75), 3631-3636

    The larger 'lmbd', the smoother the data.
    For smoothing of a complete data series, sampled at equal intervals
    This implementation uses sparse matrices enabling high-speed processing
    of large input vectors

    ---------

    Arguments :

    y       : vector containing raw data
    lmbd    : parameter for the smoothing algorithm (roughness penalty)
    d       : order of the smoothing

    ---------
    Returns :

    z       : vector of the smoothed data.
    """

    m = len(y)
    E = sparse.eye(m, format='csc')
    D = speyediff(m, d, format='csc')
    coefmat = E + lmbd * D.conj().T.dot(D)
    z = splu(coefmat).solve(y)
    return z

def smooth(array, method = 'Savitzky–Golay', window = 3, polynomial = 0, axis = 1, fourior_values = 3, wavelet = 'db29',
           wavelet_level = 1, lambda_val = 50000, d = 2):
    array = np.transpose(np.stack(array))
    smoothed_array = np.zeros(np.shape(array))
    if method == 'Savitzky–Golay':
        for spectra in range(np.shape(array)[axis]):
            smoothed_array[:,spectra] = savgol_filter(array[:,spectra],window,polynomial)
        return smoothed_array
    elif method == 'FFT':
        for spectra in range(np.shape(array)[axis]):
            padded_array = np.pad(array[:,spectra],
                                 (100, 100), 'constant',
                                 constant_values=((array[:,spectra][0],array[:,spectra][-1])))
            rft = np.fft.rfft(padded_array)
            rft[fourior_values:] = 0
            smoothed_array[:,spectra] = np.fft.irfft(rft)[100:-100]
        return smoothed_array
    elif method == 'Wavelet':
        for spectra in range(np.shape(array)[axis]):
            smoothed_array[:,spectra] = waveletSmooth(array[:,spectra], wavelet=wavelet, level=wavelet_level)
        return smoothed_array
    elif method == 'Whittaker':
        for spectra in range(np.shape(array)[axis]):
            smoothed_array[:,spectra] = whittaker_smooth(array[:,spectra], lambda_val, d = d)
        return smoothed_array

def normalise(array, axis = 1, method = 'max_within_range', normalisation_indexs = [890,910], wavenumbers=False,
              zero_min=False, custom_values = False, return_norlaisation_values = False):
    array = np.transpose(np.stack(array))
    if zero_min == True:
        array = array + abs(np.min(array))
    normalised_array = np.zeros(np.shape(array))
    normalisation_indexs_2 = normalisation_indexs
    if type(wavenumbers) == np.ndarray:
        normalisation_indexs_2[0] = np.absolute(wavenumbers - normalisation_indexs[0]).argmin()
        normalisation_indexs_2[1] = np.absolute(wavenumbers - normalisation_indexs[1]).argmin()
    normalisation_indexs_2 = sorted(normalisation_indexs_2)
    normalisation_values = []
    if method == 'scale':
        max_value = np.max(array)
        normalisation_values.append(max_value)
        normalised_array = array / max_value
    else:
        for spectra in range(np.shape(array)[axis]):
            if method == 'max_within_range':
                max_value = max(array[normalisation_indexs_2[0]:normalisation_indexs_2[1],spectra])
                normalised_array[:,spectra] = array[:,spectra] / max_value
                normalisation_values.append(max_value)
            elif method == 'max_whole_array':
                max_value = max(array[:,spectra])
                normalised_array[:,spectra] = array[:,spectra] / max_value
                normalisation_values.append(max_value)
            elif method == 'whole_array':
                max_value = sum(array[:,spectra])
                normalised_array[:,spectra] = array[:,spectra] / max_value
                normalisation_values.append(max_value)
            elif method == 'singel_point':
                normalised_array[:,spectra] = array[:,spectra] / array[normalisation_indexs_2[0],spectra]
                normalisation_values.append(array[normalisation_indexs_2[0],spectra])
            elif method == 'area':
                max_value = sum(array[normalisation_indexs_2[0]:normalisation_indexs_2[1],spectra])
                normalised_array[:,spectra] = array[:,spectra] / max_value
                normalisation_values.append(max_value)
            elif method == 'interp_area':
                f = interpolate.interp1d(range((normalisation_indexs_2[1]-normalisation_indexs_2[0])),
                                         array[normalisation_indexs_2[0]:normalisation_indexs_2[1],spectra],
                                         kind='quadratic')
                normalised_array[:,spectra] = array[:,spectra] / sum(f(np.arange(0,
                                                                    (normalisation_indexs_2[1]-normalisation_indexs_2[0])-1,
                                                                     0.1)))
                normalisation_values.append(sum(f(np.arange(0,(normalisation_indexs_2[1]-normalisation_indexs_2[0])-1,0.1))))
            elif method == 'max_within_interp_range':
                f = interpolate.interp1d(range((normalisation_indexs_2[1]-normalisation_indexs_2[0])),
                                         array[normalisation_indexs_2[0]:normalisation_indexs_2[1],spectra],
                                         kind='quadratic')
                normalised_array[:,spectra] = array[:,spectra] / np.max(f(np.arange(0,
                                                                    (normalisation_indexs_2[1]-normalisation_indexs_2[0])-1,
                                                                     0.1)))
                normalisation_values.append(np.max(f(np.arange(0,(normalisation_indexs_2[1]-normalisation_indexs_2[0])-1,0.1))))
            elif method == 'custom_values':
                normalised_array[:,spectra] = array[:,spectra] / custom_values[spectra]
                normalisation_values.append(custom_values[spectra])

    if method == 'area':
        max_value = np.max(np.mean(normalised_array[normalisation_indexs_2[0]:normalisation_indexs_2[1],:],axis=1))
        normalised_array = normalised_array / max_value
        normalisation_values = np.array(normalisation_values) * max_value
    elif method == 'interp_area':
        max_value = np.max(np.mean(normalised_array[normalisation_indexs_2[0]:normalisation_indexs_2[1],:],axis=1))
        normalised_array = normalised_array / max_value
        normalisation_values = np.array(normalisation_values) * max_value
    elif method == 'custom_values':
        max_value = np.max(np.mean(normalised_array,axis=1))
        normalised_array = normalised_array / max_value
        normalisation_values = np.array(normalisation_values) * max_value
    elif method == 'whole_array':
        max_value = np.max(np.mean(normalised_array,axis=1))
        normalised_array = normalised_array / max_value
        normalisation_values = np.array(normalisation_values) * max_value
    else:
        pass
    if return_norlaisation_values == True:
        return normalised_array, np.array(normalisation_values)
    else:
        return normalised_array

def subtractBackground(array, background, axis = 1, method = 'Dynamic'):
    array = np.transpose(np.stack(array))
    corrected_array = np.zeros(np.shape(array))
    index = 0
    if method == 'Dynamic':
        subtraction_factor = 0.01
        done = False
        while done == False:
            if np.shape(array)[1] <= index:
                done = True
            elif np.min(array[:,index]-(background*subtraction_factor)) <= 0:
                corrected_array[:,index] = array[:,index]-(background*subtraction_factor)
                index += 1
                subtraction_factor = 0.01
            else:
                subtraction_factor += 0.01
    elif method == 'Static':
        for index in range(np.shape(array)[1]):
            corrected_array[:,index] = array[:,index]-background
    return corrected_array

def removeCosmicRaySpikes(array, method = 'wavenumber', threshold_diff=5, threshold_wn=5, return_CRS_positions=False):
    array = np.transpose(np.stack(array))
    removed_spikes = []
    despiked_raman_array = copy.deepcopy(array)
    if method == 'wavenumber':
        row = 0
        for x in tqdm(array, desc='Despike', leave=False):
            index = 0
            for y in x:
                if y > np.mean(x) + np.std(x):
                    new_array = np.delete(x, index)
                    if  y > np.mean(new_array) + threshold_wn * np.std(new_array):
                        despiked_raman_array[row,index] = np.mean(new_array)
                        removed_spikes.append([index,row,y])
                index += 1
            row += 1
        if return_CRS_positions == True:
            return despiked_raman_array, removed_spikes
        else:
            return despiked_raman_array
    elif method == 'diff':
        matrix = array
        spikes = 1
        while spikes != 0:
            spikes_desc = spikes
            spikes = 0
            min_threshold = 0 - np.mean(np.std(np.diff(matrix,2),axis=1))*threshold_diff
            for spectrum in tqdm(range(np.shape(matrix)[0]),desc='Despike (' + str(spikes_desc) + ' Spikes Remaing)' , leave=False):
                second_diff_spectrum = np.diff(matrix[spectrum,:],2)
                for wavenumber in range(np.shape(matrix)[1]-2):
                    if min_threshold > second_diff_spectrum[wavenumber] or abs(min_threshold) < second_diff_spectrum[wavenumber]:
                        removed_spikes.append([wavenumber+1,spectrum,matrix[spectrum,wavenumber+1]])
                        matrix[spectrum,wavenumber+1] = matrix[spectrum,wavenumber+1]*0.95
                        #spikes += 1
        if return_CRS_positions == True:
            return matrix, removed_spikes
        else:
            return matrix
    elif method == 'consensus':
        removed_spikes_wn = []
        removed_spikes_diff = []
        despiked_raman_array = copy.deepcopy(array)
        row = 0
        for x in tqdm(array, desc='Despike', leave=False):
            index = 0
            for y in x:
                if y > np.mean(x) + np.std(x):
                    new_array = np.delete(x, index)
                    if  y > np.mean(new_array) + threshold_wn * np.std(new_array):
                        removed_spikes_wn.append([index,row,y,np.mean(new_array)])
                        #print('spike')
                index += 1
            row += 1

        matrix = array
        spikes = 1
        while spikes != 0:
            spikes_desc = spikes
            spikes = 0
            min_threshold = 0 - np.mean(np.std(np.diff(matrix,2),axis=1))*threshold_diff
            for spectrum in tqdm(range(np.shape(matrix)[0]),desc='Despike (' + str(spikes_desc) + ' Spikes Remaing)' , leave=False):
                second_diff_spectrum = np.diff(matrix[spectrum,:],2)
                for wavenumber in range(np.shape(matrix)[1]-2):
                    if min_threshold > second_diff_spectrum[wavenumber] or abs(min_threshold) < second_diff_spectrum[wavenumber]:
                        removed_spikes_diff.append([wavenumber+1,spectrum,matrix[spectrum,wavenumber+1]])

        #print(np.shape(np.array(removed_spikes_wn)))
        #print(np.shape(np.array(removed_spikes_diff)))
        mask = np.isin(np.array(removed_spikes_wn)[:,1:3],np.array(removed_spikes_diff)[:,1:3])
        detected_spikes = np.array(removed_spikes_wn)[mask[:,0],0:2].astype(int)
        new_values = np.array(removed_spikes_wn)[mask[:,0],3]
        index = 0
        for x in detected_spikes[:,0]:
            despiked_raman_array[detected_spikes[index,1],x] = new_values[index]
            index += 1
        return despiked_raman_array
    else:
        print('Error: Method not found = ' + str(method))

def removeCRSFast(array,threshold_wn=5, threshold_diff=5):
    array = np.stack(array)
    working_array = copy.deepcopy(array)

    mean = np.mean(working_array,axis=0)
    std = np.std(working_array,axis=0)

    mean = mean.reshape(mean.shape[0],-1)
    std = std.reshape(std.shape[0],-1)

    mean_array = np.pad(mean, [(0, 0), (0, np.shape(working_array)[0]-1)], 'mean')
    std_array = np.pad(std, [(0, 0), (0, np.shape(working_array)[0]-1)], 'mean')

    threshold_array = mean_array + (std_array)

    comparison_array = working_array > np.transpose(threshold_array)

    editted_array = copy.deepcopy(working_array)
    editted_array = editted_array.astype('float')
    editted_array[comparison_array] = np.nan

    mean_eddited = np.nanmean(editted_array,axis=0)
    std_eddited = np.nanstd(editted_array,axis=0)

    mean_eddited = mean_eddited.reshape(mean_eddited.shape[0],-1)
    std_eddited = std_eddited.reshape(std_eddited.shape[0],-1)

    mean_array_eddited = np.pad(mean_eddited, [(0, 0), (0, np.shape(array)[0]-1)], 'mean')
    std_array_eddited = np.pad(std_eddited, [(0, 0), (0, np.shape(array)[0]-1)], 'mean')

    threshold_array_eddited = mean_array_eddited + (std_array_eddited*threshold_wn)

    comparison_array_eddited = array > np.transpose(threshold_array_eddited)

    working_array = copy.deepcopy(array)

    diff = np.diff(working_array,2,axis=1)

    diff_mean = np.mean(diff,axis=1)
    diff_std = np.std(diff,axis=1)

    diff_mean = diff_mean.reshape(diff_mean.shape[0],-1)
    diff_std = diff_std.reshape(diff_std.shape[0],-1)

    diff_mean_array = np.pad(diff_mean, [(0, 0), (np.shape(diff)[1]-1, 0)], 'mean')
    diff_std_array = np.pad(diff_std, [(0, 0), (np.shape(diff)[1]-1, 0)], 'mean')

    threshold_array_diff = diff_mean_array + (diff_std_array*threshold_diff)

    comparison_array_diff = working_array > np.pad(threshold_array_diff, [(0,0),(1,1)], 'edge')

    editted_array_diff = copy.deepcopy(working_array)
    editted_array_diff = editted_array_diff.astype('float')
    editted_array_diff[comparison_array_diff] = np.nan

    consensus_array = np.logical_and(comparison_array_eddited, comparison_array_diff)

    corrected_array = working_array
    corrected_array[consensus_array] = np.transpose(mean_array_eddited)[consensus_array]
    return np.transpose(corrected_array)

#def baselineALS(y, lam, p, niter=10):
#    L = len(y)
#    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
#    w = np.ones(L)
#    for i in range(niter):
#        W = sparse.spdiags(w, 0, L, L)
#        Z = W + lam * D.dot(D.transpose())
#        z = spsolve(Z, w*y)
#        w = p * (y > z) + (1-p) * (y < z)
#    return z

def baselineALS(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def readFileToDataFrame(file, column_name):
    with open(file) as filecontent:
        total_data = []
        index = 0
        for data in filecontent:
            if index != 0:
                index = 1
                data = data.rstrip("\n")
                data_list = data.split(",")
                del data_list[0]
                total_data.append(data_list)
            index = 1
        A = np.array(total_data).astype(np.float)
    data = pd.DataFrame(columns=[column_name])
    for index in range(0,np.shape(A)[1]):
        row = pd.DataFrame({column_name:[A[:,index]]})
        data = data.append(row)
    return data

def readArrayToDataFrame(array, column_name, axis = 0):
    df = pd.DataFrame(columns=[str(column_name)])
    for sample in tqdm(array,desc='Data Frame',leave=False):
        data = pd.DataFrame({str(column_name) : [sample]}),
        df = df.append(data, ignore_index=True)
    return df

def addColumnToDataFrame(dataframe, column, column_name, axis = 1):
    dataframe[column_name] = ''
    index = 0
    if axis == 0:
        for sample in tqdm(column,desc=column_name,leave=False):
            dataframe[column_name].iloc[index] = sample
            index += 1
        return dataframe
    elif axis == 1:
        for sample in tqdm(np.transpose(column),desc=column_name,leave=False):
            dataframe[column_name].iloc[index] = sample
            index += 1
        return dataframe

def oval(h,w,y):
    return np.sqrt(((1-((y**2)/(h**2)))*(w**2)))

def rollingBall(spectrum,ball_H,ball_W):
    baseline = []
    oval_shape = []
    index = -ball_W
    for o in range(ball_W*2):
        oval_shape.append(oval(ball_W,ball_H,abs(index)))
        index += 1
    for x in range(len(spectrum)):
        ball = spectrum[x]-ball_H
        for y in range(ball_W*2):
            if (x-ball_W+y > 0) & (x-ball_W+y < len(spectrum)) & (index != 0):
                if spectrum[x-ball_W+y] < ball + oval_shape[y]:
                    ball = spectrum[x-ball_W+y] - oval_shape[y]
        baseline.append(ball)
    return np.array(baseline) + ball_H

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector

    https://github.com/zmzhang/airPLS/blob/master/airPLS.py
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=30, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector

    https://github.com/zmzhang/airPLS/blob/master/airPLS.py
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax):
                print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn)
        w[-1]=w[0]
    return z

def manuelBaseline(array,mask,polyorder=20):
    spectra_copy = copy.deepcopy(array)
    spectra_copy[mask] = np.nan
    idx = np.isfinite(spectra_copy)
    ab = np.polyfit(np.arange(len(array))[idx], spectra_copy[idx], 20)
    p = np.polyval(ab,range(len(array)))
    return p

def baselineCorrection(array, method = 'ALS', lam = 10**4, p = 0.01, niter = 10, fourier_values = 3,
                       fourier_type = 'RDFT', polynomial = 3, itterations = 10, ball_H = 0.1,
                       ball_W = 25, window_size = 100, lambda_airPLS = 30, porder = 1, itermax = 15,
                       return_baseline = False, mask=False, manuelPolyOrder=20):
    array = np.stack(array)
    baselined_array = np.zeros(np.shape(array))
    if method == 'ALS':
        index = 0
        for spectra in tqdm(array,desc='Baseline (ALS)',leave=False):
            baselined_array[index,:] = spectra - baselineALS(spectra, lam=lam, p=p, niter=niter)
            index += 1
    elif method == 'airPLS':
        index = 0
        for spectra in tqdm(array,desc='Baseline (airPLS)',leave=False):
            baselined_array[index,:] = spectra - airPLS(spectra, lambda_=lambda_airPLS, porder=porder, itermax=itermax)
            index += 1
    elif method == 'FFT':
        index = 0
        for spectra in tqdm(array,desc='Baseline (FFT / ' + str(fourier_type) + ')',leave=False):
            if fourier_type == 'DFT':
                rft = np.fft.fft(spectra)
                rft[:fourier_values] = 0
                baselined_array[index,:] = np.fft.ifft(rft)
            elif fourier_type == 'RDFT':
                rft = np.fft.rfft(spectra)
                rft[:fourier_values] = 0
                baselined_array[index,:] = np.fft.irfft(rft)
            elif fourier_type == 'Hermitian':
                rft = np.fft.hfft(spectra)
                rft[:fourier_values] = 0
                baselined_array[index,:] = np.fft.ihfft(rft)
            index += 1
    elif method == 'ModPoly':
        index = 0
        for spectra in tqdm(array,desc='Baseline (ModPoly)',leave=False):
            baseObj=BaselineRemoval(spectra)
            baselined_array[index,:] = baseObj.ModPoly(polynomial)
            index += 1
    elif method == 'IModPoly':
        index = 0
        for spectra in tqdm(array,desc='Baseline (IModPoly)',leave=False):
            baseObj=BaselineRemoval(spectra)
            baselined_array[index,:] = baseObj.IModPoly(polynomial)
            index += 1
    elif method == 'Zhang':
        index = 0
        for spectra in tqdm(array,desc='Baseline (Zhang)',leave=False):
            baseObj=BaselineRemoval(spectra)
            baselined_array[index,:] = baseObj.ZhangFit(polynomial)
            index += 1
    elif method == 'SWiMA':
        index = 0
        for spectra in tqdm(array,desc='Baseline (SWiMA)',leave=False):
            window = 3
            working_spectra = spectra
            for repeat in range(itterations):
                working_spectra = np.pad(working_spectra, 2, mode='reflect')
                smoothed_array = savgol_filter(working_spectra,window,0)
                a = working_spectra-smoothed_array
                a[a > 0] = 0
                working_spectra = a + smoothed_array
                window += 2
            baselined_array[index,:] = spectra - working_spectra[(window-3):-(window-3)]
            index += 1
    elif method == 'RollingBall':
        index = 0
        for spectra in tqdm(array,desc='Baseline (Rolling Ball)',leave=False):
            baselined_array[index,:] = spectra - rollingBall(spectra,ball_H,ball_W)
            index += 1
    elif method == 'Average':
        index = 0
        for spectra in tqdm(array,desc='Baseline (Moving Average)',leave=False):
            f = np.pad(spectra, (int(window_size/2)), 'constant', constant_values=(spectra[0], spectra[-1]))
            baselined_array[index,:] = spectra - moving_average(f, int(window_size))[0:len(spectra)]
            index += 1
    elif method == 'Manuel':
        index = 0
        for spectra in tqdm(array,desc='Baseline (Manuel)',leave=False):
            baselined_array[index,:] = spectra - manuelBaseline(spectra,mask,polyorder=manuelPolyOrder)
            index += 1
    if return_baseline == True:
        return np.transpose(baselined_array), np.transpose(array-baselined_array)
    else:
        return np.transpose(baselined_array)

def xAling(array, alingnemt_indexes = [895,901], wavenumbers=False, pre_normalise=False, aling_to_max=True):
    array = np.transpose(np.stack(array))
    alinged_array = np.zeros(np.shape(array))
    aline_list = []
    alingnemt_indexes_2 = alingnemt_indexes
    if type(wavenumbers) == np.ndarray:
        alingnemt_indexes_2[0] = np.absolute(wavenumbers - alingnemt_indexes[0]).argmin()
        alingnemt_indexes_2[1] = np.absolute(wavenumbers - alingnemt_indexes[1]).argmin()
        alingnemt_indexes_2 = sorted(alingnemt_indexes_2)
        alingnemt_indexe_1 = (alingnemt_indexes_2[0] * 10) + 30
        alingnemt_indexe_2 = (alingnemt_indexes_2[1] * 10) + 30
    else:
        alingnemt_indexe_1 = alingnemt_indexes[0] * 10
        alingnemt_indexe_2 = alingnemt_indexes[1] * 10
    index = 0
    copied_array = copy.deepcopy(array)
    f = interpolate.interp1d(range(len(array[:,0])), array[:,0], kind='quadratic')
    interp1 = f(np.arange(0, len(array[:,0])-1, 0.1))
    interp1 = np.pad(interp1, (30, 30), 'constant', constant_values=((array[:,0][0],array[:,0][-1])))
    #print(len(interp1[alingnemt_indexe_1:alingnemt_indexe_2]))
    alinged_array = np.zeros((len(interp1[30:len(interp1)-30]),np.shape(array)[1]))
    if pre_normalise == True:
        norm_array = normalise(copied_array, axis = 1, method = 'area', normalisation_indexs = [alingnemt_indexes_2[0],alingnemt_indexes_2[1]], wavenumbers=False)
        norm_array = np.transpose(norm_array)
        f = interpolate.interp1d(range(len(norm_array[:,0])), norm_array[:,0], kind='quadratic')
        interp1N = f(np.arange(0, len(norm_array[:,0])-1, 0.1))
        interp1N = np.pad(interp1N, (30, 30), 'constant', constant_values=((norm_array[:,0][0],norm_array[:,0][-1])))
        alinged_array = np.zeros((len(interp1N[30:len(interp1N)-30]),np.shape(norm_array)[1]))
        for spectra in tqdm(np.transpose(norm_array),desc='X Aling',leave=False):
            #print(np.shape(spectra))
            f = interpolate.interp1d(range(len(spectra)), spectra, kind='quadratic')
            interp2N = f(np.arange(0, len(spectra)-1, 0.1))
            interp2N = np.pad(interp2N, (30, 30), 'constant', constant_values=((spectra[0],spectra[-1])))
            f2 = interpolate.interp1d(range(len(spectra)), array[:,index], kind='quadratic')
            interp2 = f2(np.arange(0, len(spectra)-1, 0.1))
            interp2 = np.pad(interp2, (30, 30), 'constant', constant_values=((array[0,index],array[-1,index])))
            role_list = []
            for x in range(-30,30):
                role_list.append(sum(abs(interp1N[alingnemt_indexe_1:alingnemt_indexe_2] - np.roll(interp2N, x)[alingnemt_indexe_1:alingnemt_indexe_2])))
            aling_index = np.argmin(role_list)-30
            aline_list.append(aling_index)
            #print(np.shape(alinged_array))
            #print(np.shape(interp2))
            alinged_array[:,index] = np.roll(interp2,aling_index)[30:len(interp2)-30]
            index += 1
    else:
        for spectra in tqdm(np.transpose(array),desc='X Aling',leave=False):
            f = interpolate.interp1d(range(len(spectra)), spectra, kind='quadratic')
            interp2 = f(np.arange(0, len(spectra)-1, 0.1))
            interp2 = np.pad(interp2, (30, 30), 'constant', constant_values=((spectra[0],spectra[-1])))
            #print(str(np.argmax(interp1[alingnemt_indexe_1:alingnemt_indexe_2])) + ':' + str(np.argmax(interp2[alingnemt_indexe_1:alingnemt_indexe_2])))
            if aling_to_max == True:
                #plt.plot(interp2[alingnemt_indexe_1:alingnemt_indexe_2])
                #plt.show()
                max_index1 = np.argmax(interp1[alingnemt_indexe_1:alingnemt_indexe_2])
                max_index2 = np.argmax(interp2[alingnemt_indexe_1:alingnemt_indexe_2])
                alinged_array[:,index] = np.roll(interp2,(max_index1-max_index2))[30:len(interp2)-30]
            else:
                role_list = []
                for x in range(-30,30):
                    role_list.append(sum(abs(interp1[alingnemt_indexe_1:alingnemt_indexe_2] - np.roll(interp2, x)[alingnemt_indexe_1:alingnemt_indexe_2])))
                aling_index = np.argmin(role_list)-30
                aline_list.append(aling_index)
                alinged_array[:,index] = np.roll(interp2,aling_index)[30:len(interp2)-30]
            index += 1
    return alinged_array

def vectorInterp(WN,kind='linear'):
    f = interpolate.interp1d(range(len(WN)), WN, kind=kind)
    return f(np.arange(0, len(WN)-1, 0.1))

# Analysis
def applyRamanSignalMask(array,bool_arr):
    baseline_arr = copy.deepcopy(array)
    index = 0
    start = False
    end = False
    start_index = 0
    end_index = 0
    for boolian in bool_arr:
        if boolian != True:
            if start == False:
                pass
            else:
                end_value = array[index]
                end = True
                end_index = index
        else:
            if start != False:
                pass
            else:
                start_value = array[index]
                start = True
                start_index = index
        if start != False:
            if end != False:
                line = np.linspace(start_value, end_value, num=int(end_index-start_index))
                baseline_arr[start_index:end_index] = line
                start = False
                end = False
        index += 1
    return baseline_arr

def signalToNoise(matrix, axis=0, sqrt_signal=True, mask='none'):
    if mask == 'none':
        mean_val = abs(np.mean(matrix,axis=axis))
        std_val = np.std(matrix,axis=axis)
        zero_values = np.where(std_val<=0.00001)
        std_val[zero_values] = np.mean(std_val)
        #if sqrt_signal == True:
        #    StN = np.sqrt(mean_val) / std_val
        #else:
        #    StN = mean_val / std_val
        #return np.mean(StN)
        if sqrt_signal == True:
            StN = np.mean(np.sqrt(mean_val)) / np.mean(std_val)
        else:
            StN = np.mean(mean_val) / np.mean(std_val)
        return StN
    else:
        new_array = np.zeros(np.shape(matrix))
        index = 0
        for x in matrix:
            x2 = copy.deepcopy(x)
            y = x2 - applyRamanSignalMask(x,mask)
            y[y<0] = 0
            new_array[index,:] = y
            index += 1
        mean_val = abs(np.mean(new_array,axis=axis))
        std_val = np.std(new_array,axis=axis)
        zero_values = np.where(std_val<=0.00001)
        std_val[zero_values] = np.mean(std_val)
        #if sqrt_signal == True:
        #    StN = np.sqrt(mean_val) / std_val
        #else:
        #    StN = mean_val / std_val
        #return np.mean(StN)
        if sqrt_signal == True:
            StN = np.mean(np.sqrt(mean_val)) / np.mean(std_val)
        else:
            StN = np.mean(mean_val) / np.mean(std_val)
            #StN = (1000*np.mean(mean_val))/((1000*np.mean(std_val))**2)
        return StN

def signalToNoiseOfDataframe(dataframe, scale=True, subsample=False, subsample_size=30, subsample_repeats=20,
                             display=True, plot_lables=True, print_plot=True, sqrt_signal=True, mask='none'):
    SNR = {}
    if subsample == True:
        if scale == True:
            for (columnName, columnData) in dataframe.iteritems():
                try:
                    rng = default_rng()
                    rand_ints = rng.choice(np.shape(columnData.values)[0], size=subsample_size, replace=False)
                    Sigan_to_noise_ratio = signalToNoise(normalise(columnData.values[rand_ints],method = 'scale'),axis=1,sqrt_signal=sqrt_signal,mask=mask)
                    SNR[columnName] = [Sigan_to_noise_ratio]
                except:
                    Sigan_to_noise_ratio = None
                    SNR[columnName] = Sigan_to_noise_ratio
        else:
            for (columnName, columnData) in dataframe.iteritems():
                try:
                    rng = default_rng()
                    rand_ints = rng.choice(np.shape(columnData.values)[0], size=subsample_size, replace=False)
                    Sigan_to_noise_ratio = signalToNoise(np.stack(columnData.values[rand_ints]),axis=0,sqrt_signal=sqrt_signal,mask=mask)
                    SNR[columnName] = [Sigan_to_noise_ratio]
                except:
                    Sigan_to_noise_ratio = None
                    SNR[columnName] = Sigan_to_noise_ratio
        for x in range(subsample_repeats-1):
            if scale == True:
                for (columnName, columnData) in dataframe.iteritems():
                    try:
                        rng = default_rng()
                        rand_ints = rng.choice(np.shape(columnData.values)[0], size=subsample_size, replace=False)
                        Sigan_to_noise_ratio = signalToNoise(normalise(columnData.values[rand_ints],method = 'scale'),axis=1,sqrt_signal=sqrt_signal,mask=mask)
                        SNR[columnName].append(Sigan_to_noise_ratio)
                    except:
                        Sigan_to_noise_ratio = None
            else:
                for (columnName, columnData) in dataframe.iteritems():
                    try:
                        rng = default_rng()
                        rand_ints = rng.choice(np.shape(columnData.values)[0], size=subsample_size, replace=False)
                        Sigan_to_noise_ratio = signalToNoise(np.stack(columnData.values[rand_ints]),axis=0,sqrt_signal=sqrt_signal,mask=mask)
                        SNR[columnName].append(Sigan_to_noise_ratio)
                    except:
                        Sigan_to_noise_ratio = None
    else:
        if scale == True:
            for (columnName, columnData) in dataframe.iteritems():
                try:
                    Sigan_to_noise_ratio = signalToNoise(normalise(columnData.values,method = 'scale'),axis=1,sqrt_signal=sqrt_signal,mask=mask)
                except:
                    Sigan_to_noise_ratio = None
                SNR[columnName] = Sigan_to_noise_ratio
        else:
            for (columnName, columnData) in dataframe.iteritems():
                try:
                    Sigan_to_noise_ratio = signalToNoise(np.stack(columnData.values),axis=0,sqrt_signal=sqrt_signal,mask=mask)
                except:
                    Sigan_to_noise_ratio = None
                SNR[columnName] = Sigan_to_noise_ratio

    if display == True:
        if subsample == True:
            mean_dict = {}
            std_dict = {}
            for (columnName, columnData) in SNR.items():
                if columnData != None:
                    mean_dict[columnName] = np.mean(columnData)
                    std_dict[columnName] = np.std(columnData)
            plt.bar(range(len(mean_dict)),mean_dict.values(),yerr=std_dict.values());
            if plot_lables == True:
                plt.xticks(range(len(mean_dict)),[x for x in mean_dict.keys()],rotation = 90);
                plt.title('SRN for Different Steps in the Raman Processing')
                plt.xlabel('Processing Step')
                plt.ylabel('Signal to Noise Ratio')
            if print_plot == True:
                plt.show()
        else:
            mean_dict = {}
            for (columnName, columnData) in SNR.items():
                if columnData != None:
                    mean_dict[columnName] = columnData
            plt.bar(range(len(mean_dict)),mean_dict.values());
            if plot_lables == True:
                plt.xticks(range(len(mean_dict)),[x for x in mean_dict.keys()],rotation = 90);
                plt.title('SRN for Different Steps in the Raman Processing')
                plt.xlabel('Processing Step')
                plt.ylabel('Signal to Noise Ratio')
            if print_plot == True:
                plt.show()
    else:
        SNR_dict = {}
        for (columnName, columnData) in SNR.items():
            if columnData != None:
                SNR_dict[columnName] = columnData
        return SNR_dict


def prediction(classifier, X_test, y_test):
    count = 0
    correct = 0
    for testX, testY in zip(globals()[classifier].predict(X_test),y_test):
        count += 1
        if testX == testY:
            correct += 1
    return ((correct/count)*100)

def find_best_principal_components(pca_1,pca_2,axis=0):
    d = np.argsort(abs(np.mean(pca_1,axis=axis) - np.mean(pca_2,axis=axis))/(np.std(pca_1,axis=axis)+np.std(pca_2,axis=axis)))
    return [d[-1],d[-2],d[-3]]

def applyMachineLearingPredictors(array,classifier_lables,decomposition=False,number_of_components=10,
                                  plot_varian_ratio=False,CV=10,randomstate=0):
    correct      = {'lgr'  :  [],
                    'rcc'  :  [],
                    'per'  :  [],
                    'pac'  :  [],
                    'ann'  :  [],
                    #'brbm' :  [],
                    'lda'  :  [],
                    'qda'  :  [],
                    'bc'   :  [],
                    'rfs'  :  [],
                    'abc'  :  [],
                    'etc'  :  [],
                    'gbc'  :  [],
                    #'hgbc' :  [],
                    'gpc'  :  [],
                    'sgd'  :  [],
                    'lsvm' :  [],
                    'nsvm' :  [],
                    'knnu' :  [],
                    'knnd' :  []}

    array = np.stack(array)

    if decomposition == False:
        X = array
        y = classifier_lables
        X_train = X
    elif decomposition == 'PCA':
        X = array
        y = classifier_lables
        pca = PCA(n_components=number_of_components)
        pca.fit(X)
        X_train = pca.transform(X)
        list_RC = []
        if plot_varian_ratio == True:
            for inter in range(1,len(pca.explained_variance_ratio_)+1):
                list_RC.append(sum(pca.explained_variance_ratio_[0:inter])*100)
            plt.plot(range(1,len(list_RC)+1),list_RC);
            plt.title('Roman Cheplyaka Plot for PCA')
            plt.xlabel('Principal Componants')
            plt.ylabel('Explained Variance')
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.xticks(np.arange(1,len(list_RC)+1))
            plt.grid()
            plt.show()
    elif decomposition == 'ICA':
        X = array
        y = classifier_lables
        ica = FastICA(n_components=number_of_components,random_state=randomstate)
        ica.fit(X)
        X_train = ica.transform(X)
    elif decomposition == 'NMF':
        X = array
        y = classifier_lables
        nmf = NMF(n_components=number_of_components)
        nmf.fit(X)
        X_train = nmf.transform(X)
    elif decomposition == 'FA':
        X = array
        y = classifier_lables
        fa = FactorAnalysis(n_components=number_of_components)
        fa.fit(X)
        X_train = fa.transform(X)
    elif decomposition == 'IPCA':
        X = array
        y = classifier_lables
        ipca = IncrementalPCA(n_components=number_of_components)
        ipca.fit(X)
        X_train = ipca.transform(X)
    elif decomposition == 'KPCA':
        X = array
        y = classifier_lables
        kpca = KernelPCA(n_components=number_of_components)
        kpca.fit(X_train)
        X_train = kpca.transform(X)
    elif decomposition == 'LDAL':
        X = array
        y = classifier_lables
        ldal = LatentDirichletAllocation(n_components=number_of_components)
        ldal.fit(X)
        X_train = ldal.transform(X)
    elif decomposition == 'SPCA':
        X = array
        y = classifier_lables
        spca = SparsePCA(n_components=number_of_components)
        spca.fit(X)
        X_train = spca.transform(X)
    elif decomposition == 'TSVD':
        X = array
        y = classifier_lables
        tsvd = TruncatedSVD(n_components=number_of_components)
        tsvd.fit(X)
        X_train = tsvd.transform(X)
    elif decomposition == 'DL':
        X = array
        y = classifier_lables
        dl = DictionaryLearning(n_components=number_of_components)
        dl.fit(X)
        X_train = dl.transform(X)

    lgr = LogisticRegression()
    lgr.fit(X_train, y)
    rcc = RidgeClassifierCV()
    rcc.fit(X_train, y)
    per = Perceptron()
    per.fit(X_train, y)
    pac = PassiveAggressiveClassifier()
    pac.fit(X_train, y)
    lsvm = LinearSVC(random_state=0, tol=1e-5)
    lsvm.fit(X_train, y)
    nsvm = NuSVC()
    nsvm.fit(X_train, y)
    ann = MLPClassifier(hidden_layer_sizes=(50,50),activation='relu',learning_rate='adaptive',max_iter=10000)
    ann.fit(X_train, y)
    #brbm = BernoulliRBM()
    #brbm.fit(X_train, y)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y)
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y)
    rfs = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=0)
    rfs.fit(X_train, y)
    abc = AdaBoostClassifier()
    abc.fit(X_train, y)
    bc = BaggingClassifier()
    bc.fit(X_train, y)
    etc = ExtraTreesClassifier()
    etc.fit(X_train, y)
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y)
    #hgbc = HistGradientBoostingClassifier()
    #hgbc.fit(X_train, y)
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y)
    sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    sgd.fit(X_train, y)
    knnu =  neighbors.KNeighborsClassifier(30, weights='uniform')
    knnu.fit(X_train, y)
    knnd =  neighbors.KNeighborsClassifier(30, weights='distance')
    knnd.fit(X_train, y)

    for key in tqdm(correct.keys(), desc='Cross-Validating Models', leave=False):
        correct[key] = cross_val_score(eval(key), X_train, y, cv=CV, n_jobs=-1)
    return correct

def dispayCVResults(Result_dictionary,ylims=None):
    plt.bar(range(len(Result_dictionary)),
            [x.mean() for x in Result_dictionary.values()],
            align='center');
    plt.errorbar(range(len(Result_dictionary)),
                 [x.mean() for x in Result_dictionary.values()],
                 yerr=[x.std() for x in Result_dictionary.values()],
                 fmt='none',
                 c='k');
    plt.xticks(range(len(Result_dictionary)),
               [x for x in Result_dictionary.keys()]);
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Predictive Power of Different Modles for Seperating Raman Data')
    if ylims != None:
        plt.ylims(ylims)
    plt.show()

def plotSpectraByClass(data_frame,x_axis,column,spectra_ids,spetcra_ids_coulmn,print_plot=True,offset=0,
                       plot_labels=True,colours = ['k','r','b','g','m','y'],linewidth=1,axis_object=False,
                       mean_line_payload={'linestyle': '-','linewidth':1},
                       area_fill_payload={'linestyle': '-','linewidth':1,'alpha':0.3},
                       legend_payload={'loc':'best'}):
    index = 0
    if axis_object == False:
        plotinbg_handel = eval('plt')
    else:
        plotinbg_handel = axis_object
    for spectra_class in spectra_ids:
        plotinbg_handel.plot(x_axis,
                 np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))+offset,
                 c=colours[index], **mean_line_payload)
        plotinbg_handel.fill_between(x_axis,
                         np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))-np.transpose(np.std(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))+offset,
                         np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))+np.transpose(np.std(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))+offset,
                         facecolor=colours[index],**area_fill_payload)
        index += 1
    if plot_labels == True:
        if axis_object == False:
            plotinbg_handel.title('Spectra Seperated by Sample Class')
            plotinbg_handel.xlabel('Wavenumbers (cm$^{-1}$)')
            plotinbg_handel.ylabel('Intensity (AU)')
            plotinbg_handel.legend(spectra_ids, **legend_payload)
        else:
            plotinbg_handel.set_title('Spectra Seperated by Sample Class')
            plotinbg_handel.set_xlabel('Wavenumbers (cm$^{-1}$)')
            plotinbg_handel.set_ylabel('Intensity (AU)')
            plotinbg_handel.legend(spectra_ids, **legend_payload)
    plotinbg_handel.autoscale(enable=True, axis='x', tight=True)
    if print_plot == True:
        plt.show()

def plotDifferenceSpectra(data_frame,x_axis,column,spectra_ids,spetcra_ids_coulmn,
                          print_plot=True, offset=0, colour=['k'], plot_labels=True,
                          linewidth=1,axis_object=False, return_spectrum=False,
                          mean_line_payload={'linestyle': '-','linewidth':1},
                          zero_line_payload={'linestyle': '-','linewidth':1}):
    if axis_object == False:
        plotinbg_handel = eval('plt')
    else:
        plotinbg_handel = axis_object
    spectra_ids = [i for i in spectra_ids]
    plotinbg_handel.plot(x_axis,
             (np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_ids[0])][str(column)]),axis=0))-np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_ids[1])][str(column)]),axis=0)))+offset,
             c=colour[0],**mean_line_payload)
    plotinbg_handel.plot(x_axis,
             np.zeros(np.shape(np.stack(data_frame[str(column)]))[1])+offset,
             ('--' + colour[-1]),**zero_line_payload)
    if plot_labels == True:
        if axis_object == False:
            plotinbg_handel.title('Difference Spectra')
            plotinbg_handel.xlabel('Wavenumbers (cm$^{-1}$)')
            plotinbg_handel.ylabel('Intensity (AU)')
        else:
            plotinbg_handel.set_title('Difference Spectra')
            plotinbg_handel.set_xlabel('Wavenumbers (cm$^{-1}$)')
            plotinbg_handel.set_ylabel('Intensity (AU)')
    plotinbg_handel.autoscale(enable=True, axis='x', tight=True)
    if print_plot == True:
        plt.show()
    if return_spectrum == True:
        return np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_ids[0])][str(column)]),axis=0))-np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_ids[1])][str(column)]),axis=0))

def plotPCAByClass(data_frame,column,spectra_ids,spetcra_ids_coulmn,principal_components=10,
                   PCs_plot=(0,1),print_plot=True,return_eigenvalues=False,plot_ids=False,
                   colours=['k','r','b','g','m','y'],return_axis=False,plot_labels=True,
                   axis_object=False,payload={'marker':'o'}):
    if axis_object == False:
        plotinbg_handel = eval('plt')
    else:
        plotinbg_handel = axis_object
    index = 0
    pca = PCA(n_components=principal_components)
    indexes = []
    for ids in range(len(spectra_ids)):
        if ids == 0:
            X = np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_ids[ids])][str(column)])
            pre_indexes = [x for x in np.arange(np.shape(X)[0])]
            indexes.append(pre_indexes)
        else:
            pre_X = np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_ids[ids])][str(column)])
            X = np.concatenate((X, pre_X), axis=0)
            pre_indexes = [x for x in np.arange(pre_indexes[-1]+1,np.shape(pre_X)[0]+pre_indexes[-1]+1)]
            indexes.append(pre_indexes)
    if len(spectra_ids) == 1:
        indexes = [indexes]
    X_pca = pca.fit(X).transform(X)
    if plot_ids == False:
        for id_itter in spectra_ids:
            plotinbg_handel.scatter(X_pca[indexes[index],PCs_plot[0]],X_pca[indexes[index],PCs_plot[1]],c=colours[index],**payload)
            index += 1
    else:
        for id_itter in plot_ids:
            if id_itter == True:
                plotinbg_handel.scatter(X_pca[indexes[index],PCs_plot[0]],X_pca[indexes[index],PCs_plot[1]],c=colours[index],**payload)
                index += 1
            else:
                index += 1
    if plot_labels == True:
        if axis_object == False:
            plotinbg_handel.title('PCA of Spcetra by Sample Class')
            plotinbg_handel.xlabel('Principal component ' + str(PCs_plot[0]+1))
            plotinbg_handel.ylabel('Principal component ' + str(PCs_plot[1]+1))
            plotinbg_handel.legend(spectra_ids)
        else:
            plotinbg_handel.set_title('PCA of Spcetra by Sample Class')
            plotinbg_handel.set_xlabel('Principal component ' + str(PCs_plot[0]+1))
            plotinbg_handel.set_ylabel('Principal component ' + str(PCs_plot[1]+1))
            plotinbg_handel.legend(spectra_ids)
    if print_plot == True:
        plt.show()
    if return_axis == True:
        return X_pca, indexes
    if return_eigenvalues == True:
        return pca

def extractWavenumber(data_frame,wavenumber,column,spectra_ids,spetcra_ids_coulmn,wavenumbers):
    X = np.stack(data_frame[str(column)])
    WN = np.absolute(wavenumbers - wavenumber).argmin()
    wavenumber_list_master = []
    for spectra_class in spectra_ids:
        indexes = [int(x) for x in data_frame.index[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)].tolist()]
        wavenumber_list = list(X[indexes,WN])
        wavenumber_list_master.append(wavenumber_list)
    return wavenumber_list_master
