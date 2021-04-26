#inport modules
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

from scipy.ndimage.interpolation import shift
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.stats as stats
from scipy import interpolate

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

from BaselineRemoval import BaselineRemoval
warnings.filterwarnings("ignore")
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

def readArrayFromFile(file, sample_ID):
    # Opens a file and reads the content into an array of spectras and corrisponding wavnumbers
    # (can accept both single file names and lists of files)
    if type(file) == list:
        master_array = False
        master_sample_ID_list = []
        index = 0
        for file_active in file:
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
    elif type(file) == str:
        with open(file) as filecontent:
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

def smooth(array, method = 'Savitzky–Golay', window = 3, polynomial = 0, axis = 1, fourior_values = 3):
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
    
def normalise(array, axis = 1, method = 'max_within_range', normalisation_indexs = [890,910], wavenumbers=False, zero_min=False, custom_values = False):
    array = np.transpose(np.stack(array))
    if zero_min == True:
        array = array + abs(np.min(array))
    normalised_array = np.zeros(np.shape(array))
    normalisation_indexs_2 = normalisation_indexs
    if type(wavenumbers) == np.ndarray:
        normalisation_indexs_2[0] = np.absolute(wavenumbers - normalisation_indexs[0]).argmin()
        normalisation_indexs_2[1] = np.absolute(wavenumbers - normalisation_indexs[1]).argmin()
    normalisation_indexs_2 = sorted(normalisation_indexs_2)
    if method == 'scale':
        max_value = np.max(array)
        normalised_array = array / max_value
    else:
        for spectra in range(np.shape(array)[axis]):
            if method == 'max_within_range':
                max_value = max(array[normalisation_indexs_2[0]:normalisation_indexs_2[1],spectra])
                normalised_array[:,spectra] = array[:,spectra] / max_value
            elif method == 'max_whole_array':
                max_value = max(array[:,spectra])
                normalised_array[:,spectra] = array[:,spectra] / max_value
            elif method == 'singel_point':
                normalised_array[:,spectra] = array[:,spectra] / array[normalisation_indexs_2[0],spectra]
            elif method == 'area':
                max_value = sum(array[normalisation_indexs_2[0]:normalisation_indexs_2[1],spectra])
                normalised_array[:,spectra] = array[:,spectra] / max_value
            elif method == 'interp_area':
                f = interpolate.interp1d(range((normalisation_indexs_2[1]-normalisation_indexs_2[0])),
                                         array[normalisation_indexs_2[0]:normalisation_indexs_2[1],spectra],
                                         kind='quadratic')
                normalised_array[:,spectra] = array[:,spectra] / sum(f(np.arange(0,
                                                                    (normalisation_indexs_2[1]-normalisation_indexs_2[0])-1,
                                                                     0.1)))
            elif method == 'custom_values':
                normalised_array[:,spectra] = array[:,spectra] / custom_values[spectra]
        
    if method == 'area':
        max_value = np.max(normalised_array[normalisation_indexs_2[0]:normalisation_indexs_2[1],:])
        normalised_array = normalised_array / max_value
    elif method == 'interp_area':
        max_value = np.max(normalised_array[normalisation_indexs_2[0]:normalisation_indexs_2[1],:])
        normalised_array = normalised_array / max_value
    elif method == 'custom_values':
        max_value = np.max(normalised_array)
        normalised_array = normalised_array / max_value
    else:
        pass
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

def removeCosmicRaySpikes(array, threshold=5):
    array = np.transpose(np.stack(array))
    removed_spikes = []
    despiked_raman_array = copy.deepcopy(array)
    row = 0
    for x in tqdm(array, desc='Despike', leave=False):
        index = 0
        for y in x:
            if y > np.mean(x) + np.std(x):
                new_array = np.delete(x, index)
                if  y > np.mean(new_array) + threshold * np.std(new_array):
                    despiked_raman_array[row,index] = np.mean(new_array)
                    removed_spikes.append([row, y])
            index += 1
        row += 1
    return despiked_raman_array

def baselineALS(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
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

def baselineCorrection(array, method = 'ALS', lam=10**7, p=0.01, niter=10, fourior_values = 3, polynomial = 3, return_baseline = False):
    array = np.stack(array)
    baselined_array = np.zeros(np.shape(array))
    if method == 'ALS':
        index = 0
        for spectra in tqdm(array,desc='Baseline',leave=False):
            baselined_array[index,:] = spectra - baselineALS(spectra, lam=lam, p=p, niter=niter)
            index += 1
    elif method == 'FFT':
        index = 0
        for spectra in tqdm(array,desc='Baseline',leave=False):
            rft = np.fft.rfft(spectra)
            rft[:fourior_values] = 0
            baselined_array[index,:] = np.fft.irfft(rft)
            index += 1
    elif method == 'ModPoly':
        index = 0
        for spectra in tqdm(array,desc='Baseline',leave=False):
            baseObj=BaselineRemoval(spectra)
            baselined_array[index,:] = baseObj.ModPoly(polynomial)
            index += 1
    elif method == 'IModPoly':
        index = 0
        for spectra in tqdm(array,desc='Baseline',leave=False):
            baseObj=BaselineRemoval(spectra)
            baselined_array[index,:] = baseObj.IModPoly(polynomial)
            index += 1
    elif method == 'Zhang':
        index = 0
        for spectra in tqdm(array,desc='Baseline',leave=False):
            baseObj=BaselineRemoval(spectra)
            baselined_array[index,:] = baseObj.ZhangFit(polynomial)
            index += 1
    if return_baseline == True:
        return np.transpose(baselined_array), np.transpose(array-baselined_array)
    else:
        return np.transpose(baselined_array)

def xAling(array, alingnemt_indexes = [895,901], wavenumbers=False):
    array = np.transpose(np.stack(array))
    alinged_array = np.zeros(np.shape(array))
    aline_list = []
    alingnemt_indexes_2 = alingnemt_indexes
    if type(wavenumbers) == np.ndarray:
        alingnemt_indexes_2[0] = np.absolute(wavenumbers - alingnemt_indexes[0]).argmin()
        alingnemt_indexes_2[1] = np.absolute(wavenumbers - alingnemt_indexes[1]).argmin()
        alingnemt_indexes_2 = sorted(alingnemt_indexes_2)
        alingnemt_indexe_1 = alingnemt_indexes_2[0] * 10
        alingnemt_indexe_2 = alingnemt_indexes_2[1] * 10
    else:
        alingnemt_indexe_1 = alingnemt_indexes[0] * 10
        alingnemt_indexe_2 = alingnemt_indexes[1] * 10
    index = 0
    f = interpolate.interp1d(range(len(array[:,0])), array[:,0], kind='quadratic')
    interp1 = f(np.arange(0, len(array[:,0])-1, 0.1))
    interp1 = np.pad(interp1, (30, 30), 'constant', constant_values=((array[:,0][0],array[:,0][-1])))
    alinged_array = np.zeros((len(interp1[30:len(interp1)-30]),np.shape(array)[1]))
    #alinged_array[:,index] = interp1[30:len(interp1)-30]
    for spectra in tqdm(np.transpose(array),desc='X Aling',leave=False):
        f = interpolate.interp1d(range(len(spectra)), spectra, kind='quadratic')
        interp2 = f(np.arange(0, len(spectra)-1, 0.1))
        interp2 = np.pad(interp2, (30, 30), 'constant', constant_values=((spectra[0],spectra[-1])))
        role_list = []
        for x in range(-30,30):   
            role_list.append(sum(abs(interp1[alingnemt_indexe_1:alingnemt_indexe_2] - np.roll(interp2, x)[alingnemt_indexe_1:alingnemt_indexe_2])))
        aling_index = np.argmin(role_list)-30   
        aline_list.append(aling_index)
        alinged_array[:,index] = np.roll(interp2,aling_index)[30:len(interp2)-30]
        index += 1
    return alinged_array

def vectorInterp(WN):
    f = interpolate.interp1d(range(len(WN)), WN, kind='linear')
    return f(np.arange(0, len(WN)-1, 0.1))

# Analysis
def signalToNoise(matrix, axis=0, sqrt_signal=True):
    mean_val = abs(np.mean(matrix,axis=axis))
    std_val = np.std(matrix,axis=axis)
    zero_values = np.where(std_val<=0.00001)
    std_val[zero_values] = np.mean(std_val)
    if sqrt_signal == True:
        StN = np.sqrt(mean_val) / std_val
    else:
        StN = mean_val / std_val
    return np.mean(StN)

def signalToNoiseOfDataframe(dataframe, scale=True, subsample=False, subsample_size=30, subsample_repeats=20, display=True, plot_lables=True, print_plot=True):
    SNR = {}
    if subsample == True:
        if scale == True:
            for (columnName, columnData) in dataframe.iteritems():
                try:
                    rng = default_rng()
                    rand_ints = rng.choice(np.shape(columnData.values)[0], size=subsample_size, replace=False)
                    Sigan_to_noise_ratio = signalToNoise(normalise(columnData.values[rand_ints],method = 'scale'),axis=1)
                    SNR[columnName] = [Sigan_to_noise_ratio]
                except:
                    Sigan_to_noise_ratio = None
                    SNR[columnName] = Sigan_to_noise_ratio
        else:
            for (columnName, columnData) in dataframe.iteritems():
                try:
                    rng = default_rng()
                    rand_ints = rng.choice(np.shape(columnData.values)[0], size=subsample_size, replace=False)
                    Sigan_to_noise_ratio = signalToNoise(np.stack(columnData.values[rand_ints]),axis=0)
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
                        Sigan_to_noise_ratio = signalToNoise(normalise(columnData.values[rand_ints],method = 'scale'),axis=1)
                        SNR[columnName].append(Sigan_to_noise_ratio)
                    except:
                        Sigan_to_noise_ratio = None
            else:
                for (columnName, columnData) in dataframe.iteritems():
                    try:
                        rng = default_rng()
                        rand_ints = rng.choice(np.shape(columnData.values)[0], size=subsample_size, replace=False)
                        Sigan_to_noise_ratio = signalToNoise(np.stack(columnData.values[rand_ints]),axis=0)
                        SNR[columnName].append(Sigan_to_noise_ratio)
                    except:
                        Sigan_to_noise_ratio = None
    else:
        if scale == True:
            for (columnName, columnData) in dataframe.iteritems():
                try:
                    Sigan_to_noise_ratio = signalToNoise(normalise(columnData.values,method = 'scale'),axis=1)
                except:
                    Sigan_to_noise_ratio = None
                SNR[columnName] = Sigan_to_noise_ratio
        else:
            for (columnName, columnData) in dataframe.iteritems():
                try:
                    Sigan_to_noise_ratio = signalToNoise(np.stack(columnData.values),axis=0)
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

def applyMachineLearingPredictors(array,classifier_lables,decomposition=False,number_of_components=10,plot_varian_ratio=False,CV=10,randomstate=0):
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
        tsvd.fit(X_train)
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
        correct[key] = cross_val_score(eval(key), X_train, y, cv=CV)
    return correct

def dispayCVResults(Result_dictionary):
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
    plt.show()

def plotSpectraByClass(data_frame,x_axis,column,spectra_ids,spetcra_ids_coulmn,print_plot=True,offset=0,plot_lables=True,colours = ['k','r','b','g','m','y']):
    index = 0
    for spectra_class in spectra_ids:
        plt.plot(x_axis,
                 np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))+offset,
                 c=colours[index])
        plt.fill_between(x_axis,
                         np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))-np.transpose(np.std(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))+offset,
                         np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))+np.transpose(np.std(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))+offset,
                         facecolor=colours[index],alpha=0.3)
        index += 1
    if plot_lables == True:
        plt.title('Spectra Seperated by Sample Class')
        plt.xlabel('Wavenumbers (CM$^{-1}$)')
        plt.ylabel('Intencity (AU)')
        plt.legend(spectra_ids)
    plt.autoscale(enable=True, axis='x', tight=True)
    if print_plot == True:
        plt.show()
    
def plotDifferenceSpectra(data_frame,x_axis,column,spectra_ids,spetcra_ids_coulmn,
                          print_plot=True, offset=0, colour=['k'], plot_lables=True):
    spectra_ids = [i for i in spectra_ids]
    plt.plot(x_axis,
             (np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_ids[0])][str(column)]),axis=0))-np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_ids[1])][str(column)]),axis=0)))+offset,
             c=colour[0])
    plt.plot(x_axis,
             np.zeros(np.shape(np.stack(data_frame[str(column)]))[1])+offset,
             ('--' + colour[-1]))
    if plot_lables == True:
        plt.title('Difference Spectra')
        plt.xlabel('Wavenumbers (CM$^{-1}$)')
        plt.ylabel('Intencity (AU)')
    plt.autoscale(enable=True, axis='x', tight=True)
    if print_plot == True:
        plt.show()

def plotPCAByClass(data_frame,column,spectra_ids,spetcra_ids_coulmn,principal_components=10,PCs_plot=(0,1),print_plot=True,return_eigenvalues=False,colours = ['k','r','b','g','m','y']):
    index = 0
    pca = PCA(n_components=principal_components)
    X = np.stack(data_frame[str(column)])
    X_pca = pca.fit(X).transform(X)
    for spectra_class in spectra_ids:
        indexes = [int(x) for x in data_frame.index[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)].tolist()]
        plt.scatter(X_pca[indexes,PCs_plot[0]],X_pca[indexes,PCs_plot[1]],c=colours[index])
        index += 1
    plt.title('PCA of Spcetra by Sample Class')
    plt.xlabel('Principal component ' + str(PCs_plot[0]+1))
    plt.ylabel('Principal component ' + str(PCs_plot[1]+1))
    plt.legend(spectra_ids)
    if print_plot == True:
        plt.show()
    if return_eigenvalues == True:
        return pca
