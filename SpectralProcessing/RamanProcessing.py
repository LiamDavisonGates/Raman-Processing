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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from scipy import interpolate
from sklearn.model_selection import cross_val_score
from scipy import interpolate
from BaselineRemoval import BaselineRemoval

# Define all function used in the program

# Processing
def quickProcess(file_paths, sample_type):
    WN, array, sample_ID = readArrayFromFile(file_paths, sample_type)
    df = readArrayToDataFrame(array, 'Raw_array')
    df['Sample_type'] = sample_ID
    df = addColumnToDataFrame(df,
                              smooth(df['Raw_array'],
                                     method = 'FFT',
                                     fourior_values = 250),
                              'Smoothed_array')
    df = addColumnToDataFrame(df,
                              normalise(df['Smoothed_array'],
                                        method = 'interp_area',
                                        normalisation_indexs = (895,901)),
                              'Normalized_array')
    df = addColumnToDataFrame(df,
                              baselineCorrection(df['Normalized_array'],
                                                 lam=10**5),
                              'Baseline_corrected_array')
    df = addColumnToDataFrame(df,
                              removeCosmicRaySpikes(df['Baseline_corrected_array'],
                                                    threshold = 5),
                              'Despiked_array')
    df = addColumnToDataFrame(df,
                              normalise(df['Despiked_array'],
                                        method = 'interp_area',
                                        normalisation_indexs = (895,901)),
                              'Baseline_corrected_normalized_array')
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

def normalise(array, axis = 1, method = 'max_within_range', normalisation_indexs = [890,910], wavenumbers=False):
    array = np.transpose(np.stack(array))
    normalised_array = np.zeros(np.shape(array))
    normalisation_indexs_2 = normalisation_indexs
    if type(wavenumbers) == np.ndarray:
        normalisation_indexs_2[0] = np.absolute(wavenumbers - normalisation_indexs[0]).argmin()
        normalisation_indexs_2[1] = np.absolute(wavenumbers - normalisation_indexs[1]).argmin()
    normalisation_indexs_2 = sorted(normalisation_indexs_2)
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
    if method == 'area':
        max_value = np.max(normalised_array[normalisation_indexs_2[0]:normalisation_indexs_2[1],:])
        normalised_array = normalised_array / max_value
    elif method == 'interp_area':
        max_value = np.max(normalised_array[normalisation_indexs_2[0]:normalisation_indexs_2[1],:])
        normalised_array = normalised_array / max_value
    else:
        pass
    return normalised_array

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

def baselineCorrection(array, method = 'ALS', lam=10**7, p=0.01, niter=10, fourior_values = 3, polynomial = 3):
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
    return np.transpose(baselined_array)

def xAling(array, alingnemt_indexes = (895,901)):
    array = np.transpose(np.stack(array))
    alinged_array = np.zeros(np.shape(array))
    aline_list = []
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
def signalToNoise(matrix, axis=0):
    mean_val = abs(np.mean(matrix,axis=axis))
    std_val = np.std(matrix,axis=axis)
    zero_values = np.where(std_val==0)
    std_val[zero_values] = np.mean(std_val)
    StN = np.sqrt(mean_val) / std_val
    return np.mean(StN)

def signalToNoiseOfDataframe(dataframe, colums='All'):
    SNR = {}
    for (columnName, columnData) in dataframe.iteritems():
        try:
            Sigan_to_noise_ratio = signalToNoise(np.stack(columnData.values),axis=0)
        except:
            Sigan_to_noise_ratio = None
        SNR[columnName] = Sigan_to_noise_ratio
    return SNR
    
    
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

def applyMachineLearingPredictors(array,classifier_lables,principal_components=False,CV=10,test_size=0.2):
    correct_test = {'lgr'  :  [],
                    'ann'  :  [],
                    'lda'  :  [],
                    'qda'  :  [],
                    'rfs'  :  [],
                    'sgd'  :  [],
                    'lsvm' :  [],
                    'knnu' :  [],
                    'knnd' :  []}
    
    correct_train = {'lgr'  :  [],
                     'ann'  :  [],
                     'lda'  :  [],
                     'qda'  :  [],
                     'rfs'  :  [],
                     'sgd'  :  [],
                     'lsvm' :  [],
                     'knnu' :  [],
                     'knnd' :  []}
    
    array = np.stack(array)
    
    if PCA == False:
        X = array
        y = classifier_lables
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train_2 = X_train
        X_test_2 = X_test
    else:
        X = array
        y = classifier_lables
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        pca = PCA(n_components=principal_components)
        pca.fit(X_train)
        X_train_2 = pca.transform(X_train)
        X_test_2 = pca.transform(X_test)

    lgr = LogisticRegression()
    lgr.fit(X_train_2, y_train)
    lsvm = LinearSVC(random_state=0, tol=1e-5)
    lsvm.fit(X_train_2, y_train)
    ann = MLPClassifier(hidden_layer_sizes=(50,50))
    ann.fit(X_train_2, y_train)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_2, y_train)
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_2, y_train)
    rfs = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=0)
    rfs.fit(X_train_2, y_train)
    sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    sgd.fit(X_train_2, y_train)
    knnu =  neighbors.KNeighborsClassifier(30, weights='uniform')
    knnu.fit(X_train_2, y_train)
    knnd =  neighbors.KNeighborsClassifier(30, weights='distance')
    knnd.fit(X_train_2, y_train) 

    for key in tqdm(correct_train.keys(), desc='Cross-Validating Models', leave=False):
        correct_train[key] = cross_val_score(eval(key), X_train_2, y_train, cv=CV)
        correct_test[key] = cross_val_score(eval(key), X_test_2, y_test, cv=CV)
    return correct_train, correct_test

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

def plotSpectraByClass(data_frame,x_axis,column,spectra_ids,spetcra_ids_coulmn,print_plot=True):
    colours = ['k','r','b','g','m']
    index = 0
    for spectra_class in spectra_ids:
        plt.plot(x_axis,
                 np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0)),
                 c=colours[index])
        plt.fill_between(x_axis,
                         np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))-np.transpose(np.std(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0)),
                         np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0))+np.transpose(np.std(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_class)][str(column)]),axis=0)),
                         facecolor=colours[index],alpha=0.3)
        index += 1
    plt.title('Spectra Seperated by Sample Class')
    plt.xlabel('Wavenumbers (CM$^{-1}$)')
    plt.ylabel('Intencity (AU)')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend(spectra_ids)
    if print_plot == True:
        plt.show()
    
def plotDifferenceSpectra(data_frame,x_axis,column,spectra_ids,spetcra_ids_coulmn,print_plot=True):
    spectra_ids = [i for i in spectra_ids]
    plt.plot(x_axis,
             np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_ids[0])][str(column)]),axis=0))-np.transpose(np.mean(np.stack(data_frame[data_frame[str(spetcra_ids_coulmn)] == str(spectra_ids[1])][str(column)]),axis=0)),
             c='k')
    plt.plot(x_axis,
             np.zeros(np.shape(np.stack(data_frame[str(column)]))[1]),
             '--k')
    plt.title('Difference Spectra')
    plt.xlabel('Wavenumbers (CM$^{-1}$)')
    plt.ylabel('Intencity (AU)')
    plt.autoscale(enable=True, axis='x', tight=True)
    if print_plot == True:
        plt.show()

def plotPCAByClass(data_frame,column,spectra_ids,spetcra_ids_coulmn,principal_components=10,PCs_plot=(0,1),print_plot=True):
    colours = ['k','r','b','g','m']
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
    
#def analyisePipeline(data_frame,classifier_lables,PCA=False,principal_components=10):
#    for column in data_frame:
#        array = np.stack(data_frame[column])
#    
#        if PCA == False:
#            X = array
#            y = classifier_lables
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#            X_train_2 = X_train
#            X_test_2 = X_test
#        else:
#            X = array
#            y = classifier_lables
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#            pca = PCA(n_components=principal_components)
#            pca.fit(X_train)
#            X_train_2 = pca.transform(X_train)
#            X_test_2 = pca.transform(X_test)