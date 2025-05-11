#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from joblib import dump
import json
from itertools import product
from datetime import datetime
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler,RobustScaler,Normalizer,QuantileTransformer,LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score,roc_auc_score

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

from outputfunctions import addtocompressedfile,logmessage,saveimportancesincompressedfile,saveaccuracyindexincompressedfile

from sklearn.feature_selection import VarianceThreshold

def getcasestoexecute(cases_to_be_tested):
    executions = [dict(zip(cases_to_be_tested.keys(), values)) for values in itertools.product(*cases_to_be_tested.values())]
    numbered_executions = [{'sequence': i+1, **execution} for i, execution in enumerate(executions)]
    return (numbered_executions)

def adddatatoexecution(execution,parameters):
    # temporarity_index=parameters['General']['temporality_index']
    execution['analyzed_model']=str(execution['sequence']).zfill(2)+'-'+execution['complete_model']+'-'+execution['dimensionality_alter']+'-'+execution['preprocessing']+'-'+execution['balancing']+'-'+execution['optimization']+'-'+execution['behaviors_to_use']
    execution['output_confussion_matrix_image_file_name_train']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-'+execution['analyzed_model']+'-ConfussionMatrix-Train.png'
    execution['output_confussion_matrix_image_file_name_validation']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-'+execution['analyzed_model']+'-ConfussionMatrix-Validation.png'
    execution['output_confussion_matrix_image_file_name_test']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-'+execution['analyzed_model']+'-ConfussionMatrix-Test.png'
    execution['output_feature_importances_image_file_name']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-'+execution['analyzed_model']+'-FeatureImportances.png'
    execution['output_joblib_model_file_name']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-'+execution['analyzed_model']+'-Algorithm.joblib'
    execution['output_joblib_scaler_file_name']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-'+execution['analyzed_model']+'-Scaler.joblib'
    execution['output_keras_model_file_name']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-'+execution['analyzed_model']+'-Algorithm.keras'
    execution['output_xlsx_detail_file_name']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-'+execution['analyzed_model']+'-Detail.xlsx'
    execution['output_xlsx_pivot_table_file_name']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-'+execution['analyzed_model']+'-PivotTable.xlsx'
    execution['output_xlsx_accuracy_index_analysis_file_name']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-'+execution['analyzed_model']+'-AccuracyAnalysis.xlsx'
    execution['output_xlsx_feature_importances']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-'+execution['analyzed_model']+'-FeatureImportances.xlsx'
    execution['compressed_file_name']=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'.7z'
    execution['all_labels']=parameters['General']['all_classes']
    execution['images_resolution']=parameters['General']['images_resolution']
    return (execution)

def balancedata(X,y,execution):
    balancing=execution['balancing']
    behaviors_to_use=execution['behaviors_to_use']
    if (balancing=='BalUnder' and behaviors_to_use=='Train(AB+NB)'):
        sampler = RandomUnderSampler(sampling_strategy='auto')
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        X=X_resampled
        y=y_resampled
    elif (balancing=='BalOver' and behaviors_to_use=='Train(AB+NB)'):
        sampler = RandomOverSampler(sampling_strategy='auto')
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        X=X_resampled
        y=y_resampled
    elif (balancing=='NoBal' or behaviors_to_use!='Train(AB+NB)'):
        pass
    else:
        print(f'Unexpected balancing: {balancing}')
        exit()
    return (X,y)
    
def preprocessdata(input_X_train,input_X_validation,input_X_test,execution):
    preprocessing = execution['preprocessing']
    if preprocessing == 'StanScal':
        scaler = StandardScaler()
    elif preprocessing == 'RobuScal':
        scaler = RobustScaler()
    elif preprocessing == 'Normalize':
        scaler = Normalizer()
    elif preprocessing == 'QuanTran':
        scaler = QuantileTransformer(n_quantiles=50, output_distribution='normal')
    elif preprocessing == 'NoPreprocess':
        output_X_train=input_X_train
        output_X_validation=input_X_validation
        output_X_test=input_X_test
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(input_X_train.shape[1]) 
        scaler.scale_ = np.ones(input_X_train.shape[1])
       
        return output_X_train, output_X_validation,output_X_test,scaler
    else:
        print(f'Unexpected Preprocessing: {preprocessing}')
        output_X_train=input_X_train
        output_X_validation=input_X_validation
        output_X_test=input_X_test
        return output_X_train, output_X_validation, output_X_test,scaler
    output_X_train=pd.DataFrame(scaler.fit_transform(input_X_train) , columns=input_X_train.columns)
    output_X_validation=pd.DataFrame(scaler.transform(input_X_validation) , columns=input_X_validation.columns)
    output_X_test=pd.DataFrame(scaler.transform(input_X_test) , columns=input_X_test.columns)
    return output_X_train, output_X_validation, output_X_test,scaler

def selectbehaviortypes(X,y,execution):
    behaviors_to_use=execution['behaviors_to_use']
    if (behaviors_to_use=='Train(AB+NB)'):
        return (X,y)
    elif (behaviors_to_use=='Train(NB)'):
        columns_X=X.columns
        columns_y=y.columns
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        dataframe = pd.concat([X, y], axis=1)
        dataframe = dataframe[dataframe['data_class']=='NB']
        X = dataframe[columns_X]
        y = dataframe[columns_y]
        return (X,y)
    elif (behaviors_to_use=='Train(AB)'):
        columns_X=X.columns
        columns_y=y.columns
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        dataframe = pd.concat([X, y], axis=1)
        dataframe = dataframe[dataframe['data_class']=='AB']
        X = dataframe[columns_X]
        y = dataframe[columns_y]
        return (X,y)
                  
    elif (behaviors_to_use=='Train(AB)'):
        columns_X=X.columns
        columns_y=y.columns
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        dataframe = pd.concat([X, y], axis=1)
        dataframe = dataframe[dataframe['data_class']=='AB']
        X = dataframe[columns_X]
        y = dataframe[columns_y]        
        return (X,y)
    else:
       print(f'Unexpected behaviors_to_use: {behaviors_to_use}')
       exit()
       
       
def getparametercombinations(execution,parameters):
    complete_model=execution['complete_model']
    optimization=execution['optimization']
    if (optimization=='NoOpt'):
        cases_to_test=[parameters['Model'][complete_model]['default_parameters'],]
        
    elif (optimization=='CrossValScore'):
        parameter_values=parameters['Model'][complete_model]['optimization_parameters']
        param_combinations = product(*parameter_values.values())
        cases_to_test = []
        for combination in param_combinations:
            case_to_test = dict(zip(parameter_values.keys(), combination))
            cases_to_test.append(case_to_test)
    else: 
        print(f'Unexpected optimization value: {optimization}')
        exit()
    return(cases_to_test)

def getalgorithm(execution):
    complete_model=execution['complete_model']
    if (complete_model=='ClassKNN'):
        algorithm=KNeighborsClassifier()
    elif (complete_model=='ClassSVM'):
        algorithm=SVC() 
    elif (complete_model=='ClassNB'):
        algorithm=GaussianNB() 
    elif (complete_model=='ClassLR'):
        algorithm=LogisticRegression() 
    elif (complete_model=='ClassDT'):
        algorithm=DecisionTreeClassifier() 
    elif (complete_model=='ClassRF'):
        algorithm=RandomForestClassifier()
    elif (complete_model=='ClassGBM'):
        algorithm=GradientBoostingClassifier()
    elif (complete_model=='ClassABC'):
        algorithm=AdaBoostClassifier()
    elif (complete_model=='ClassXBG'):
        algorithm=XGBClassifier()
    elif (complete_model=='Detect-OCSVM'):
        algorithm=OneClassSVM()
    elif (complete_model=='Detect-IF'):
        algorithm=IsolationForest()
    elif (complete_model=='Detect-LOF'):
        algorithm=LocalOutlierFactor()
    elif (complete_model=='ClassANN'):
        algorithm= Sequential()
        algorithm.add(Dense(64, activation='relu', input_shape=(execution['num_features'],)))
        algorithm.add(Dense(64, activation='relu'))  
        algorithm.add(Dense(1, activation='sigmoid'))
        algorithm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    elif (complete_model=='ClassDNN'):
        algorithm = Sequential()

        algorithm.add(Dense(256, input_shape=(execution['num_features'],), kernel_regularizer=l2(0.001)))
        algorithm.add(LeakyReLU(alpha=0.1))
        algorithm.add(Dropout(0.1))
    
        algorithm.add(Dense(128, kernel_regularizer=l2(0.001)))
        algorithm.add(LeakyReLU(alpha=0.1))
        algorithm.add(Dropout(0.1))
    
        algorithm.add(Dense(64, kernel_regularizer=l2(0.001)))
        algorithm.add(LeakyReLU(alpha=0.1))
        algorithm.add(Dropout(0.1))
    
        algorithm.add(Dense(32, kernel_regularizer=l2(0.001)))
        algorithm.add(LeakyReLU(alpha=0.1))
        algorithm.add(Dropout(0.1))
    
        algorithm.add(Dense(1, activation='sigmoid'))
    
        optimizer = Adam(learning_rate=0.001)
        algorithm.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                                                                                 

    else: 
        print(f'Unexpected complete_model: {complete_model}')
        exit()
    
    # Debug information
    
    # model_name = algorithm.__class__.__name__+'.json'
    # model_parameters = algorithm.get_params()
    # model_parameters_json = json.dumps(model_parameters, indent=4) 
    # with open(model_name, "w") as file:
    #     file.write(model_parameters_json)
    
    return(algorithm)
        

def generateandsaveconfussionmatrix(y_test_array,y_pred_array,algorithm,scaler,best_model_parameters,execution,train_validation_test):
    analyzed_model=execution['analyzed_model']
    if train_validation_test=='Train':
        output_confussion_matrix_image_file_name=execution['output_confussion_matrix_image_file_name_train']
    elif train_validation_test=='Validation':
        output_confussion_matrix_image_file_name=execution['output_confussion_matrix_image_file_name_validation']
    elif train_validation_test=='Test':
        output_confussion_matrix_image_file_name=execution['output_confussion_matrix_image_file_name_test']
    
    else:
        print(f'Unexpected value {train_validation_test}')
    
    output_joblib_model_file_name=execution['output_joblib_model_file_name']
    output_joblib_scaler_file_name=execution['output_joblib_scaler_file_name']
    output_keras_model_file_name=execution['output_keras_model_file_name']
    compressed_file_name=execution['compressed_file_name']
    all_labels=execution['all_labels']
    complete_model=execution['complete_model']
    
    confussion_matrix=confusion_matrix(y_test_array,y_pred_array,labels=all_labels)
    precision = round(precision_score(y_test_array, y_pred_array, average='macro', zero_division=0), 4)
    recall = round(recall_score(y_test_array, y_pred_array, average='macro', zero_division=0), 4)
    accuracy = round(accuracy_score(y_test_array, y_pred_array), 4)
    f1score = round(f1_score(y_test_array, y_pred_array, average='macro', zero_division=0), 4)
    tn, fp, fn, tp = confussion_matrix.ravel()
    score=round((fp+fn)/max(f1score,0.1),4)
    label_encoder = LabelEncoder()
    y_test_array_encoded = label_encoder.fit_transform(y_test_array.values.ravel())
    y_pred_array_encoded = label_encoder.transform(y_pred_array.values.ravel())
    roc_auc=round(roc_auc_score(y_test_array_encoded,y_pred_array_encoded,average='macro'),4)
    samples=y_test_array.shape[0]
    parameters_string = '|'.join([
        f"{key}={value}" 
        for key, value in best_model_parameters.items()
        ])
    matrix_title = f'{analyzed_model}\n{parameters_string}\nPrec={round(precision,4)}|Rec={round(recall,4)}|Acc={round(accuracy,4)}|F1Sc={round(f1score,4)}|Rauc={roc_auc}|CustSc={round(score,4)}\nSamples={samples}|TP={tp}|TN={tn}|FP={fp}|FN={fn}'
    matrix_to_show = metrics.ConfusionMatrixDisplay(confusion_matrix = confussion_matrix, display_labels = all_labels)
    plt.figure(figsize=(14, 6)) 
    matrix_to_show.plot(xticks_rotation='horizontal')
    matrix_to_show.ax_.set_title(matrix_title)
    plt.tight_layout()
    plt.autoscale()
    plt.gcf().subplots_adjust(top=0.8,right=0.9)
    plt.savefig(output_confussion_matrix_image_file_name, dpi=execution['images_resolution'])
    plt.close()
    if (complete_model=='ClassANN'):
        save_model(algorithm, output_keras_model_file_name)
        dump(scaler,output_joblib_scaler_file_name) 

        addtocompressedfile(compressed_file_name, output_confussion_matrix_image_file_name)
        addtocompressedfile(compressed_file_name, output_keras_model_file_name)
        addtocompressedfile(compressed_file_name, output_joblib_scaler_file_name)
    elif (complete_model=='ClassDNN'):
        save_model(algorithm, output_keras_model_file_name)
        dump(scaler,output_joblib_scaler_file_name) 

        addtocompressedfile(compressed_file_name, output_confussion_matrix_image_file_name)
        addtocompressedfile(compressed_file_name, output_keras_model_file_name)
        addtocompressedfile(compressed_file_name, output_joblib_scaler_file_name)
    else: 
        dump(algorithm, output_joblib_model_file_name)
        dump(scaler,output_joblib_scaler_file_name)        
        addtocompressedfile(compressed_file_name, output_confussion_matrix_image_file_name)
        addtocompressedfile(compressed_file_name, output_joblib_model_file_name)
        addtocompressedfile(compressed_file_name, output_joblib_scaler_file_name)

    return(score,precision,recall,accuracy,f1score,roc_auc,tp,tn,fp,fn,matrix_title)

def optimizemodel(execution,parameters,X,y):
    # Create diferent combinations
    cases_to_test=getparametercombinations(execution,parameters)
    optimization=execution['optimization']
    y_converted=convertclassestexttonum(y,execution)
    X_array=X.to_numpy()
    y_array=y_converted.values.ravel()
    
    if (optimization=='NoOpt'):
        best_model_parameters=cases_to_test[0]
    elif (optimization=='CrossValScore'):
        print(f'Optimizing model - Data dimensions: {X_array.shape}')
        results=[]
        for case_to_test in cases_to_test:
            try:
                algorithm = getalgorithm(execution)
                algorithm.set_params(**case_to_test)
                cross_val_results = cross_val_score(algorithm, X_array, y_array, cv=5, n_jobs=-1, verbose=1)
                average_results = -np.mean(cross_val_results)
                results.append((average_results, case_to_test))
            except Exception as exception:
                print('-------------------------------------')
                print('-------------------------------------')
                print('-------------------------------------')
                print('-------------------------------------')
                print(f'Exception found for {case_to_test}: {exception}')
                print('-------------------------------------')
                print('-------------------------------------')
                print('-------------------------------------')
                print('-------------------------------------')
                results.append((1e18, case_to_test))
        best_case = min(results, key=lambda x: x[0])
        best_model_parameters=best_case[1]
    else: 
        print(f'Unexpected value optimization: {optimization}')
        exit()
    return(best_model_parameters)

def alterdatadimensionality(data_train,data_test,execution):
    parts = execution['dimensionality_alter'].split('-')
    if len(parts) == 2:
        dimensionality_alter = parts[0]
        components = int(parts[1])
    else:
        dimensionality_alter = parts[0]
        components = None
    
    if (dimensionality_alter=='NoDimAlter'):
        return(data_train,data_test)
    else:
        component_names = [f'pca{i}' for i in range(1, components + 1)]
        scaler = StandardScaler()
        data_train_scaled = scaler.fit_transform(data_train)
        data_test_scaled = scaler.transform(data_test)
          
        if (dimensionality_alter=='PcaChg'):
            pca = PCA(n_components=components)
            pca.fit(data_train_scaled)
            data_train_pca=pca.transform(data_train_scaled)
            data_test_pca=pca.transform(data_test_scaled)
            data_train_out = pd.DataFrame(data=data_train_pca, columns=component_names) 
            data_test_out = pd.DataFrame(data=data_test_pca, columns=component_names)
            return(data_train_out,data_test_out)
        
        elif (dimensionality_alter=='PcaAdd'):
            pca = PCA(n_components=components)
            pca.fit(data_train_scaled)
            data_train_pca=pca.transform(data_train_scaled)
            data_test_pca=pca.transform(data_test_scaled)
            data_train_out = pd.DataFrame(data=data_train_pca, columns=component_names) 
            data_test_out = pd.DataFrame(data=data_test_pca, columns=component_names)
            data_train = data_train.reset_index(drop=True)
            data_train_out = data_train_out.reset_index(drop=True)
            data_train_out=pd.concat([data_train, data_train_out], axis=1)
            data_test = data_test.reset_index(drop=True)
            data_test_out = data_test_out.reset_index(drop=True)
            data_test_out=pd.concat([data_test, data_test_out], axis=1)
            return(data_train_out,data_test_out)
        else:
            print('Unexpected dimensionality alter mode: {dimensionality_alter}')
        

def convertclassestexttonum(y_train, execution):
    complete_model=execution['complete_model']
    behaviors_to_use = execution['behaviors_to_use']
    if complete_model in ['ClassANN','ClassDNN','ClassXBG']:
        if behaviors_to_use == 'Train(AB)':
            y_train['data_class'] = y_train['data_class'].replace({'NB': 0, 'AB': 1})
            return y_train
        elif behaviors_to_use in ['Train(NB)', 'Train(AB+NB)']:
            y_train['data_class'] = y_train['data_class'].replace({'NB': 1, 'AB': 0})
            return y_train
        else:
            raise ValueError(f'Unexpected behaviors to use: {behaviors_to_use}')
    elif complete_model.startswith('Detect'):
        if behaviors_to_use == 'Train(AB)':
            y_train['data_class'] = y_train['data_class'].replace({'NB': -1, 'AB': 1})
            return y_train
        elif behaviors_to_use in ['Train(NB)', 'Train(AB+NB)']:
            y_train['data_class'] = y_train['data_class'].replace({'NB': 1, 'AB': -1})
            return y_train
        else:
            raise ValueError(f'Unexpected behaviors to use: {behaviors_to_use}')
    elif complete_model.startswith('Class'):
        
        return y_train
    else:
        raise ValueError(f'Unexpected model type: {complete_model}')

        
def convertclassesnumtotext(y_pred,execution):
    
    if y_pred.columns[0] != 'data_class':
        y_pred.columns = ['data_class']

    complete_model=execution['complete_model']
    behaviors_to_use = execution['behaviors_to_use']
    if complete_model in ['ClassANN','ClassDNN','ClassXBG']:
        y_pred=y_pred.round()
        if behaviors_to_use == 'Train(AB)':
            y_pred['data_class'] = y_pred['data_class'].replace({0:'NB',1:'AB'})
            return y_pred
        elif behaviors_to_use in ['Train(NB)', 'Train(AB+NB)']:
            y_pred['data_class'] = y_pred['data_class'].replace({1:'NB',0:'AB'})
            return y_pred
        else:
            raise ValueError(f'Unexpected behaviors to use: {behaviors_to_use}')
    elif complete_model.startswith('Detect'):
        if behaviors_to_use == 'Train(AB)':
            y_pred['data_class'] = y_pred['data_class'].replace({-1:'NB',1:'AB'})
            return y_pred
        elif behaviors_to_use in ['Train(NB)', 'Train(AB+NB)']:
            y_pred['data_class'] = y_pred['data_class'].replace({1:'NB',-1:'AB'})
            return y_pred
        else:
            raise ValueError(f'Unexpected behaviors to use: {behaviors_to_use}')
    elif complete_model.startswith('Class'):
        return y_pred
    else:
        raise ValueError(f'Unexpected model type: {complete_model}')


def predictoutput(X_test,algorithm,execution):
    X_test_array=X_test.to_numpy()
    X_test_array = np.array(X_test, dtype=np.float32)
    y_pred_array=algorithm.predict(X_test_array)
    y_pred = pd.DataFrame(y_pred_array, columns=['data_class'])
    return (y_pred)

    
def trainmodel(X,y,best_model_parameters,execution):
    complete_model=execution['complete_model']
    y_converted=convertclassestexttonum(y,execution)
    X_array=X.to_numpy()
    y_array=y_converted.values.ravel()
    print(f'Training model - Data dimensions: {X_array.shape}')
    
    if (complete_model=='ClassANN'):
        X_train_array, X_val_array, y_train_array, y_val_array = train_test_split(X_array, y_array, test_size=0.2)
        algorithm = getalgorithm(execution)
        X_train_array = np.array(X_train_array, dtype=np.float32)
        y_train_array = np.array(y_train_array, dtype=np.float32)
        X_val_array = np.array(X_val_array, dtype=np.float32)
        y_val_array = np.array(y_val_array, dtype=np.float32)
        algorithm.fit(X_train_array, y_train_array, epochs=20, batch_size=32, validation_data=(X_val_array, y_val_array))
    elif (complete_model=='ClassDNN'):
        X_train_array, X_val_array, y_train_array, y_val_array = train_test_split(X_array, y_array, test_size=0.2)
        algorithm = getalgorithm(execution)
        X_train_array = np.array(X_train_array, dtype=np.float32)
        y_train_array = np.array(y_train_array, dtype=np.float32)
        X_val_array = np.array(X_val_array, dtype=np.float32)
        y_val_array = np.array(y_val_array, dtype=np.float32)
        early_stopping = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
        algorithm.fit(X_train_array, y_train_array, epochs=100, batch_size=64, callbacks=[early_stopping], validation_data=(X_val_array, y_val_array))
    else:
        algorithm = getalgorithm(execution)
        algorithm.set_params(**best_model_parameters)
        algorithm.fit(X_array,y_array)
    return(algorithm)
    
    
        
def analyzedata(X_train,X_validation,X_test,y_train,y_validation,y_test,meta_train,meta_validation,meta_test,execution,parameters):
    
    
    last_time=datetime.now()

    # Preprocess data
    
    last_time=logmessage('Preprocess data - Start',last_time,parameters)

    # # Get train and test data
    
    # X_train,X_test,y_train,y_test=gettrainandtestdata(known_data,unknown_data,known_class,unknown_class,execution)
    
    # Random class balancing  
    
    X_train,y_train=balancedata(X_train,y_train,execution)

    # Dimensionalyty alter - Falta adaptar a Validation

    X_train,X_test=alterdatadimensionality(X_train,X_test,execution)

    # Preprocess data
    
    X_train,X_validation,X_test,scaler=preprocessdata(X_train,X_validation,X_test,execution)
    
    # Select behavior to use
    
    X_train_all_behaviors=X_train
    y_train_all_behaviors=y_train
    
    X_train,y_train=selectbehaviortypes(X_train,y_train,execution)
    
    last_time=logmessage('Preprocess data - End',last_time,parameters)
    
    # Optimize model - For future revision
    
    last_time=logmessage('Optimization model - Start',last_time,parameters)
    best_model_parameters=optimizemodel(execution,parameters,X_validation,y_validation)

    last_time=logmessage('Optimization model - End',last_time,parameters)

    # Train model
    
    last_time=logmessage('Train model - Start',last_time,parameters)
    
    algorithm=trainmodel(X_train,y_train,best_model_parameters,execution)
        
    last_time=logmessage('Train model - End',last_time,parameters)
    
    # Predict 
    
    last_time=logmessage('Predict data - Start',last_time,parameters)
    y_train_pred=predictoutput(X_train_all_behaviors,algorithm,execution)
    y_validation_pred=predictoutput(X_validation,algorithm,execution)
    y_test_pred=predictoutput(X_test,algorithm,execution)
    
    last_time=logmessage('Predict data - End',last_time,parameters)
    
   # Evaluate model - Generate confussion matrix
   
    last_time=logmessage('Evaluate model - Start',last_time,parameters)
    
    
    #### For future revision
    
    y_train_pred=convertclassesnumtotext(y_train_pred,execution)
    y_train=convertclassesnumtotext(y_train,execution)
    y_train_all_behaviors=convertclassesnumtotext(y_train_all_behaviors,execution)
    y_validation_pred=convertclassesnumtotext(y_validation_pred,execution)
    y_validation=convertclassesnumtotext(y_validation,execution)
    y_test_pred=convertclassesnumtotext(y_test_pred,execution)
    y_test=convertclassesnumtotext(y_test,execution)

    
    score_train,precision_train,recall_train,accuracy_train,f1score_train,roc_auc_train,tp_train,tn_train,fp_train,fn_train,matrix_title_train=generateandsaveconfussionmatrix(y_train_all_behaviors,y_train_pred,algorithm,scaler,best_model_parameters,execution,'Train')

    
    score_validation,precision_validation,recall_validation,accuracy_validation,f1score_validation,roc_auc_validation,tp_validation,tn_validation,fp_validation,fn_validation,matrix_title_validation=generateandsaveconfussionmatrix(y_validation,y_validation_pred,algorithm,scaler,best_model_parameters,execution,'Validation')
    score_test,precision_test,recall_test,accuracy_test,f1score_test,roc_auc_test,tp_test,tn_test,fp_test,fn_test,matrix_title_test=generateandsaveconfussionmatrix(y_test,y_test_pred,algorithm,scaler,best_model_parameters,execution,'Test')
    
    
    strings_train = matrix_title_train.split('\n', 3)
    strings_validation = matrix_title_validation.split('\n', 3)
    strings_test = matrix_title_test.split('\n', 3)
    
    last_time=logmessage('Evaluate model - End',last_time,parameters)
    
    # Evaluate model - Generate importances matrix
    
    last_time=logmessage('Generate and save importances graph - Start',last_time,parameters)
    saveimportancesincompressedfile(algorithm,X_train.columns,execution,matrix_title_test,percentage=97)
    last_time=logmessage('Generate and save importances graph - Start',last_time,parameters)
    
    # Save evaluation results to excel
    if parameters['General']['savevaluationresults']=='Yes': 
        last_time=logmessage('Save evaluation results to excel - Start',last_time,parameters)
        saveaccuracyindexincompressedfile(y_train,y_train_pred,meta_train,y_validation,y_validation_pred,meta_validation,y_test,y_test_pred,meta_test,execution)
        # saveaccuracyindexincompressedfileold(y_train,y_train_pred,meta_train,y_validation,y_validation_pred,meta_validation,y_test,y_test_pred,meta_test,execution)
        # saveevaluationresultsincompressedfile(y_test,y_test_pred,meta_test,execution)
        last_time=logmessage('Save evaluation results to excel - End',last_time,parameters)
    else: 
        last_time=logmessage('Save evaluation results - The saving of the evaluation results was skipped due to the configuration',last_time,parameters)

    return (score_train,precision_train,recall_train,accuracy_train,f1score_train,roc_auc_train,tp_train,tn_train,fp_train,fn_train,strings_train,score_validation,precision_validation,recall_validation,accuracy_validation,f1score_validation,roc_auc_validation,tp_validation,tn_validation,fp_validation,fn_validation,strings_validation,score_test,precision_test,recall_test,accuracy_test,f1score_test,roc_auc_test,tp_test,tn_test,fp_test,fn_test,strings_test,y_train_pred,y_validation_pred,y_test_pred)


def lowvariancecolumns(input_data, threshold=0.01):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(input_data.select_dtypes(include=[float, int]))
    low_variance_columns = [
        column for column, var in zip(input_data.select_dtypes(include=[float, int]).columns, selector.variances_)
        if var < threshold
    ]
    return low_variance_columns


def removecolumns(input_data, columns_to_remove):

    columns_to_remove = [col for col in columns_to_remove if col in input_data.columns]
    output_data = input_data.drop(columns=columns_to_remove)
    
    return output_data
