#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import json
from os import remove
import os

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib

def confussionmatrixclassification(row):
    if row['actual_class'] == row['predicted_class']:
        if row['predicted_class'] == 'AB':
            return 'TP'  
        elif row['predicted_class'] == 'NB':
            return 'TN'  
    else:
        if row['predicted_class'] == 'AB':
            return 'FP'  
        elif row['predicted_class'] == 'NB':
            return 'FN'

def vectorizedconfussionmatrixclassification(actual_class, predicted_class):
    conditions = [
        (actual_class == 'AB') & (predicted_class == 'AB'),
        (actual_class == 'NB') & (predicted_class == 'NB'),
        (actual_class == 'NB') & (predicted_class == 'AB'),
        (actual_class == 'AB') & (predicted_class == 'NB')
    ]
    choices = ['TP', 'TN', 'FP', 'FN']
    return np.select(conditions, choices, default='Unknown')
    
def addtocompressedfile(compressed_file_name, output_file_name,folder=None,erase='Yes'):
    if (folder!=None):
        with open('/dev/null', 'w') as null_file:
            subprocess.run(['7z', 'a', '-aoa', compressed_file_name, output_file_name, '-mx9', '-t7z', '-bsp0','-o' + folder], stdout=null_file)
            if (erase=='Yes' and os.path.exists(output_file_name)):
                    remove(output_file_name)
    else:
        with open('/dev/null', 'w') as null_file:
            subprocess.run(['7z', 'a', '-aoa', compressed_file_name, output_file_name, '-mx9', '-t7z', '-bsp0'], stdout=null_file)
            if (erase=='Yes' and os.path.exists(output_file_name)):
                    remove(output_file_name)
            
def savecodeincompressedfile(parameters):
    code_to_save=parameters['General']['code_to_save']
    compressed_file_name=parameters['General']['compressed_file_name']
    code_used_compressed_file_name=parameters['General']['code_used_compressed_file_name']
    for n1,code_file_name in enumerate(code_to_save):
        with open('/dev/null', 'w') as null_file:
            subprocess.run(['7z', 'a',code_used_compressed_file_name,code_file_name, '-mx9', '-t7z', '-bsp0'], stdout=null_file)
    with open('/dev/null', 'w') as null_file:
        subprocess.run(['7z', 'a',compressed_file_name,code_used_compressed_file_name,'-mx9', '-t7z', '-bsp0'], stdout=null_file)
        remove(code_used_compressed_file_name)  


def generatesha256file(input_file,output_file):
    
    sha256_hash = hashlib.sha256()
    with open(input_file, "rb") as file:
        for block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(block)
    output_hash = sha256_hash.hexdigest()

    with open(output_file, "w") as output:
        output.write(output_hash)

def savedataincompressedfile(train_data,train_class,train_metadata,validation_data,validation_class,validation_metadata,test_data,test_class,test_metadata,parameters):
    compressed_file_name=parameters['General']['compressed_file_name']
    data_used_compressed_file_name=parameters['General']['data_used_compressed_file_name']
    train_data_json_file_name=parameters['General']['results_files_path']+'TrainData'+'.json'
    train_class_json_file_name=parameters['General']['results_files_path']+'TrainClass'+'.json'
    train_metadata_json_file_name=parameters['General']['results_files_path']+'TrainMetadata'+'.json'
    
    validation_data_json_file_name=parameters['General']['results_files_path']+'ValidationData'+'.json'
    validation_class_json_file_name=parameters['General']['results_files_path']+'ValidationClass'+'.json'
    validation_metadata_json_file_name=parameters['General']['results_files_path']+'ValidationMetadata'+'.json'
    
    test_data_json_file_name=parameters['General']['results_files_path']+'TestData'+'.json'
    test_class_json_file_name=parameters['General']['results_files_path']+'TestClass'+'.json'
    test_metadata_json_file_name=parameters['General']['results_files_path']+'TestMetadata'+'.json'
    

    
    train_data_json = train_data.to_json(orient='records', lines=False, indent=4)
    with open(train_data_json_file_name, 'w') as file:
        file.write(train_data_json)
    generatesha256file(train_data_json_file_name,train_data_json_file_name+'.hash')
    addtocompressedfile(data_used_compressed_file_name, train_data_json_file_name, erase='Yes')
    addtocompressedfile(data_used_compressed_file_name, train_data_json_file_name+'.hash', erase='Yes')
    
    train_class_json = train_class.to_json(orient='records', lines=False, indent=4)
    with open(train_class_json_file_name, 'w') as file:
        file.write(train_class_json)
    addtocompressedfile(data_used_compressed_file_name, train_class_json_file_name, erase='Yes')
    
    train_metadata_json = train_metadata.to_json(orient='records', lines=False, indent=4)
    with open(train_metadata_json_file_name, 'w') as file:
        file.write(train_metadata_json)
    addtocompressedfile(data_used_compressed_file_name, train_metadata_json_file_name, erase='Yes')
    
    validation_data_json = validation_data.to_json(orient='records', lines=False, indent=4)
    with open(validation_data_json_file_name, 'w') as file:
        file.write(validation_data_json)
    generatesha256file(validation_data_json_file_name, validation_data_json_file_name+'.hash')
    addtocompressedfile(data_used_compressed_file_name, validation_data_json_file_name, erase='Yes')
    addtocompressedfile(data_used_compressed_file_name, validation_data_json_file_name+'.hash', erase='Yes')
    
    validation_class_json = validation_class.to_json(orient='records', lines=False, indent=4)
    with open(validation_class_json_file_name, 'w') as file:
        file.write(validation_class_json)
    addtocompressedfile(data_used_compressed_file_name, validation_class_json_file_name, erase='Yes')
    
    validation_metadata_json = validation_metadata.to_json(orient='records', lines=False, indent=4)
    with open(validation_metadata_json_file_name, 'w') as file:
        file.write(validation_metadata_json)
    addtocompressedfile(data_used_compressed_file_name, validation_metadata_json_file_name, erase='Yes')
    
    test_data_json = test_data.to_json(orient='records', lines=False, indent=4)
    with open(test_data_json_file_name, 'w') as file:
        file.write(test_data_json)
    generatesha256file(test_data_json_file_name,test_data_json_file_name+'.hash')
    addtocompressedfile(data_used_compressed_file_name, test_data_json_file_name, erase='Yes')
    addtocompressedfile(data_used_compressed_file_name, test_data_json_file_name+'.hash', erase='Yes')
    
    test_class_json = test_class.to_json(orient='records', lines=False, indent=4)
    with open(test_class_json_file_name, 'w') as file:
        file.write(test_class_json)
    addtocompressedfile(data_used_compressed_file_name, test_class_json_file_name, erase='Yes')
    
    test_metadata_json = test_metadata.to_json(orient='records', lines=False, indent=4)
    with open(test_metadata_json_file_name, 'w') as file:
        file.write(test_metadata_json)
    addtocompressedfile(data_used_compressed_file_name, test_metadata_json_file_name, erase='Yes')
    
    addtocompressedfile(compressed_file_name, data_used_compressed_file_name, erase='Yes')
    
def logmessage(message,last_time,parameters):
    timestamp_format=parameters['General']['timestamp_format']
    log_type=parameters['General']['default_log_type']
    log_file=parameters['General']['process_output_log_file']
    current_time=datetime.now()
    elapsed_time=current_time-last_time
    elapsed_time=round(elapsed_time.total_seconds(),2)
    message=current_time.strftime(timestamp_format)+': '+message+f' - Elapsed time: {elapsed_time} seconds'
    if (log_type=='screen'):
        print(message)
    elif (log_type=='file'):
        with open(log_file, 'a') as file:
            file.write(message+'\n')
    elif (log_type=='screen+file'):
        print(message)
        with open(log_file, 'a') as file:
            file.write(message+'\n')
    else:
        print('Unexpeted log type')
        exit() 
    return(current_time)

def savedatadistributionincompressedfile(metadata,output_xls_data_distribution_file_name,compressed_file_name):
    
    if 'behavior_type' in metadata.columns:

        detail_data_distribution = pd.DataFrame(metadata,columns=['behavior_type','label','behavior_group','behavior','direction','protocol','intensity','machine_platform','machine_name','file_name','statistical_behavior'])
        
        by_behavior_type_data_distribution=detail_data_distribution.groupby(['behavior_type']).size().reset_index(name='Number of Samples')
        total_samples=by_behavior_type_data_distribution['Number of Samples'].sum()
        by_behavior_type_data_distribution['%']=by_behavior_type_data_distribution['Number of Samples']/total_samples
        # by_behavior_group_data_distribution = detail_data_distribution.groupby(['behavior_group']).size().reset_index(name='Number of Samples')
        # by_behavior_group_data_distribution['%']=by_behavior_group_data_distribution['Number of Samples']/total_samples
        by_behavior_data_distribution = detail_data_distribution.groupby(['statistical_behavior']).size().reset_index(name='Number of Samples')
        by_behavior_data_distribution['%']=by_behavior_data_distribution['Number of Samples']/total_samples
        by_platform_data_distribution = detail_data_distribution.groupby(['machine_platform']).size().reset_index(name='Number of Samples')
        by_platform_data_distribution['%']=by_platform_data_distribution['Number of Samples']/total_samples
        by_machine_data_distribution = detail_data_distribution.groupby(['machine_name']).size().reset_index(name='Number of Samples')
        by_machine_data_distribution['%']=by_machine_data_distribution['Number of Samples']/total_samples

    else:
        
        detail_data_distribution = pd.DataFrame(metadata,columns=['behavior_type_0','label_0','behavior_group_0','behavior_0','direction_0','protocol_0','intensity_0','machine_platform_0','machine_name_0','file_name_0','statistical_behavior_0'])
        
        by_behavior_type_data_distribution=detail_data_distribution.groupby(['behavior_type_0']).size().reset_index(name='Number of Samples')
        total_samples=by_behavior_type_data_distribution['Number of Samples'].sum()
        by_behavior_type_data_distribution['%']=by_behavior_type_data_distribution['Number of Samples']/total_samples
        # by_behavior_group_data_distribution = detail_data_distribution.groupby(['behavior_group_0']).size().reset_index(name='Number of Samples')
        # by_behavior_group_data_distribution['%']=by_behavior_group_data_distribution['Number of Samples']/total_samples
        by_behavior_data_distribution = detail_data_distribution.groupby(['statistical_behavior_0']).size().reset_index(name='Number of Samples')
        by_behavior_data_distribution['%']=by_behavior_data_distribution['Number of Samples']/total_samples
        by_platform_data_distribution = detail_data_distribution.groupby(['machine_platform_0']).size().reset_index(name='Number of Samples')
        by_platform_data_distribution['%']=by_platform_data_distribution['Number of Samples']/total_samples
        by_machine_data_distribution = detail_data_distribution.groupby(['machine_name_0']).size().reset_index(name='Number of Samples')
        by_machine_data_distribution['%']=by_machine_data_distribution['Number of Samples']/total_samples
  
    with pd.ExcelWriter(output_xls_data_distribution_file_name) as writer:  

        by_behavior_type_data_distribution.to_excel(writer, sheet_name='By Behavior Type')
        # by_behavior_group_data_distribution.to_excel(writer, sheet_name='ByBehaviorGroup')
        by_behavior_data_distribution.to_excel(writer, sheet_name='By Behavior')
        by_platform_data_distribution.to_excel(writer, sheet_name='By Platform')
        by_machine_data_distribution.to_excel(writer, sheet_name='By Machine')

    addtocompressedfile(compressed_file_name, output_xls_data_distribution_file_name)
    
def saveimportancesincompressedfile(algorithm, features, execution, matrix_title,percentage=95):
    model = algorithm.__class__.__name__
    try:
        if model in ['RandomForestClassifier', 'DecisionTreeClassifier','GradientBoostingClassifier','AdaBoostClassifier','XGBClassifier']:
            feature_importances = pd.DataFrame({'Feature': features, 'Importance': algorithm.feature_importances_})
        elif model == 'LogisticRegression':
            feature_importances = pd.DataFrame({'Feature': features, 'Importance': np.abs(algorithm.coef_[0])})
        elif model == 'SVC':
            feature_importances = pd.DataFrame({'Feature': features, 'Importance': np.abs(algorithm.coef_.ravel())})
        elif model in ['KNeighborsClassifier', 'GaussianNB']:
            print(f'No feature importances available for {model}')
            return
        else:
            print(f'Unexpected model {model}')
            return
    except AttributeError as e:
        print(f'Error accessing model attributes: {e}')
        return

    if len(features) != len(feature_importances):
        print("Mismatch in number of features and importances")
        return
    
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    total_importance = feature_importances['Importance'].sum()
    feature_importances['Importance'] = 100 * feature_importances['Importance'] / total_importance
    feature_importances.to_excel(execution['output_xlsx_feature_importances'], index=False)
    addtocompressedfile(execution['compressed_file_name'], execution['output_xlsx_feature_importances'])
    cumulative_percentage = 0
    selected_features = []
    for _, row in feature_importances.iterrows():
        cumulative_percentage += row['Importance']
        selected_features.append(row['Feature'])
        if cumulative_percentage >= percentage:
            break

    feature_importances = feature_importances[feature_importances['Feature'].isin(selected_features)]
    total_percentage=round(feature_importances['Importance'].sum(),2)
    number_of_features=feature_importances.shape[0]
    strings_list = matrix_title.split('\n', 3)
    title=strings_list[0]+'\n'+f'Featrures={number_of_features}|Percentage={total_percentage}'+'\n'+strings_list[2]
 
    if feature_importances.empty:
        print("No important features found")
        return

    plt.figure(figsize=(10, 6))
    plt.bar(feature_importances['Feature'], feature_importances['Importance'])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(execution['output_feature_importances_image_file_name'], dpi=execution['images_resolution'])
    plt.close()

    addtocompressedfile(execution['compressed_file_name'], execution['output_feature_importances_image_file_name'])

def savecasestobetestedincompressedfile(cases_to_be_tested,parameters):
    cases_to_be_tested_json = json.dumps(cases_to_be_tested, indent=4) 
    with open(parameters['General']['analisys_definition'], "a") as file:
        file.write(cases_to_be_tested_json)
        
    addtocompressedfile(parameters['General']['compressed_file_name'], parameters['General']['analisys_definition'])
    
    
def getaccuracyindexgrouped(y_actual,y_predicted,metadata,column_name,group_name_to_show,text_to_concatenate):
    y_actual_to_process=y_actual.copy()
    y_actual_to_process.columns = ['actual_class']
    
    y_predicted_to_process=y_predicted.copy()
    y_predicted_to_process.columns = ['predicted_class']
    
    metadata_to_process=metadata.copy()
    metadata_to_process[group_name_to_show] = metadata_to_process.get(column_name, metadata_to_process.get(column_name+'_0'))
    metadata_to_process=metadata_to_process[[group_name_to_show]]

    detailed_accuracy_data = pd.concat([y_actual_to_process, y_predicted_to_process, metadata_to_process], axis=1)
    
    detailed_accuracy_data['Classification'] = vectorizedconfussionmatrixclassification(detailed_accuracy_data['actual_class'], detailed_accuracy_data['predicted_class'])
    
    grouped_accuracy_data = detailed_accuracy_data.groupby([group_name_to_show, 'Classification']).size().unstack(fill_value=0)
    if 'TP' not in grouped_accuracy_data.columns:
        grouped_accuracy_data['TP'] = 0
    if 'TN' not in grouped_accuracy_data.columns:
        grouped_accuracy_data['TN'] = 0
    if 'FP' not in grouped_accuracy_data.columns:
        grouped_accuracy_data['FP'] = 0
    if 'FN' not in grouped_accuracy_data.columns:
        grouped_accuracy_data['FN'] = 0
    
    grouped_accuracy_data['Total '+text_to_concatenate+' Samples']=grouped_accuracy_data['TP']+grouped_accuracy_data['TN']+grouped_accuracy_data['FP']+grouped_accuracy_data['FN']
    grouped_accuracy_data['Correct '+text_to_concatenate+' Samples']=grouped_accuracy_data['TP']+grouped_accuracy_data['TN']
    grouped_accuracy_data['Incorrect '+text_to_concatenate+' Samples']=grouped_accuracy_data['FP']+grouped_accuracy_data['FN']
    grouped_accuracy_data['% Correct '+text_to_concatenate+' Samples'] = (grouped_accuracy_data['Correct '+text_to_concatenate+' Samples'] / grouped_accuracy_data['Total '+text_to_concatenate+' Samples']).replace([np.inf, -np.inf], 0).fillna(0).round(4)
    grouped_accuracy_data['% Incorrect '+text_to_concatenate+' Samples'] = (grouped_accuracy_data['Incorrect '+text_to_concatenate+' Samples'] / grouped_accuracy_data['Total '+text_to_concatenate+' Samples']).replace([np.inf, -np.inf], 0).fillna(0).round(4)
    grouped_accuracy_data = grouped_accuracy_data.rename(columns={'TP': 'TP '+text_to_concatenate+' Samples', 'TN': 'TN '+text_to_concatenate+' Samples','FP': 'FP '+text_to_concatenate+' Samples','FN': 'FN '+text_to_concatenate+' Samples'}).reset_index()
    
    grouped_accuracy_data = grouped_accuracy_data[[group_name_to_show, 'Total '+text_to_concatenate+' Samples', 'Correct '+text_to_concatenate+' Samples', '% Correct '+text_to_concatenate+' Samples','Incorrect '+text_to_concatenate+' Samples', '% Incorrect '+text_to_concatenate+' Samples','TP '+text_to_concatenate+' Samples', 'TN '+text_to_concatenate+' Samples','FP '+text_to_concatenate+' Samples','FN '+text_to_concatenate+' Samples']]
    
    return(grouped_accuracy_data)

def gettrainingdatagrouped(metadata,column_name,group_name_to_show,text_to_concatenate):
    metadata_to_process=metadata.copy()
    metadata_to_process[group_name_to_show] = metadata_to_process.get(column_name, metadata_to_process.get(column_name+'_0'))
    metadata_to_process=metadata_to_process[[group_name_to_show]]
    grouped_metadata=metadata_to_process.groupby(group_name_to_show).size().reset_index(name=text_to_concatenate+' Samples')
    total_train_samples = grouped_metadata[text_to_concatenate+' Samples'].sum()
    grouped_metadata['% '+text_to_concatenate+' Samples'] = np.where(total_train_samples == 0, 0,(grouped_metadata[text_to_concatenate+' Samples'] / total_train_samples).round(4))
    grouped_metadata = grouped_metadata[[group_name_to_show, text_to_concatenate+' Samples','% '+text_to_concatenate+' Samples']]
    
    return(grouped_metadata)
    
def saveaccuracyindexincompressedfile(y_train,y_train_pred,meta_train,y_validation,y_validation_pred,meta_validation,y_test,y_test_pred,meta_test,execution):
    output_xlsx_accuracy_index_analysis=execution['output_xlsx_accuracy_index_analysis_file_name']
    compressed_file_name=execution['compressed_file_name']

    # By Behavior Type
    train_data_by_behavior_type=gettrainingdatagrouped(meta_train,'behavior_type','Behavior Type','Train')
    validation_accuracy_by_behavior_type=getaccuracyindexgrouped(y_validation,y_validation_pred,meta_validation,'behavior_type','Behavior Type','Validation')
    test_accuracy_by_behavior_type=getaccuracyindexgrouped(y_test,y_test_pred,meta_test,'behavior_type','Behavior Type','Test')
    accuracy_by_behavior_type = pd.merge(train_data_by_behavior_type, validation_accuracy_by_behavior_type, on='Behavior Type', how='outer').fillna(0)
    accuracy_by_behavior_type = pd.merge(accuracy_by_behavior_type, test_accuracy_by_behavior_type, on='Behavior Type', how='outer').fillna(0)
    accuracy_by_behavior_type.insert(0, 'test_identifier', execution['output_file_name_format'])
    accuracy_by_behavior_type.insert(1, 'TID', execution['TID'])
    accuracy_by_behavior_type.insert(2, 'analyzed_model', execution['analyzed_model'])
    del train_data_by_behavior_type, validation_accuracy_by_behavior_type, test_accuracy_by_behavior_type
    with pd.ExcelWriter(output_xlsx_accuracy_index_analysis, engine='openpyxl', mode='w') as writer:  
        accuracy_by_behavior_type.to_excel(writer, sheet_name='By Behavior Type')
    del accuracy_by_behavior_type
    
    # By Behavior
    train_data_by_behavior=gettrainingdatagrouped(meta_train,'statistical_behavior','Statistical Behavior','Train')
    validation_accuracy_by_behavior=getaccuracyindexgrouped(y_validation,y_validation_pred,meta_validation,'statistical_behavior','Statistical Behavior','Validation')
    test_accuracy_by_behavior=getaccuracyindexgrouped(y_test,y_test_pred,meta_test,'statistical_behavior','Statistical Behavior','Test')
    accuracy_by_behavior = pd.merge(train_data_by_behavior, validation_accuracy_by_behavior, on='Statistical Behavior', how='outer').fillna(0)
    accuracy_by_behavior = pd.merge(accuracy_by_behavior, test_accuracy_by_behavior, on='Statistical Behavior', how='outer').fillna(0)
    accuracy_by_behavior.insert(0, 'test_identifier', execution['output_file_name_format'])
    accuracy_by_behavior.insert(1, 'TID', execution['TID'])
    accuracy_by_behavior.insert(2, 'analyzed_model', execution['analyzed_model'])
    detailed_accuracy_by_BITD=accuracy_by_behavior.copy()
    del train_data_by_behavior, validation_accuracy_by_behavior, test_accuracy_by_behavior
    with pd.ExcelWriter(output_xlsx_accuracy_index_analysis, engine='openpyxl', mode='a') as writer:  
        accuracy_by_behavior.to_excel(writer, sheet_name='By Behavior')
    del accuracy_by_behavior

    # By Platform
    train_data_by_platform=gettrainingdatagrouped(meta_train,'machine_platform','Platform','Train')
    validation_accuracy_by_platform=getaccuracyindexgrouped(y_validation,y_validation_pred,meta_validation,'machine_platform','Platform','Validation')
    test_accuracy_by_platform=getaccuracyindexgrouped(y_test,y_test_pred,meta_test,'machine_platform','Platform','Test')
    accuracy_by_platform = pd.merge(train_data_by_platform, validation_accuracy_by_platform, on='Platform', how='outer').fillna(0)
    accuracy_by_platform = pd.merge(accuracy_by_platform, test_accuracy_by_platform, on='Platform', how='outer').fillna(0)
    accuracy_by_platform.insert(0, 'test_identifier', execution['output_file_name_format'])
    accuracy_by_platform.insert(1, 'TID', execution['TID'])
    accuracy_by_platform.insert(2, 'analyzed_model', execution['analyzed_model'])
    del train_data_by_platform, validation_accuracy_by_platform, test_accuracy_by_platform
    with pd.ExcelWriter(output_xlsx_accuracy_index_analysis, engine='openpyxl', mode='a') as writer:  
        accuracy_by_platform.to_excel(writer, sheet_name='By Platform')
    del accuracy_by_platform
    
    
    
    # By Machine
    train_data_by_machine=gettrainingdatagrouped(meta_train,'machine_name','Machine','Train')
    validation_accuracy_by_machine=getaccuracyindexgrouped(y_validation,y_validation_pred,meta_validation,'machine_name','Machine','Validation')
    test_accuracy_by_machine=getaccuracyindexgrouped(y_test,y_test_pred,meta_test,'machine_name','Machine','Test')
    accuracy_by_machine = pd.merge(train_data_by_machine, validation_accuracy_by_machine, on='Machine', how='outer').fillna(0)
    accuracy_by_machine = pd.merge(accuracy_by_machine, test_accuracy_by_machine, on='Machine', how='outer').fillna(0)
    accuracy_by_machine.insert(0, 'test_identifier', execution['output_file_name_format'])
    accuracy_by_machine.insert(1, 'TID', execution['TID'])
    accuracy_by_machine.insert(2, 'analyzed_model', execution['analyzed_model'])
    del train_data_by_machine, validation_accuracy_by_machine, test_accuracy_by_machine
    with pd.ExcelWriter(output_xlsx_accuracy_index_analysis, engine='openpyxl', mode='a') as writer:  
        accuracy_by_machine.to_excel(writer, sheet_name='By Machine')
    del accuracy_by_machine
    
    
    # By File Name
    train_data_by_file_name=gettrainingdatagrouped(meta_train,'file_name','File Name','Train')
    validation_accuracy_by_file_name=getaccuracyindexgrouped(y_validation,y_validation_pred,meta_validation,'file_name','File Name','Validation')
    test_accuracy_by_file_name=getaccuracyindexgrouped(y_test,y_test_pred,meta_test,'file_name','File Name','Test')
    accuracy_by_file_name = pd.merge(train_data_by_file_name, validation_accuracy_by_file_name, on='File Name', how='outer').fillna(0)
    accuracy_by_file_name = pd.merge(accuracy_by_file_name, test_accuracy_by_file_name, on='File Name', how='outer').fillna(0)
    accuracy_by_file_name.insert(0, 'test_identifier', execution['output_file_name_format'])
    accuracy_by_file_name.insert(1, 'TID', execution['TID'])
    accuracy_by_file_name.insert(2, 'analyzed_model', execution['analyzed_model'])
    del train_data_by_file_name, validation_accuracy_by_file_name, test_accuracy_by_file_name
    with pd.ExcelWriter(output_xlsx_accuracy_index_analysis, engine='openpyxl', mode='a') as writer:  
        accuracy_by_file_name.to_excel(writer, sheet_name='By File Name')
    del accuracy_by_file_name
    
   
    # By Behavior in Training Dataset
    
    detailed_accuracy_by_BITD['Behavior in Training Dataset'] = detailed_accuracy_by_BITD['Train Samples'].apply(lambda x: 'Not Present' if x == 0 else 'Present')
    accuracy_by_BITD = (
    detailed_accuracy_by_BITD
    .groupby('Behavior in Training Dataset')[[
        'Train Samples', 
        '% Train Samples',
        'Total Validation Samples', 
        'Correct Validation Samples', 
        'Incorrect Validation Samples', 
        'TP Validation Samples', 
        'TN Validation Samples', 
        'FP Validation Samples', 
        'FN Validation Samples', 
        'Total Test Samples', 
        'Correct Test Samples', 
        'Incorrect Test Samples', 
        'TP Test Samples', 
        'TN Test Samples', 
        'FP Test Samples', 
        'FN Test Samples'
    ]]
    .sum()
    .reset_index()
    )
    accuracy_by_BITD['% Correct Validation Samples']=accuracy_by_BITD['Correct Validation Samples']/accuracy_by_BITD['Total Validation Samples']
    accuracy_by_BITD['% Incorrect Validation Samples']=accuracy_by_BITD['Incorrect Validation Samples']/accuracy_by_BITD['Total Validation Samples']
    accuracy_by_BITD['% Correct Test Samples']=accuracy_by_BITD['Correct Test Samples']/accuracy_by_BITD['Total Test Samples']
    accuracy_by_BITD['% Incorrect Test Samples']=accuracy_by_BITD['Incorrect Test Samples']/accuracy_by_BITD['Total Test Samples']
    accuracy_by_BITD = accuracy_by_BITD[['Behavior in Training Dataset','Train Samples','% Train Samples','Total Validation Samples', 'Correct Validation Samples', '% Correct Validation Samples','Incorrect Validation Samples', '% Incorrect Validation Samples','TP Validation Samples','TN Validation Samples','FP Validation Samples','FN Validation Samples','Total Test Samples','Correct Test Samples','% Correct Test Samples', 'Incorrect Test Samples', '% Incorrect Test Samples','TP Test Samples','TN Test Samples','FP Test Samples','FN Test Samples']]
    accuracy_by_BITD.insert(0, 'test_identifier', execution['output_file_name_format'])
    accuracy_by_BITD.insert(1, 'TID', execution['TID'])
    accuracy_by_BITD.insert(2, 'analyzed_model', execution['analyzed_model'])
    del detailed_accuracy_by_BITD
    with pd.ExcelWriter(output_xlsx_accuracy_index_analysis, engine='openpyxl', mode='a') as writer:  
        accuracy_by_BITD.to_excel(writer, sheet_name='By BITD')
    del accuracy_by_BITD

    # with pd.ExcelWriter(output_xlsx_accuracy_index_analysis) as writer:  
    #     accuracy_by_behavior_type.to_excel(writer, sheet_name='By Behavior Type')
    #     accuracy_by_behavior.to_excel(writer, sheet_name='By Behavior')
    #     accuracy_by_platform.to_excel(writer, sheet_name='By Platform')
    #     accuracy_by_machine.to_excel(writer, sheet_name='By Machine')
    #     accuracy_by_BITD.to_excel(writer, sheet_name='By BITD')
    
    
    addtocompressedfile(compressed_file_name, output_xlsx_accuracy_index_analysis)
 