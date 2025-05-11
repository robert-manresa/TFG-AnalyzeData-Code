#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import time
from datetime import datetime
import shutil
import os
import sys
import warnings
import argparse
from sklearn.exceptions import ConvergenceWarning


from definefunctions import parametersdefinition,casestobetesteddefinition,filenamesdefinition,tidvalidation
from inputfunctions import getfilesinfo,getalldata,getdataset
from processfunctions import analyzedata,adddatatoexecution
from processfunctions import getcasestoexecute,lowvariancecolumns,removecolumns
from outputfunctions import addtocompressedfile,savecodeincompressedfile,savedataincompressedfile,logmessage,savedatadistributionincompressedfile,savecasestobetestedincompressedfile


def main():

    parser = argparse.ArgumentParser(description="Script Parameters")

    parser.add_argument("--cases_name", type=str, default="TEST", help="Test identifier")
    parser.add_argument("--temporality_index", type=int, default=1, help="Temporality identifier")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    # Parse arguments
    args = parser.parse_args()
    
    cases_name=args.cases_name
    temporality_index=args.temporality_index
    batch_size=args.batch_size

    # Activate Console Log File
    console_log_file_name = '/home/grau/TFG/Resultats/ConsoleLogFile.log'
    console_error_file_name='/home/grau/TFG/Resultats/ConsoleErrorFile.log'
    console_log_file = open(console_log_file_name, "w")
    console_error_file = open(console_error_file_name, "w")
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        
        sys.stdout = console_log_file
        sys.stderr = console_error_file
        warnings.simplefilter("default")
        warnings.filterwarnings("default", category=ConvergenceWarning)
        warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: print(f"{category.__name__} in {filename}, line {lineno}: {message}", file=sys.stderr)
        



        # Define cases to be tested - For manual execution
        start_batch_time=datetime.now()
        last_time=datetime.now()
          
        # cases_name='ALLFASTMODELS-TRAINSET_A_50K-DATABASE_B'
        # cases_name='ALLXNNMODELS-TRAINSET_A_50K-DATABASE_B'
        # cases_name='ALLFASTMODELS-TRAINSET_B_50K-DATABASE_A'
        # cases_name='EXAMPLE-A-TO-B'
        # cases_name='EXAMPLE-B-TO-A'
        # cases_name='TEST'
        # temporality_index=2
        # batch_size=5
        
        batch_identifier=cases_name+'-'+f'TID{temporality_index}'+'-'+f'BATCH{batch_size}'+'-'+datetime.now().strftime('%Y%m%dT%H%M%S')
        
        
        if not tidvalidation(cases_name):
            cases_name=cases_name+f'-TID{temporality_index}'
        
        parameters=parametersdefinition(cases_name,temporality_index)
        
        last_time=logmessage('Start process',last_time,parameters)
        
        parameters['General']['savedatadistribution']='Yes'
        parameters['General']['savevaluationresults']='Yes'
        parameters['General']['savedataused']='No'
        parameters['General']['savecodeused']='Yes'
        parameters['General']['saveanalysisdefinition']='Yes'
        
        cases_to_be_tested,meta_models=casestobetesteddefinition(cases_name,temporality_index)
        
        # Get data definition and files to be loaded for known and unknown data
        
        last_time=logmessage('Get data definition - Start',last_time,parameters)   
        
        known_data_files_info,unknown_data_files_info,validation_data_files_info,known_data_definition,unknown_data_definition,validation_data_definition=getfilesinfo(cases_to_be_tested)
        
        last_time=logmessage('Get data definition - End',last_time,parameters)    
        
        # Get all known data 
    
        last_time=logmessage('Get known data - Start',last_time,parameters)   
        
        all_known_data=getalldata(known_data_files_info,known_data_definition,cases_to_be_tested['data_platform_to_be_used'],parameters)
    
        last_time=logmessage('Get known data - End',last_time,parameters)
        
        # Get all validation data 
    
        last_time=logmessage('Get validation data - Start',last_time,parameters)   
        
        all_validation_data=getalldata(validation_data_files_info,validation_data_definition,cases_to_be_tested['data_platform_to_be_used'],parameters)
    
        last_time=logmessage('Get validation data - End',last_time,parameters)   
        
        # Get all unknown data 
        
        last_time=logmessage('Get unknown data - Start',last_time,parameters)
        
        all_unknown_data=getalldata(unknown_data_files_info,unknown_data_definition,cases_to_be_tested['data_platform_to_be_used'],parameters)
        
        last_time=logmessage('Get unknown data - End',last_time,parameters)
        
        original_load_data_output_log_file=parameters['General']['process_output_log_file']
        
        # Remove Columns
        
        # low_variance_columns_known=lowvariancecolumns(all_known_data, threshold=0.000001)
        # low_variance_columns_unknown=lowvariancecolumns(all_unknown_data, threshold=0.000001)
        # columns_to_remove=['files_read', 'files_write', 'files_created', 'files_deleted', 'command_min', 'gpu_proc', 'gpu_mem']
        # all_known_data=removecolumns(all_known_data, columns_to_remove)   
        # all_unknown_data=removecolumns(all_unknown_data, columns_to_remove)   
        # parameters['General']['data_columns_to_process']=all_known_data.columns
        
        start_iteration_time=datetime.now()
        load_time=start_iteration_time-start_batch_time
        
        for n in range (1,batch_size+1):
            
            # Get iteration parameters
            
    
            parameters=filenamesdefinition(cases_name,parameters,temporality_index)
            load_data_output_log_file=parameters['General']['load_data_output_log_file']
            process_output_log_file=parameters['General']['process_output_log_file']
            
            
            shutil.copy(original_load_data_output_log_file, load_data_output_log_file)
            
            
            last_time=logmessage(f'Start batch iteration {n}',last_time,parameters)
            
            # Save code and cases to be tested
    
            if parameters['General']['savecodeused']=='Yes': 
                last_time=logmessage('Save code used - Start',last_time,parameters)
                savecodeincompressedfile(parameters)
                last_time=logmessage('Save code used - End',last_time,parameters)
            else: 
                last_time=logmessage('Save code used - The saving of the code used was skipped due to the configuration',last_time,parameters)
    
            if parameters['General']['saveanalysisdefinition']=='Yes': 
                last_time=logmessage('Save analysis definition - Start',last_time,parameters)
                information = cases_to_be_tested.copy()
                information.update(meta_models)
                information['temporality_index']=temporality_index
                information['batch_identifier']=batch_identifier
                information['batch_size']=batch_size
                information['batch_iteration']=n
                savecasestobetestedincompressedfile(information,parameters)
                last_time=logmessage('Save analysis definition - End',last_time,parameters)
            else: 
                last_time=logmessage('Save analysis definition - The saving of the analysis definition was skipped due to the configuration',last_time,parameters)
                
            # Get all executions defined in cases_to_be_tested
    
            last_time=logmessage('Get all cases - Start',last_time,parameters)
            executions=getcasestoexecute(cases_to_be_tested)
            last_time=logmessage('Get all cases - End',last_time,parameters)
            
            
            compressed_file_name=parameters['General']['compressed_file_name']
    
            output_xls_results_file_name_train=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-Results-A-Train.xlsx'
            output_xls_results_file_name_validation=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-Results-B-Validation.xlsx'
            output_xls_results_file_name_test=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-Results-C-Test.xlsx'
            
            output_xls_known_data_distribution_file_name=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-DataDistribution-A-Known.xlsx'
            output_xls_unknown_data_distribution_file_name=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-DataDistribution-B-Unknown.xlsx'
            output_xls_train_data_distribution_file_name=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-DataDistribution-C-Train.xlsx'
            output_xls_validation_data_distribution_file_name=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-DataDistribution-D-Validation.xlsx'
            output_xls_test_data_distribution_file_name=parameters['General']['results_files_path']+parameters['General']['output_file_name_format']+'-DataDistribution-E-Test.xlsx'
            
            
            # Get train dataset
            
            
            last_time=logmessage('Get train dataset - Start',last_time,parameters)
            train_data,train_class,train_metadata=getdataset(all_known_data,known_data_definition,parameters,data_type='Train')
            sys.stdout.flush()
            last_time=logmessage('Get train dataset - End',last_time,parameters)
            
            # Get validation dataset
            
            last_time=logmessage('Get validation dataset - Start',last_time,parameters)
            validation_data,validation_class,validation_metadata=getdataset(all_validation_data,validation_data_definition,parameters,data_type='Validation')
            sys.stdout.flush()
            last_time=logmessage('Get validation dataset - End',last_time,parameters)
            
            # Get test dataset
            
            last_time=logmessage('Get test dataset - Start',last_time,parameters)
            test_data,test_class,test_metadata=getdataset(all_unknown_data,unknown_data_definition,parameters,data_type='Test')
            sys.stdout.flush()
            last_time=logmessage('Get test dataset - End',last_time,parameters)
            
            # Save known, unknown, train, validation and test data distribution
                    
            
            if parameters['General']['savedatadistribution']=='Yes': 
                
                # Commented to save execution time
                
                # last_time=logmessage('Save known data distribution - Start',last_time,parameters)
                # savedatadistributionincompressedfile(all_known_data,output_xls_known_data_distribution_file_name,compressed_file_name)
                # last_time=logmessage('Save known data distribution - End',last_time,parameters)
                
                # last_time=logmessage('Save unknown data distribution - Start',last_time,parameters)
                # savedatadistributionincompressedfile(all_unknown_data,output_xls_unknown_data_distribution_file_name,compressed_file_name)
                # last_time=logmessage('Save unknown data distribution - End',last_time,parameters)
                
                last_time=logmessage('Save train data distribution - Start',last_time,parameters)
                savedatadistributionincompressedfile(train_metadata,output_xls_train_data_distribution_file_name,compressed_file_name)
                last_time=logmessage('Save train data distribution - End',last_time,parameters)
                
                last_time=logmessage('Save validation data distribution - Start',last_time,parameters)
                savedatadistributionincompressedfile(validation_metadata,output_xls_validation_data_distribution_file_name,compressed_file_name)
                last_time=logmessage('Save validation data distribution - End',last_time,parameters)
                
                last_time=logmessage('Save test data distribution - Start',last_time,parameters)
                savedatadistributionincompressedfile(test_metadata,output_xls_test_data_distribution_file_name,compressed_file_name)
                last_time=logmessage('Save test data distribution - End',last_time,parameters)
                
            else:
                last_time=logmessage('Save known data distribution- The saving of the known data distribution was skipped due to the configuration',last_time,parameters)
                last_time=logmessage('Save unknown data distribution- The saving of the unknown data distribution was skipped due to the configuration',last_time,parameters)
                last_time=logmessage('Save train data distribution- The saving of the train data distribution was skipped due to the configuration',last_time,parameters)
                last_time=logmessage('Save test validation distribution- The saving of the validation data distribution was skipped due to the configuration',last_time,parameters)
                last_time=logmessage('Save test data distribution- The saving of the test data distribution was skipped due to the configuration',last_time,parameters)
    
            # Save data in compressed file
            
            if parameters['General']['savedataused']=='Yes': 
                last_time=logmessage('Save data used - Start',last_time,parameters)
                savedataincompressedfile(train_data,train_class,train_metadata,validation_data,validation_class,validation_metadata,test_data,test_class,test_metadata,parameters)
                last_time=logmessage('Save data used - End',last_time,parameters)
            else: 
                last_time=logmessage('Save data used - The saving of the data used was skipped due to the configuration',last_time,parameters)
            
            # Analysis loop
            
            results_train = pd.DataFrame(columns=['test_identifier','TID','model_hierarchy','score', 'precision', 'recall', 'accuracy', 'f1score','roc_auc', 'tp', 'tn', 'fp', 'fn', 'model', 'dimensionality_alter', 'preprocessing', 'balancing', 'optimization', 'behaviors_to_use', 'string1', 'string2', 'string3', 'string4', 'elapsed_time','precision_diff','recall_diff','accuracy_diff','f1score_diff','roc_auc_diff'])
            results_validation = pd.DataFrame(columns=['test_identifier','TID','model_hierarchy','score', 'precision', 'recall', 'accuracy', 'f1score','roc_auc', 'tp', 'tn', 'fp', 'fn', 'model', 'dimensionality_alter', 'preprocessing', 'balancing', 'optimization', 'behaviors_to_use', 'string1', 'string2', 'string3', 'string4', 'elapsed_time','precision_diff','recall_diff','accuracy_diff','f1score_diff','roc_auc_diff'])
            results_test = pd.DataFrame(columns=['test_identifier','TID','model_hierarchy','score', 'precision', 'recall', 'accuracy', 'f1score','roc_auc', 'tp', 'tn', 'fp', 'fn', 'model', 'dimensionality_alter', 'preprocessing', 'balancing', 'optimization', 'behaviors_to_use', 'string1', 'string2', 'string3', 'string4', 'elapsed_time','precision_diff','recall_diff','accuracy_diff','f1score_diff','roc_auc_diff'])
            
            X_metamodel_train=pd.DataFrame()
            X_metamodel_validation=pd.DataFrame()
            X_metamodel_test=pd.DataFrame()
            
            for execution in executions:
                
                execution['complete_model'],execution['preprocessing'],execution['dimensionality_alter'],execution['balancing'],execution['optimization'],execution['behaviors_to_use']=execution['base_model'].split('-')
                execution['TID'] = 'TID' + str(temporality_index)
                execution['model_hierarchy']='Base Model'
                execution['output_file_name_format']=parameters['General']['output_file_name_format']
                
                start_time = time.time()
                execution['num_features']=train_data.shape[1]
    
                execution=adddatatoexecution(execution,parameters)
                information_to_print=execution['analyzed_model']
                last_time=logmessage(f'{information_to_print} - Start',last_time,parameters)
              
               
                score_train,precision_train,recall_train,accuracy_train,f1score_train,roc_auc_train,tp_train,tn_train,fp_train,fn_train,strings_train,score_validation,precision_validation,recall_validation,accuracy_validation,f1score_validation,roc_auc_validation,tp_validation,tn_validation,fp_validation,fn_validation,strings_validation,score_test,precision_test,recall_test,accuracy_test,f1score_test,roc_auc_test,tp_test,tn_test,fp_test,fn_test,strings_test,y_train_pred,y_validation_pred,y_test_pred=analyzedata(train_data,validation_data,test_data,train_class,validation_class,test_class,train_metadata,validation_metadata,test_metadata,execution,parameters)
                X_metamodel_train[execution['complete_model']+'-'+execution['preprocessing']]=y_train_pred
                X_metamodel_validation[execution['complete_model']+'-'+execution['preprocessing']]=y_validation_pred
                X_metamodel_test[execution['complete_model']+'-'+execution['preprocessing']]=y_test_pred
                end_time=time.time()
                elapsed_time=end_time-start_time
                
                
                results_to_add_train = {'test_identifier':execution['output_file_name_format'],'TID':execution['TID'],'model_hierarchy':execution['model_hierarchy'],'score': score_train, 'precision': precision_train, 'recall': recall_train, 'accuracy': accuracy_train, 'f1score': f1score_train,'roc_auc':roc_auc_train, 'tp': tp_train, 'tn': tn_train, 'fp': fp_train, 'fn': fn_train, 'model': execution['complete_model'], 'dimensionality_alter': execution['dimensionality_alter'], 'preprocessing': execution['preprocessing'], 'balancing': execution['balancing'], 'optimization': execution['optimization'], 'behaviors_to_use': execution['behaviors_to_use'], 'string1': strings_train[0], 'string2': strings_train[1], 'string3': strings_train[2], 'string4': strings_train[3], 'elapsed_time': round(elapsed_time,2),'precision_diff': round(precision_train-precision_train,3),'recall_diff': round(recall_train-recall_train,3),'accuracy_diff': round(accuracy_train-accuracy_train,3), 'f1score_diff': round(f1score_train-f1score_train,3),'roc_auc_diff': round(roc_auc_train-roc_auc_train,3)}
                results_train = pd.concat([results_train, pd.DataFrame(results_to_add_train, index=[0])], ignore_index=True)
                
                results_to_add_validation = {'test_identifier':execution['output_file_name_format'],'TID':execution['TID'],'model_hierarchy':execution['model_hierarchy'],'score': score_validation, 'precision': precision_validation, 'recall': recall_validation, 'accuracy': accuracy_validation, 'f1score': f1score_validation,'roc_auc':roc_auc_validation, 'tp': tp_validation, 'tn': tn_validation, 'fp': fp_validation, 'fn': fn_validation, 'model': execution['complete_model'], 'dimensionality_alter': execution['dimensionality_alter'], 'preprocessing': execution['preprocessing'], 'balancing': execution['balancing'], 'optimization': execution['optimization'], 'behaviors_to_use': execution['behaviors_to_use'], 'string1': strings_validation[0], 'string2': strings_validation[1], 'string3': strings_validation[2], 'string4': strings_validation[3], 'elapsed_time': round(elapsed_time,2),'precision_diff': round(precision_validation-precision_train,3),'recall_diff': round(recall_validation-recall_train,3),'accuracy_diff': round(accuracy_validation-accuracy_train,3), 'f1score_diff': round(f1score_validation-f1score_train,3),'roc_auc_diff': round(roc_auc_validation-roc_auc_train,3)}
                results_validation = pd.concat([results_validation, pd.DataFrame(results_to_add_validation, index=[0])], ignore_index=True)
                
                results_to_add_test = {'test_identifier':execution['output_file_name_format'],'TID':execution['TID'],'model_hierarchy':execution['model_hierarchy'],'score': score_test, 'precision': precision_test, 'recall': recall_test, 'accuracy': accuracy_test, 'f1score': f1score_test,'roc_auc': roc_auc_test, 'tp': tp_test, 'tn': tn_test, 'fp': fp_test, 'fn': fn_test, 'model': execution['complete_model'],'dimensionality_alter': execution['dimensionality_alter'], 'preprocessing': execution['preprocessing'], 'balancing': execution['balancing'], 'optimization': execution['optimization'], 'behaviors_to_use': execution['behaviors_to_use'], 'string1': strings_test[0], 'string2': strings_test[1], 'string3': strings_test[2], 'string4': strings_test[3], 'elapsed_time': round(elapsed_time,2),'precision_diff': round(precision_test-precision_train,3),'recall_diff': round(recall_test-recall_train,3),'accuracy_diff': round(accuracy_test-accuracy_train,3), 'f1score_diff': round(f1score_test-f1score_train,3),'roc_auc_diff': round(roc_auc_test-roc_auc_train,3) }
                results_test = pd.concat([results_test, pd.DataFrame(results_to_add_test, index=[0])], ignore_index=True)
                
                last_time=logmessage(f'{information_to_print} - End',last_time,parameters)
              
                
                
                results_train = pd.concat([results_train, pd.DataFrame(results_to_add_train, index=[0])], ignore_index=True)
                results_validation = pd.concat([results_validation, pd.DataFrame(results_to_add_validation, index=[0])], ignore_index=True)
                results_test = pd.concat([results_test, pd.DataFrame(results_to_add_test, index=[0])], ignore_index=True)
                
                
                results_train = results_train.drop_duplicates()
                results_validation = results_validation.drop_duplicates()
                results_test = results_test.drop_duplicates()
                
                results_train = results_train.sort_values(by='score')
                results_validation = results_validation.sort_values(by='score')
                results_test = results_test.sort_values(by='score')
                
                results_train.to_excel(output_xls_results_file_name_train, index=False)
                results_validation.to_excel(output_xls_results_file_name_validation, index=False)
                results_test.to_excel(output_xls_results_file_name_test, index=False)
                
                
            for meta_model in meta_models['meta_model']:
         
                execution['num_features']=X_metamodel_train.shape[1]
                execution['complete_model'], execution['preprocessing'], execution['dimensionality_alter'], execution['balancing'], execution['optimization'], execution['behaviors_to_use'] = meta_model.split('-')
                execution['TID'] = 'TID' + str(temporality_index)
                execution['model_hierarchy']='Meta Model'
                execution['output_file_name_format']=parameters['General']['output_file_name_format']
                execution['sequence']+=1
                execution=adddatatoexecution(execution,parameters)
                
                information_to_print=execution['analyzed_model']
                
                last_time=logmessage(f'{information_to_print} - Start',last_time,parameters)
            
                X_metamodel_train.replace({'NB': 1, 'AB': 0}, inplace=True)
                
                X_metamodel_validation.replace({'NB': 1, 'AB': 0}, inplace=True)
                
                X_metamodel_test.replace({'NB': 1, 'AB': 0}, inplace=True)
            
                train_class.replace({1:'NB', 0: 'AB'}, inplace=True)
                validation_class.replace({1:'NB', 0: 'AB'}, inplace=True)
                test_class.replace({1:'NB', 0: 'AB'}, inplace=True)
            
                score_train,precision_train,recall_train,accuracy_train,f1score_train,roc_auc_train,tp_train,tn_train,fp_train,fn_train,strings_train,score_validation,precision_validation,recall_validation,accuracy_validation,f1score_validation,roc_auc_validation,tp_validation,tn_validation,fp_validation,fn_validation,strings_validation,score_test,precision_test,recall_test,accuracy_test,f1score_test,roc_auc_test,tp_test,tn_test,fp_test,fn_test,strings_test,y_train_pred,y_validation_pred,y_test_pred=analyzedata(X_metamodel_train,X_metamodel_validation,X_metamodel_test,train_class,validation_class,test_class,train_metadata,validation_metadata,test_metadata,execution,parameters)
                   
                
                results_to_add_train = {'test_identifier':execution['output_file_name_format'],'TID':execution['TID'],'model_hierarchy':execution['model_hierarchy'],'score': score_train, 'precision': precision_train, 'recall': recall_train, 'accuracy': accuracy_train, 'f1score': f1score_train,'roc_auc':roc_auc_train, 'tp': tp_train, 'tn': tn_train, 'fp': fp_train, 'fn': fn_train, 'model': execution['complete_model'], 'dimensionality_alter': execution['dimensionality_alter'], 'preprocessing': execution['preprocessing'], 'balancing': execution['balancing'], 'optimization': execution['optimization'], 'behaviors_to_use': execution['behaviors_to_use'], 'string1': strings_train[0], 'string2': strings_train[1], 'string3': strings_train[2], 'string4': strings_train[3], 'elapsed_time': round(elapsed_time,2),'precision_diff': round(precision_train-precision_train,3),'recall_diff': round(recall_train-recall_train,3),'accuracy_diff': round(accuracy_train-accuracy_train,3), 'f1score_diff': round(f1score_train-f1score_train,3),'roc_auc_diff': round(roc_auc_train-roc_auc_train,3)}
                results_train = pd.concat([results_train, pd.DataFrame(results_to_add_train, index=[0])], ignore_index=True)
                
                results_to_add_validation = {'test_identifier':execution['output_file_name_format'],'TID':execution['TID'],'model_hierarchy':execution['model_hierarchy'],'score': score_validation, 'precision': precision_validation, 'recall': recall_validation, 'accuracy': accuracy_validation, 'f1score': f1score_validation,'roc_auc': roc_auc_validation, 'tp': tp_validation, 'tn': tn_validation, 'fp': fp_validation, 'fn': fn_validation, 'model': execution['complete_model'],'dimensionality_alter': execution['dimensionality_alter'], 'preprocessing': execution['preprocessing'], 'balancing': execution['balancing'], 'optimization': execution['optimization'], 'behaviors_to_use': execution['behaviors_to_use'], 'string1': strings_validation[0], 'string2': strings_validation[1], 'string3': strings_validation[2], 'string4': strings_validation[3], 'elapsed_time': round(elapsed_time,2),'precision_diff': round(precision_validation-precision_train,3),'recall_diff': round(recall_validation-recall_train,3),'accuracy_diff': round(accuracy_validation-accuracy_train,3), 'f1score_diff': round(f1score_validation-f1score_train,3),'roc_auc_diff': round(roc_auc_validation-roc_auc_train,3) }
                results_validation = pd.concat([results_validation, pd.DataFrame(results_to_add_validation, index=[0])], ignore_index=True)
                
                results_to_add_test = {'test_identifier':execution['output_file_name_format'],'TID':execution['TID'],'model_hierarchy':execution['model_hierarchy'],'score': score_test, 'precision': precision_test, 'recall': recall_test, 'accuracy': accuracy_test, 'f1score': f1score_test,'roc_auc': roc_auc_test, 'tp': tp_test, 'tn': tn_test, 'fp': fp_test, 'fn': fn_test, 'model': execution['complete_model'],'dimensionality_alter': execution['dimensionality_alter'], 'preprocessing': execution['preprocessing'], 'balancing': execution['balancing'], 'optimization': execution['optimization'], 'behaviors_to_use': execution['behaviors_to_use'], 'string1': strings_test[0], 'string2': strings_test[1], 'string3': strings_test[2], 'string4': strings_test[3], 'elapsed_time': round(elapsed_time,2),'precision_diff': round(precision_test-precision_train,3),'recall_diff': round(recall_test-recall_train,3),'accuracy_diff': round(accuracy_test-accuracy_train,3), 'f1score_diff': round(f1score_test-f1score_train,3),'roc_auc_diff': round(roc_auc_test-roc_auc_train,3) }
                results_test = pd.concat([results_test, pd.DataFrame(results_to_add_test, index=[0])], ignore_index=True)
                
                results_train=results_train.sort_values(by='score') 
                results_validation=results_validation.sort_values(by='score') 
                results_test=results_test.sort_values(by='score')       
                  
            
                
                results_train.to_excel(output_xls_results_file_name_train, index=False)
                results_validation.to_excel(output_xls_results_file_name_validation, index=False)
                results_test.to_excel(output_xls_results_file_name_test, index=False)
                
                last_time=logmessage(f'{information_to_print} - End',last_time,parameters)
         
            
            addtocompressedfile(parameters['General']['compressed_file_name'], output_xls_results_file_name_train)
            addtocompressedfile(parameters['General']['compressed_file_name'], output_xls_results_file_name_validation)
            addtocompressedfile(parameters['General']['compressed_file_name'], output_xls_results_file_name_test)
            
            
            last_time=logmessage(f'Load data time {load_time}',last_time,parameters)
            end_iteration_time=datetime.now()
            total_iteration_time=end_iteration_time-start_iteration_time
            start_iteration_time=datetime.now()
            last_time=logmessage(f'Execution ended - Total time {total_iteration_time}',last_time,parameters)
            
            if n== batch_size:
                end_batch_time=datetime.now()
                total_batch_time=end_batch_time-start_batch_time
                last_time=logmessage(f'Batch ended - Total time {total_batch_time}',last_time,parameters)
            
            addtocompressedfile(parameters['General']['compressed_file_name'], process_output_log_file)
            addtocompressedfile(parameters['General']['compressed_file_name'], load_data_output_log_file)
            
            
            if n==batch_size:
                sys.stdout.flush()
                sys.stderr.flush()
                addtocompressedfile(parameters['General']['compressed_file_name'], console_log_file_name)
                addtocompressedfile(parameters['General']['compressed_file_name'], console_error_file_name)
                
            else:
                sys.stdout.flush()
                sys.stderr.flush()
                addtocompressedfile(parameters['General']['compressed_file_name'], console_log_file_name, erase='No')
                addtocompressedfile(parameters['General']['compressed_file_name'], console_error_file_name, erase='No')
    
        if os.path.exists(original_load_data_output_log_file):
            os.remove(original_load_data_output_log_file)
    
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        console_log_file.close()
        console_error_file.close()
    
if __name__ == "__main__":
    main()