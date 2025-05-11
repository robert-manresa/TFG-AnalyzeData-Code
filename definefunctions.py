#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from random import randint
import numpy as np
import re


def filenamesdefinition(case_to_be_tested,parameters,temporality_index):
    results_files_path=parameters['General']['results_files_path']
    output_file_name_format=datetime.now().strftime('%Y%m%dT%H%M%S')+'-'+case_to_be_tested
    process_output_log_file=results_files_path+output_file_name_format+'-ProcessLogFile.log'
    loaddata_output_log_file=results_files_path+output_file_name_format+'-LoadDataLogFile.log'
    
    parameters['General']['results_files_path']=results_files_path
    parameters['General']['output_file_name_format']=output_file_name_format
    parameters['General']['compressed_file_name']=results_files_path+output_file_name_format+'.7z'
    parameters['General']['code_used_compressed_file_name']=results_files_path+'CodeUsed'+'.7z'
    parameters['General']['data_used_compressed_file_name']=results_files_path+'DataUsed'+'.7z'
    parameters['General']['analisys_definition']=results_files_path+'AnalysisDefinition'+'.json'
    parameters['General']['process_output_log_file']=process_output_log_file
    parameters['General']['load_data_output_log_file']=loaddata_output_log_file
    parameters['General']['default_log_type']='screen+file'
    return (parameters)

def parametersdefinition (case_to_be_tested,temporality_index):
    
    results_files_path='/home/grau/TFG/Resultats/'
    output_file_name_format=datetime.now().strftime('%Y%m%dT%H%M%S')+'-'+case_to_be_tested
    process_output_log_file=results_files_path+output_file_name_format+'-ProcessLogFile.log'
    load_data_output_log_file=results_files_path+output_file_name_format+'-LoadDataLogFile.log'

    label_to_get="label"
    behavior_to_get="behavior"
    behavior_type_to_get="behavior_type"

    code_to_save=('AnalyzeDataTFG.py','processfunctions.py','definefunctions.py','inputfunctions.py','outputfunctions.py',)
    json_version='RMM-1.0'

    
   
    if (json_version=='RMM-1.0'):
        columns_to_get=[
            "total_processes",
            "kernel_processes",
            "nonkernel_processes",
            "user_cpu",
            "nice_cpu",
            "system_cpu",
            "idle_cpu",
            "iowait_cpu",
            "irq_cpu",
            "softirq_cpu",
            "steal_cpu",
            "guest_cpu",
            "guest_nice_cpu",
            "interrupt_cpu",
            "dpc_cpu",
            "total_mem",
            "available_mem",
            "used_mem",
            "free_mem",
            "active_mem",
            "inactive_mem",
            "buffers_mem",
            "cached_mem",
            "shared_mem",
            "slab_mem",
            "buff_cache_mem",
            "total_mem_perc",
            "available_mem_perc",
            "used_mem_perc",
            "free_mem_perc",
            "active_mem_perc",
            "inactive_mem_perc",
            "buffers_mem_perc",
            "cached_mem_perc",
            "shared_mem_perc",
            "slab_mem_perc",
            "buff_cache_mem_perc",
            "swap_total_mem",
            "swap_used_mem",
            "swap_free_mem",
            "swap_sin",
            "swap_sout",
            "swap_total_mem_perc",
            "swap_used_mem_perc",
            "swap_free_mem_perc",
            "sent_bytes",
            "received_bytes",
            "sent_packets",
            "received_packets",
            "sent_bytes_per_sec",
            "received_bytes_per_sec",
            "sent_packets_per_sec",
            "received_packets_per_sec",
            "err_in",
            "err_out",
            "drop_in",
            "drop_out",
            "new_connections",
            "closed_connections",
            "current_connections",
            "current_protocol_family_AF_INET",
            "current_protocol_family_AF_INET6",
            "current_protocol_family_AF_UNIX",
            "current_protocol_family_OTHER",
            "current_protocol_type_SOCK_STREAM",
            "current_protocol_type_SOCK_DGRAM",
            "current_protocol_type_SOCK_SEQPACKET",
            "current_protocol_type_0",
            "current_protocol_type_OTHER",
            "current_country_CN",
            "current_country_RU",
            "current_country_EMPTY_IP",
            "current_country_OTHER",
            "current_continent_AF",
            "current_continent_AN",
            "current_continent_AS",
            "current_continent_EU",
            "current_continent_NA",
            "current_continent_OC",
            "current_continent_SA",
            "current_continent_EMPTY_IP",
            "current_continent_OTHER",
            "current_status_CLOSE_WAIT",
            "current_status_CLOSED",
            "current_status_ESTABLISHED",
            "current_status_FIN_WAIT1",
            "current_status_FIN_WAIT2",
            "current_status_LISTEN",
            "current_status_NONE",
            "current_status_SYN_SENT",
            "current_status_TIME_WAIT",
            "current_status_LAST_ACK",
            "current_status_CLOSING",
            'current_status_SYN_RECV',
            "current_status_OTHER",
            "disk_reads",
            "disk_writes",
            "disk_reads_bytes",
            "disk_writes_bytes",
            "disk_reads_time",
            "disk_writes_time",
            "disk_reads_merged",
            "disk_writes_merged",
            "disk_busy_time",
            "disk_reads_per_sec",
            "disk_writes_per_sec",
            "disk_reads_merged_per_sec",
            "disk_writes_merged_per_sec",
            "disk_reads_per_sec_bytes",
            "disk_writes_per_sec_bytes",
            # "files_read", # Future use
            # "files_write", # Future use
            # "files_created", # Future use
            # "files_deleted", # Future use
            # "command_min", # Future use
            # "gpu_proc", # Future use
            # "gpu_mem" # Future use
        ]
       
        metadata_to_get=[
            
            "collection_version",            
            "iteration_number",
            "iteration_started_at",
            "machine_name",
            "machine_platform",
            "machine_identifier"
        ]
        execution_time_to_get=[
            "iteration_time",
            "iteration_execution_time",
            "iteration_wait_time"    
            ]
    else:
        print('Unexpected JSON file version')
        exit()
    
    parameters = {
        'General':{
            'images_resolution':150, 
            'results_files_path':results_files_path,
            'output_file_name_format':output_file_name_format,
            'compressed_file_name':results_files_path+output_file_name_format+'.7z',
            'code_used_compressed_file_name':results_files_path+'CodeUsed'+'.7z',
            'data_used_compressed_file_name':results_files_path+'DataUsed'+'.7z',
            'analisys_definition':results_files_path+'AnalysisDefinition'+'.json',
            'process_output_log_file':process_output_log_file,
            'load_data_output_log_file':load_data_output_log_file,
            'default_log_type':'screen+file',
            'timestamp_format':'%Y-%m-%dT%H:%M:%S.%fZ',
            'code_to_save':code_to_save,
            'label_to_get':label_to_get,
            'behavior_to_get':behavior_to_get,
            'behavior_type_to_get':behavior_type_to_get,     
            'all_classes':['NB','AB'],
            'columns_to_get':columns_to_get,
            'metadata_to_get':metadata_to_get,
            'execution_time_to_get':execution_time_to_get,
            'class_to_use':'behavior_type',
            'random_number':1977,
            'temporality_index':temporality_index
            },
        'Model':{
            'Detect-IF': {
                'optimization_parameters':{
                    'n_estimators':range(3,100,5),
                    'max_samples':range(5,60,5),
                    'contamination':np.arange(0.05,0.4,0.05),
                    'bootstrap':(True,False)
                    },
                'default_parameters':{
                    'bootstrap': False,
                    'contamination': 'auto',
                    'max_features': 1.0,
                    'max_samples': 'auto',
                    'n_estimators': 100,
                    'n_jobs': 8,
                    'random_state': None,
                    'verbose': 0,
                    'warm_start': False
                    }
                },
            'Detect-OCSVM': {
                'optimization_parameters':{
                    'nu':np.arange(0.1,0.9,0.005),
                    'kernel':('linear','rbf','poly','sigmoid')
                    },
                'default_parameters':{
                    'cache_size': 200,
                    'coef0': 0.0,
                    'degree': 3,
                    'gamma': 'scale',
                    'kernel': 'rbf',
                    'max_iter': -1,
                    'nu': 0.5,
                    'shrinking': True,
                    'tol': 0.001,
                    'verbose': False
                    }
                },
            'Detect-LOF': {
                'optimization_parameters':{
                    'n_neighbors':range(20,80,5),
                    'algorithm':('auto','ball_tree','kd_tree','brute',),
                    'metric':('euclidean','manhattan','minkowski','chebyshev',),
                    'p':range(1,4,1),
                    'contamination':np.arange(0.05, 0.45, 0.1),
                    'novelty':(False)
                    },
                'default_parameters':{
                    'algorithm': 'auto',
                    'contamination': 'auto',
                    'leaf_size': 30,
                    'metric': 'minkowski',
                    'metric_params': None,
                    'n_jobs': 8,
                    'n_neighbors': 20,
                    'novelty': True,
                    'p': 2
                    }
                },
            'ClassKNN': {
                'optimization_parameters':{
                    'n_neighbors':range(5,55,10),
                    'p':(1,2,3,4,5),
                    'weights':('distance','uniform'),
                    'algorithm':('auto','ball_tree','kd_tree','brute'),
                    'metric':('euclidean','manhattan','chebyshev','minkowski')
                    },
                'default_parameters':{
                    'algorithm': 'auto',
                    'leaf_size': 30,
                    'metric': 'minkowski',
                    'metric_params': None,
                    'n_jobs': 8,
                    'n_neighbors': 5,
                    'p': 2,
                    'weights': 'uniform'                    
                    }
                },
            'ClassSVM': {
                'optimization_parameters':{
                    'C':np.arange(0.1,10.1,1),
                    'kernel':('linear','poly','rbf','sigmoid'),
                    'degree':range(1,5,1)
                    },
                'default_parameters':{
                    'C': 1.0,
                    'break_ties': False,
                    'cache_size': 200,
                    'class_weight': None,
                    'coef0': 0.0,
                    'decision_function_shape': 'ovr',
                    'degree': 3,
                    'gamma': 'scale',
                    'kernel': 'rbf',
                    'max_iter': -1,
                    'probability': False,
                    'random_state': None,
                    'shrinking': True,
                    'tol': 0.001,
                    'verbose': False
                    }
                },
            'ClassNB': {
                'optimization_parameters':{
                    'var_smoothing':(1e-12,1e-11,1e-10,1e-9, 1e-8, 1e-7, 1e-6, 1e-5)
                    },
                'default_parameters':{
                    'priors': None,
                    'var_smoothing': 1e-09
                    }
                },
            'ClassLR': {
                'optimization_parameters':{
                    'penalty':('l2','l1','elasticnet',None),
                    'C':np.arange(0.8,1.4,.2),
                    'solver':('lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga')
                    },
                'default_parameters':{
                    'C': 1.0,
                    'class_weight': None,
                    'dual': False,
                    'fit_intercept': True,
                    'intercept_scaling': 1,
                    'l1_ratio': None,
                    'max_iter': 100,
                    # 'multi_class': 'auto', # Deprecated
                    'n_jobs': 8,
                    'penalty': 'l2',
                    'random_state': None,
                    'solver': 'lbfgs',
                    'tol': 0.0001,
                    'verbose': 0,
                    'warm_start': False
                    }
                },
            'ClassDT': {
                'optimization_parameters':{
                    'criterion':('gini','entropy','log_loss'),
                    'splitter':('best','random'),
                    'max_features':('sqrt','log2',None)
                    },
                'default_parameters':{
                    'ccp_alpha': 0.0,
                    'class_weight': None,
                    'criterion': 'gini',
                    'max_depth': None,
                    'max_features': None,
                    'max_leaf_nodes': None,
                    'min_impurity_decrease': 0.0,
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'min_weight_fraction_leaf': 0.0,
                    'random_state': None,
                    'splitter': 'best'
                    }
                },
            'ClassRF': {
                'optimization_parameters':{
                    'n_estimators':range(10,150,10),
                    'criterion':('gini','entropy','log_loss'),
                    'max_features':('sqrt','log2',None)
                    },
                'default_parameters':{
                    'bootstrap': True,
                    'ccp_alpha': 0.0,
                    'class_weight': None,
                    'criterion': 'gini',
                    'max_depth': None,
                    'max_features': 'sqrt',
                    'max_leaf_nodes': None,
                    'max_samples': None,
                    'min_impurity_decrease': 0.0,
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'min_weight_fraction_leaf': 0.0,
                    'n_estimators': 100,
                    'n_jobs': 8,
                    'oob_score': False,
                    'random_state': None,
                    'verbose': 0,
                    'warm_start': False
                    }
                },
            'ClassGBM': {
                'optimization_parameters':{
                    'n_estimators':range(10,150,10),
                    'learning_rate':(0.001,0.01,0.1,0.2),
                    'max_features':('sqrt','log2', None,)
                    },
                'default_parameters':{
                    'ccp_alpha': 0.0,
                    'criterion': 'friedman_mse',
                    'init': None,
                    'learning_rate': 0.1,
                    'loss': 'log_loss',
                    'max_depth': 3,
                    'max_features': None,
                    'max_leaf_nodes': None,
                    'min_impurity_decrease': 0.0,
                    'min_samples_leaf': 1,
                    'min_samples_split': 2,
                    'min_weight_fraction_leaf': 0.0,
                    'n_estimators': 100,
                    'n_iter_no_change': None,
                    'random_state': None,
                    'subsample': 1.0,
                    'tol': 0.0001,
                    'validation_fraction': 0.1,
                    'verbose': 0,
                    'warm_start': False
                    }
                },
            'ClassABC': {
                'optimization_parameters':{
                    'estimator':(None,),
                    'n_estimators': range(10,150,10),
                    'learning_rate':(0.1,1,10,20),
                    'algorithm':('SAMME', 'SAMME.R',)
                    },
                'default_parameters':{
                    'algorithm': 'SAMME',
                    # 'base_estimator': 'deprecated',
                    'estimator': None,
                    'learning_rate': 1.0,
                    'n_estimators': 50,
                    'random_state': None
                    }
                },
            'ClassANN': {
                'optimization_parameters':{
                    'estimator':(None,),
                    'n_estimators': range(10,150,10),
                    'learning_rate':(0.1,1,10,20),
                    'algorithm':('SAMME', 'SAMME.R',)
                    },
                'default_parameters':{
                    'algorithm': 'SAMME.R',
                    'base_estimator': 'deprecated',
                    'estimator': None,
                    'learning_rate': 1.0,
                    'n_estimators': 50,
                    'random_state': None
                    }
                },
            'ClassDNN': {
                'optimization_parameters':{
                    'estimator':(None,),
                    'n_estimators': range(10,150,10),
                    'learning_rate':(0.1,1,10,20),
                    'algorithm':('SAMME', 'SAMME.R',)
                    },
                'default_parameters':{
                    'algorithm': 'SAMME.R',
                    'base_estimator': 'deprecated',
                    'estimator': None,
                    'learning_rate': 1.0,
                    'n_estimators': 50,
                    'random_state': None
                    }
                },
            'ClassXBG': {
                'optimization_parameters':{
                    'estimator':(None,),
                    'n_estimators': range(10,150,10),
                    'learning_rate':(0.1,1,10,20),
                    'algorithm':('SAMME', 'SAMME.R',)
                    },
                'default_parameters':{
                    'objective': 'binary:logistic',
                    'base_score': None,
                    'booster': None,
                    'callbacks': None,
                    'colsample_bylevel': None,
                    'colsample_bynode': None,
                    'colsample_bytree': None,
                    'device': None,
                    'early_stopping_rounds': None,
                    'enable_categorical': False,
                    'eval_metric': None,
                    'feature_types': None,
                    'gamma': None,
                    'grow_policy': None,
                    'importance_type': None,
                    'interaction_constraints': None,
                    'learning_rate': None,
                    'max_bin': None,
                    'max_cat_threshold': None,
                    'max_cat_to_onehot': None,
                    'max_delta_step': None,
                    'max_depth': None,
                    'max_leaves': None,
                    'min_child_weight': None,
                    'missing': np.nan,
                    'monotone_constraints': None,
                    'multi_strategy': None,
                    'n_estimators': None,
                    'n_jobs': 8,
                    'num_parallel_tree': None,
                    'random_state': None,
                    'reg_alpha': None,
                    'reg_lambda': None,
                    'sampling_method': None,
                    'scale_pos_weight': None,
                    'subsample': None,
                    'tree_method': None,
                    'validate_parameters': None,
                    'verbosity': None
                }
                }
            }
        }

    return(parameters)

def removertid(string):
    output=re.sub(r"-TID\d{1,3}$", "", string)
    return output

def tidvalidation(string):
    output=bool(re.search(r"TID\d{1,3}$", string))
    return (output) 


def casestobetesteddefinition(cases_name_TID,temporality_index):
    cases_to_be_tested={}
    meta_models={}
    cases_to_be_tested['name']=(cases_name_TID,)
    meta_models['name']=(cases_name_TID,)
    
    cases_name=removertid(cases_name_TID)

    if (cases_name=='TEST'):
        
        cases_to_be_tested['known_data_to_be_loaded']=('TEST',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TEST',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('TEST',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          )
       
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              )

    elif (cases_name=='TEST1'):
        
        cases_to_be_tested['known_data_to_be_loaded']=('TEST1',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TEST1',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('TEST1',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          )
       
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              )

    elif (cases_name=='TEST2'):
        
        cases_to_be_tested['known_data_to_be_loaded']=('TEST2',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TEST2',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('TEST2',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          )
       
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              )


        
    elif (cases_name=='TESTDNN'):
        cases_to_be_tested['known_data_to_be_loaded']=('TEST',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('TEST',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          )
       
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              )
        
    elif (cases_name=='EXAMPLE-A-TO-B'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_B',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          
        )
        meta_models['meta_model']=(                     
                               )
    elif (cases_name=='EXAMPLE-B-TO-A'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_A',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          
        )
        meta_models['meta_model']=()
        
    elif (cases_name=='EXAMPLE-A-TO-B-1R'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_A_PRESENT_245K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('TESTSET_B_PRESENT_245K',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
        )
        meta_models['meta_model']=(                     
                               )

    elif (cases_name=='EXAMPLE-B-TO-A-1R'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_B_PRESENT_245K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('TESTSET_A_PRESENT_245K',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
        )
        meta_models['meta_model']=()
        
    
    elif (cases_name=='ALLMODELS-TRAINSET_A-DATABASE_B'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_B',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    elif (cases_name=='ALLMODELS-TRAINSET_B-DATABASE_A'):
            cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B',)
            cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_A',)
            cases_to_be_tested['data_platform_to_be_used']=('All',)
            cases_to_be_tested['data_type']=('Unknown',)
            cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassKNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassSVM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassSVM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
            )
            meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                  'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                  'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                  'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                  'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                  'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                  'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                  'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                  'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                  'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                                  )
    
    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_50K-DATABASE_B'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('DATABASE_A',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_B',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_50K-TESTSET_B_PRESENT_245K'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_A_PRESENT_245K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('TESTSET_B_PRESENT_245K',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_50K-TESTSET_A_PRESENT_245K'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_B_PRESENT_245K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('TESTSET_A_PRESENT_245K',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_50K-TESTSET_A_NOTPRESENT_15K'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_B_NOTPRESENT_15K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('TESTSET_A_NOTPRESENT_15K',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_50K-TESTSET_B_NOTPRESENT_15K'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_A_NOTPRESENT_15K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('TESTSET_B_NOTPRESENT_15K',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    
    
    
    
    
    
    
    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_50K-TESTSET_A_200K'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_50K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('TESTSET_A_200K',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
        
    elif (cases_name=='ALLXNNMODELS-TRAINSET_A_50K-DATABASE_B'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_50K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_B',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassDNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',

        )
        meta_models['meta_model']=(
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_50K-DATABASE_A'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_50K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_A',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_100K-DATABASE_B'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_100K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_B',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_100K-DATABASE_A'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_100K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_A',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )

    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_50K-DATABASE_C'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('DATABASE_A',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_50K-DATABASE_C_PRESENT'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_A_PRESENT_245K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C_PRESENT',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_50K-DATABASE_C_PRESENT'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_B_PRESENT_245K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C_PRESENT',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_50K-DATABASE_C_NOTPRESENT'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_A_NOTPRESENT_15K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C_NOTPRESENT',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
        
        
    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_50K-DATABASE_C_RANSOMWARE'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_A_NOTPRESENT_15K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C_RANSOMWARE',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_50K-DATABASE_C_RANSOMWARE'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_B_NOTPRESENT_15K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C_RANSOMWARE',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_50K-DATABASE_C_BRUTEPASSWORD'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_A_NOTPRESENT_15K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C_BRUTEPASSWORD',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_50K-DATABASE_C_BRUTEPASSWORD'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_B_NOTPRESENT_15K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C_BRUTEPASSWORD',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_50K-DATABASE_C_RANSOMWAREBRUTEPASSWORD'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_A_NOTPRESENT_15K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C_RANSOMWAREBRUTEPASSWORD',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_50K-DATABASE_C_RANSOMWAREBRUTEPASSWORD'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_B_NOTPRESENT_15K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C_RANSOMWAREBRUTEPASSWORD',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )  
        
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_50K-DATABASE_C_NOTPRESENT'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_B_NOTPRESENT_15K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C_NOTPRESENT',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
        
    
    
    
        
    elif (cases_name=='ALLFASTMODELS-TRAINSET_AB_50K-DATABASE_C'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_AB_50K',)
        cases_to_be_tested['validation_data_to_be_loaded']=('TESTSET_B_PRESENT_245K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
        
    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_53K-DATABASE_C'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_53K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_53K-DATABASE_C'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_53K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_50K-DATABASE_C'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_50K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )

    elif (cases_name=='ALLFASTMODELS-TRAINSET_A_100K-DATABASE_C'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A_100K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    
    elif (cases_name=='ALLFASTMODELS-TRAINSET_B_100K-DATABASE_C'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B_100K',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_C',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
        
    
    elif (cases_name=='ALLMODELS-TRAINSET_A-DATABASE_A'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_A',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_A',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
        
    elif (cases_name=='ALLMODELS-TRAINSET_B-DATABASE_B'):
        cases_to_be_tested['known_data_to_be_loaded']=('TRAINSET_B',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('DATABASE_B',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
        )
        meta_models['meta_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassLR-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassANN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'                           
                              )
    ### GBMM-TID1 - BEGIN ###
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBMM70-BALANCEDKNOWNDATA-TID1'):
                 temporality_index=1
                 cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                 cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                 cases_to_be_tested['data_platform_to_be_used']=('All',)
                 cases_to_be_tested['data_type']=('Unknown',)
                 cases_to_be_tested['base_model']=('ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassRF-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   
                                                   'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                 )
         
                 meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                            )
                 
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBMM5C-BALANCEDKNOWNDATA-TID1'):
                 temporality_index=1
                 cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                 cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                 cases_to_be_tested['data_platform_to_be_used']=('All',)
                 cases_to_be_tested['data_type']=('Unknown',)
                 cases_to_be_tested['base_model']=('ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   
                                                   'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   
                                                   'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                 )
         
                 meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                            )
    ### GBMM-TID1 - END ###
    ### GBMM-TID2 - BEGIN ###
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBMM70-BALANCEDKNOWNDATA-TID2'):
                 temporality_index=2
                 cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                 cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                 cases_to_be_tested['data_platform_to_be_used']=('All',)
                 cases_to_be_tested['data_type']=('Unknown',)
                 cases_to_be_tested['base_model']=('ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   
                                                   'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassABC-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  
                                                   'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   
                                                   'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassNB-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   
                                                   'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   
                                                   'ClassKNN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   
                                                   'ClassSVM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
    
                                                   'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassXBG-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                 )
         
                 meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                            )
                 
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBMM5C-BALANCEDKNOWNDATA-TID2'):
                     temporality_index=2
                     cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                     cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                     cases_to_be_tested['data_platform_to_be_used']=('All',)
                     cases_to_be_tested['data_type']=('Unknown',)
                     cases_to_be_tested['base_model']=('ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                       
                                                       'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                       'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                       
                                                       'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                       
                                                       'ClassSVM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                     )
             
                     meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                )
    ### GBMM-TID2 - END ###
    
    ### GBMM-TID3 - BEGIN ###
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBMM70-BALANCEDKNOWNDATA-TID3'):
                 temporality_index=3
                 cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                 cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                 cases_to_be_tested['data_platform_to_be_used']=('All',)
                 cases_to_be_tested['data_type']=('Unknown',)
                 cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   
                                                   'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   
                                                   'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                 )
         
                 meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                            )
    ### GBMM-TID3 - END ###
    
    ### IGBMM-TID3 - BEGIN ###
    elif (cases_name=='INVERSE-SELECTEDBASEMODEL-METAMODEL-GBMM70-BALANCEDKNOWNDATA-TID3'):
                 temporality_index=3
                 cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-UNKNOWN-DATA',)
                 cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-KNOWN-DATA',)
                 cases_to_be_tested['data_platform_to_be_used']=('All',)
                 cases_to_be_tested['data_type']=('Unknown',)
                 cases_to_be_tested['base_model']=('ClassRF-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   
                                                   'ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   
                                                   'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                                                   'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
    
                 )
         
                 meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                            )
    ### GBMM-TID3 - END ###
    
    
    ### GBM-TID1 - BEGIN ###
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBM70-BALANCEDKNOWNDATA-TID1'):
                cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                cases_to_be_tested['data_platform_to_be_used']=('All',)
                cases_to_be_tested['data_type']=('Unknown',)
                cases_to_be_tested['base_model']=('ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                )
        
                meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                           )
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBM90-BALANCEDKNOWNDATA-TID1'):
                    cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                    cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                    cases_to_be_tested['data_platform_to_be_used']=('All',)
                    cases_to_be_tested['data_type']=('Unknown',)
                    cases_to_be_tested['base_model']=('ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                    )
            
                    meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                               )
    ### GBM-TID1 - END ###
    
    ### GBM-TID2 - BEGIN ###
    
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBM70-BALANCEDKNOWNDATA-TID2'):
                    cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                    cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                    cases_to_be_tested['data_platform_to_be_used']=('All',)
                    cases_to_be_tested['data_type']=('Unknown',)
                    cases_to_be_tested['base_model']=('ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
                    )
            
                    meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                               )
    
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBM90-BALANCEDKNOWNDATA-TID2'):
                    cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                    cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                    cases_to_be_tested['data_platform_to_be_used']=('All',)
                    cases_to_be_tested['data_type']=('Unknown',)
                    cases_to_be_tested['base_model']=('ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
                    )
            
                    meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                               )
                    
    ### GBM-TID2 - END ###
    
    
    
    
    ### GBM-TID3 - BEGIN ###
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBM70-BALANCEDKNOWNDATA-TID3'):
                    cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                    cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                    cases_to_be_tested['data_platform_to_be_used']=('All',)
                    cases_to_be_tested['data_type']=('Unknown',)
                    cases_to_be_tested['base_model']=('ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                    )
            
                    meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                               )
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBM80-BALANCEDKNOWNDATA-TID3'):
                    cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                    cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                    cases_to_be_tested['data_platform_to_be_used']=('All',)
                    cases_to_be_tested['data_type']=('Unknown',)
                    cases_to_be_tested['base_model']=('ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                      'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                    )
            
                    meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                               )
                    
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-GBM90-BALANCEDKNOWNDATA-TID3'):
                        cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                        cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                        cases_to_be_tested['data_platform_to_be_used']=('All',)
                        cases_to_be_tested['data_type']=('Unknown',)
                        cases_to_be_tested['base_model']=('ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                          'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                          'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                          'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                          'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                          'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                          'ClassANN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                          
                        )
                
                        meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                   )
    ### GBM-TID3 - END ###
    
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-RF84-BALANCEDKNOWNDATA-TID1'):
            cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
            cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
            cases_to_be_tested['data_platform_to_be_used']=('All',)
            cases_to_be_tested['data_type']=('Unknown',)
            cases_to_be_tested['base_model']=('ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassGBM-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassNB-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassNB-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassDT-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',                                              
                                              'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassSVM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassANN-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassANN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
                                              )
            meta_models['meta_model']=('ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                       )
    
    elif (cases_name=='SELECTEDBASEMODEL-METAMODEL-RF70-BALANCEDKNOWNDATA-TID1'):
                cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                cases_to_be_tested['data_platform_to_be_used']=('All',)
                cases_to_be_tested['data_type']=('Unknown',)
                cases_to_be_tested['base_model']=('ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassGBM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassABC-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassLR-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',                                              
                                                  'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassSVM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassSVM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
                                                  )
                meta_models['meta_model']=('ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                           )
    
    
    
    elif (cases_name=='SELECTEDBASEMODEL-PROVAGBMMETAMODEL-BALANCEDKNOWNDATA-TID1'):
        cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
        cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
        cases_to_be_tested['data_platform_to_be_used']=('All',)
        cases_to_be_tested['data_type']=('Unknown',)
        cases_to_be_tested['base_model']=('ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',  
                                          'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',                                              
                                          'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                          )
        
        meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                           )
    
    elif (cases_name=='SELECTEDBASEMODEL-RFMETAMODEL-BALANCEDKNOWNDATA-TID1'):
            cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
            cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
            cases_to_be_tested['data_platform_to_be_used']=('All',)
            cases_to_be_tested['data_type']=('Unknown',)
            cases_to_be_tested['base_model']=('ClassGBM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassGBM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassNB-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassDT-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassLR-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassKNN-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassKNN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassSVM-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassSVM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassANN-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassXBG-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                              'ClassXBG-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)'
                                              )
            meta_models['meta_model']=('ClassRF-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                       )
    elif (cases_name=='SELECTEDBASEMODEL-GBMMETAMODEL2-BALANCEDKNOWNDATA-TID1'):
                cases_to_be_tested['known_data_to_be_loaded']=('BALANCED-KNOWN-DATA',)
                cases_to_be_tested['unknown_data_to_be_loaded']=('ALL-RED-ALL-COMP',)
                cases_to_be_tested['data_platform_to_be_used']=('All',)
                cases_to_be_tested['data_type']=('Unknown',)
                cases_to_be_tested['base_model']=('ClassRF-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassRF-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassABC-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassABC-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassNB-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassDT-Normalize-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassDT-StanScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassDT-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassKNN-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassKNN-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassSVM-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassSVM-QuanTran-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                                  'ClassXBG-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',                                              
                                                  'ClassXBG-RobuScal-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                )
        
                meta_models['meta_model']=('ClassGBM-NoPreprocess-NoDimAlter-NoBal-NoOpt-Train(AB+NB)',
                                           )
    else: 
        print(f'Unexpected test name: {cases_name}')
        exit()
    return(cases_to_be_tested,meta_models)

