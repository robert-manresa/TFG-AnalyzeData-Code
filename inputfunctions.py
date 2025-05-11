#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
import random
import subprocess
import os
from outputfunctions import addtocompressedfile

def getlabelcategories(label):
    label = label.upper()
    identifier = label[-2:]
    behavior_to_test = ['BOINC','BACKUP', 'DOTA', 'HPING3', 'IDLE', 'NETFLIX', 'NMAP', 'NPING', 'OPENTTD', 'PLUTOTV', 'RUNTIMETV', 'VIMEO', 'YOUTUBE','SIMS','COMPILE','SYSUPGRADE','DOWNLOAD','BRUTELOGIN','BRUTEPASSWORD','RANSOMWARE']
    direction_to_test = ['OUT','IN']
    protocol_to_test = ['TCP', 'UDP', 'ICMP']
    intensity_to_test = ['ONE', 'TWO', 'T3', 'T5']
    
    behavior = next((category for category in behavior_to_test if category in label), '')

    if behavior in ['IDLE',]:
        behavior_type='NB'
        behavior_group = 'IDLE'
        direction=''
        protocol=''
        intensity=''
        statistical_behavior='IDLE'
        statistical_action='IDLE'
    elif behavior in [ 'NETFLIX', 'VIMEO', 'YOUTUBE', 'PLUTOTV', 'RUNTIMETV']:
        behavior_type='NB'
        behavior_group = 'STREAMING'
        direction=''
        protocol=''
        intensity=''
        statistical_behavior='STREAMING'
        statistical_action=behavior
    elif behavior in ['NPING', 'HPING3']:
        behavior_type='AB'
        behavior_group = 'DOS'
        direction = next((category for category in direction_to_test if category in label), '')
        protocol = next((category for category in protocol_to_test if category in label), '')
        intensity = next((category for category in intensity_to_test if category in label), '')
        if direction == 'IN':
            statistical_behavior = 'DENIAL OF SERVICE'
        elif direction == 'OUT':
            statistical_behavior = 'BOTNET'
        else:
            statistical_behavior = 'UNKNOWN'
        statistical_action=behavior+'-'+direction+'-'+protocol+'-'+intensity
        
    elif behavior in ['BRUTELOGIN',]:
        behavior_type='AB'
        behavior_group = 'BRUTEFORCE'
        direction = next((category for category in direction_to_test if category in label), '')
        protocol=''
        intensity=''
        if direction == 'IN':
            statistical_behavior = 'BRUTE LOGIN INBOUND'
        elif direction == 'OUT':
            statistical_behavior = 'BRUTE LOGIN OUTBOUND'
        else:
            statistical_behavior = 'UNKNOWN'
        statistical_action='HYDRA'+'-'+direction
    elif behavior in ['BRUTEPASSWORD',]:
        behavior_type='AB'
        behavior_group = 'BRUTEFORCE'
        direction =''
        protocol=''
        intensity=''
        statistical_behavior='BRUTE PASSWORD'
        statistical_action='JOHN THE RIPPER' 
        
    elif behavior in ['OPENTTD', 'DOTA','SIMS']:
        behavior_type='NB'
        behavior_group = 'GAMING'
        direction=''
        protocol=''
        intensity=''
        statistical_behavior='GAMING'
        statistical_action=behavior
    elif behavior in ['NMAP']:
        behavior_type='AB'
        behavior_group = 'PORTSCANNING'
        direction = next((category for category in direction_to_test if category in label), '')
        protocol = next((category for category in protocol_to_test if category in label), '')
        intensity = next((category for category in intensity_to_test if category in label), '')
        if direction == 'IN':
            statistical_behavior = 'PORT SCANNING INBOUND'
        elif direction == 'OUT':
            statistical_behavior = 'PORT SCANNING OUTBOUND'
        else:
            statistical_behavior = 'UNKNOWN'
        statistical_action=behavior+'-'+direction
    elif behavior in ['BOINC']:
        behavior_type='NB'
        behavior_group = 'OTHER'
        direction=''
        protocol=''
        intensity=''
        statistical_behavior='DISTRIBUTED COMPUTING'
        statistical_action=behavior
    elif behavior in ['BACKUP']:
        behavior_type='NB'
        behavior_group = 'OTHER'
        direction=''
        protocol=''
        intensity=''
        statistical_behavior='BACKUP'
        statistical_action='BACKUP'
    elif behavior in ['COMPILE']:
        behavior_type='NB'
        behavior_group = 'OTHER'
        direction=''
        protocol=''
        intensity=''
        statistical_behavior='COMPILING'
        statistical_action='COMPILING'
    elif behavior in ['SYSUPGRADE',]:
        behavior_type='NB'
        behavior_group = 'OTHER'
        direction=''
        protocol=''
        intensity=''
        statistical_behavior='SYSTEM UPGRADE'
        statistical_action='SYSTEM UPGRADE'
    elif behavior in ['DOWNLOAD']:
        behavior_type='NB'
        behavior_group = 'OTHER'
        direction=''
        protocol=''
        intensity=''
        statistical_behavior='DOWNLOADING'
        statistical_action='DOWNLOADING'
    elif behavior in ['RANSOMWARE']:
        behavior_type='AB'
        behavior_group = 'RANSOMWARE'
        direction=''
        protocol=''
        intensity=''
        statistical_behavior='RANSOMWARE'
        statistical_action='RANSOMWARE'
    else:
        print(label)
        behavior_type='None'
        behavior_group = 'None'
        direction='None'
        protocol='None'
        intensity='None'
        statistical_behavior='None'
        statistical_action='None'

    new_label = behavior + direction + protocol + intensity + identifier
    
    if new_label == label: 
        if not direction:
            direction = 'None'
        if not protocol:
            protocol = 'None'
        if not intensity:
            intensity = 'None'
        return {'behavior_type': behavior_type,'behavior_group': behavior_group, 'behavior': behavior, 'direction': direction, 'protocol': protocol, 'intensity': intensity,'statistical_behavior': statistical_behavior,'statistical_action': statistical_action, 'identifier': identifier}
    else: 
        print(f'Detectada la etiqueta {new_label} que no coincide con la etiqueta {label}')
        exit()
        return {'behavior_group': 'None', 'behavior': 'None', 'direction': 'None', 'protocol': 'None', 'intensity': 'None','statistical_behavior':'None','statistica_action':'None', 'identifier': 'None'}

def importdatafrom7zfile(global_path, parameters):
    columns_to_get = parameters['General']['columns_to_get']
    label_to_get = parameters['General']['label_to_get']
    metadata_to_get = parameters['General']['metadata_to_get']
    execution_time_to_get = parameters['General']['execution_time_to_get']
    class_to_use = parameters['General']['class_to_use']

    try:
        result = subprocess.run(['7z', 'e', global_path, '*-Data.json', '-so'], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError('Error leyendo el archivo 7z')

        dataframe = pd.DataFrame(json.loads(result.stdout))
        metadata = dataframe.filter(metadata_to_get)
        metadata = metadata.reindex(columns=metadata_to_get, fill_value='')
        label = dataframe.filter([label_to_get])
        label = label.reindex(columns=[label_to_get], fill_value='')
        categories = label['label'].apply(getlabelcategories)
        categories = pd.json_normalize(categories)
        time = dataframe.filter(execution_time_to_get)
        time = time.reindex(columns=execution_time_to_get, fill_value=0)
        time = time.apply(pd.to_numeric, errors='coerce').fillna(0)
        data = dataframe.filter(columns_to_get)
        data = data.reindex(columns=columns_to_get, fill_value=0)
        
        data = data.replace(-999, 0).apply(pd.to_numeric, errors='coerce').fillna(0)
       
        dataframe = pd.concat([metadata, time, label, categories, data], axis=1)
        dataframe['file_name']=os.path.basename(global_path)
        dataframe['data_class'] = dataframe[[class_to_use]]
        
        nan_exist = dataframe.isna().any().any()
        null_exist = dataframe.isnull().any().any()
        if nan_exist:
            print(f'There are NaN values after reading file {global_path}')
        if null_exist:
            print(f'There are null values after reading file {global_path}')

        return dataframe

    except Exception as error:
        print(f'{global_path}')
        print(f'Error: {error}')
        exit()
        
def searchfilesinfolders(data_definition, extension_to_search,data_platform_to_be_used):
    try: 
        files_info = []
        for name, data_info in data_definition.items():
            folder = data_info['path']
            for file in os.listdir(folder):
                if os.path.isfile(os.path.join(folder, file)) and file.endswith(extension_to_search):
                    file_name, file_extension = os.path.splitext(file)
                    platform = 'Linux' if 'Linux' in file_name else 'Windows' if 'Windows' in file_name else None
                    files_info.append({
                        'filter_field': name,
                        'platform': platform,
                        'file_name': file_name,
                        'file_extension': file_extension,
                        'file_folder': folder,
                        'global_path': os.path.abspath(os.path.join(folder, file))
                    })
                    
        if data_platform_to_be_used=='All':
            pass
        elif (data_platform_to_be_used=='Linux' or data_platform_to_be_used=='Windows'):
            files_info = [temporal_info for temporal_info in files_info if temporal_info['platform'] == data_platform_to_be_used]
        else: 
           print(f'Unexpected data_platform_to_be_used: {data_platform_to_be_used}')
           exit()
        return files_info
    except Exception as error: 
        print(f'Error: {error}')
        exit()
        
def getvalidationdatadefinition(input_definition):
    output_definition = {}
    for key, value in input_definition.items():
        output_definition[key] = {
            'path': value['path'],
            'samples': -1
            }
    return output_definition

def getfilesinfo(cases_to_be_tested):
    known_data_to_be_loaded=cases_to_be_tested['known_data_to_be_loaded'][0]
    validation_data_to_be_loaded=cases_to_be_tested['validation_data_to_be_loaded'][0]
    unknown_data_to_be_loaded=cases_to_be_tested['unknown_data_to_be_loaded'][0]
    data_platform_to_be_used=cases_to_be_tested['data_platform_to_be_used'][0]
    
    extension_to_search='7z'
    
    # Dataset Definitions
    
    TEST_A_SELECTED={
        'IDLE': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/TEST_DATABASE/KnownData/IDLE/', 'samples': 540},
        'HPING3IN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/TEST_DATABASE/KnownData/HPING3/', 'samples': 540},

        }
    TEST_A_FULL={
        'IDLE': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/TEST_DATABASE/KnownData/IDLE/', 'samples': -1},
        'HPING3IN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/TEST_DATABASE/KnownData/HPING3/', 'samples': -1},
        'OPENTTD': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/TEST_DATABASE/KnownData/OPENTTD/', 'samples': -1},
        }
    TEST_B_SELECTED={
        'IDLE': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/TEST_DATABASE/UnknownData/IDLE/', 'samples': 540},
        'HPING3IN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/TEST_DATABASE/UnknownData/HPING3/', 'samples': 540},
        }
    TEST_B_FULL={
        'ALL': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/TEST_DATABASE/UnknownData/ALL/', 'samples': -1},
        }
    
    TEST_1_FULL={
        'ALL': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/TEST_DATABASE/UnknownData/ALL/TEST1/', 'samples': -1},
        }
    
    TEST_2_FULL={
        'ALL': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/TEST_DATABASE/UnknownData/ALL/TEST2/', 'samples': -1},
        }
    
    
    DATABASE_A={
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Windows/', 'samples': -1},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Linux/', 'samples': -1},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Windows/', 'samples': -1},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Linux/', 'samples': -1},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Windows/', 'samples': -1},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Linux/', 'samples': -1},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Windows/', 'samples': -1},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Linux/', 'samples': -1},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Windows/', 'samples': -1},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Linux/', 'samples': -1},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Windows/', 'samples': -1},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Linux/', 'samples': -1},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Windows/', 'samples':-1},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Linux/', 'samples': -1},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Windows/', 'samples': -1},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Linux/', 'samples': -1},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Windows/', 'samples': -1},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Linux/', 'samples': -1},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Windows/', 'samples': -1},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Linux/', 'samples': -1},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Windows/', 'samples': -1},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Linux/', 'samples': -1},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Windows/', 'samples': -1},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Linux/', 'samples': -1},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Compile/Windows/', 'samples': -1},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Compile/Linux/', 'samples': -1},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Download/Windows/', 'samples': -1},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Download/Linux/', 'samples': -1},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/SysUpgrade/Windows/', 'samples': -1},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/SysUpgrade/Linux/', 'samples': -1},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': -1},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': -1},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Windows/', 'samples': -1},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Linux/', 'samples': -1},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Windows/', 'samples': -1},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Linux/', 'samples': -1},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': -1},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': -1},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': -1},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': -1},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': -1},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': -1},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': -1},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': -1},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Windows/', 'samples': -1},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Linux/', 'samples': -1},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Windows/', 'samples': -1},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Linux/', 'samples': -1},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': -1},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': -1},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': -1},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': -1},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': -1},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': -1},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': -1},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': -1},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': -1},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': -1},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': -1},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': -1},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': -1},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': -1},        
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': -1},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': -1},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': -1},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': -1},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginIn/Windows/', 'samples': -1},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginIn/Linux/', 'samples': -1},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginOut/Windows/', 'samples': -1},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginOut/Linux/', 'samples': -1},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BrutePassword/Windows/', 'samples': -1},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BrutePassword/Linux/', 'samples': -1},
        }
    
    
    DATABASE_B={
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Windows/', 'samples': -1},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Linux/', 'samples': -1},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Windows/', 'samples': -1},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Linux/', 'samples': -1},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Windows/', 'samples': -1},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Linux/', 'samples': -1},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Windows/', 'samples': -1},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Linux/', 'samples': -1},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Windows/', 'samples': -1},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Linux/', 'samples': -1},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Windows/', 'samples': -1},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Linux/', 'samples': -1},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Windows/', 'samples':-1},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Linux/', 'samples': -1},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Windows/', 'samples': -1},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Linux/', 'samples': -1},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Windows/', 'samples': -1},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Linux/', 'samples': -1},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Windows/', 'samples': -1},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Linux/', 'samples': -1},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Windows/', 'samples': -1},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Linux/', 'samples': -1},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Windows/', 'samples': -1},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Linux/', 'samples': -1},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Compile/Windows/', 'samples': -1},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Compile/Linux/', 'samples': -1},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Download/Windows/', 'samples': -1},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Download/Linux/', 'samples': -1},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/SysUpgrade/Windows/', 'samples': -1},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/SysUpgrade/Linux/', 'samples': -1},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': -1},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': -1},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Windows/', 'samples': -1},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Linux/', 'samples': -1},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Windows/', 'samples': -1},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Linux/', 'samples': -1},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': -1},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': -1},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': -1},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': -1},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': -1},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': -1},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': -1},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': -1},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Windows/', 'samples': -1},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Linux/', 'samples': -1},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Windows/', 'samples': -1},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Linux/', 'samples': -1},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': -1},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': -1},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': -1},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': -1},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': -1},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': -1},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': -1},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': -1},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': -1},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': -1},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': -1},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': -1},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': -1},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': -1},        
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': -1},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': -1},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': -1},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': -1},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginIn/Windows/', 'samples': -1},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginIn/Linux/', 'samples': -1},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginOut/Windows/', 'samples': -1},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginOut/Linux/', 'samples': -1},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BrutePassword/Windows/', 'samples': -1},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BrutePassword/Linux/', 'samples': -1},
        }
   
    DATABASE_C={
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Windows/', 'samples': -1},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Linux/', 'samples': -1},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Windows/', 'samples': -1},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Linux/', 'samples': -1},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Windows/', 'samples': -1},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Linux/', 'samples': -1},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Windows/', 'samples': -1},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Linux/', 'samples': -1},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Windows/', 'samples': -1},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Linux/', 'samples': -1},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Windows/', 'samples': -1},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Linux/', 'samples': -1},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Windows/', 'samples':-1},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Linux/', 'samples': -1},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Windows/', 'samples': -1},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Linux/', 'samples': -1},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Windows/', 'samples': -1},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Linux/', 'samples': -1},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Windows/', 'samples': -1},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Linux/', 'samples': -1},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Windows/', 'samples': -1},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Linux/', 'samples': -1},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Windows/', 'samples': -1},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Linux/', 'samples': -1},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Windows/', 'samples': -1},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Linux/', 'samples': -1},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Windows/', 'samples': -1},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Linux/', 'samples': -1},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Windows/', 'samples': -1},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Linux/', 'samples': -1},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': -1},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': -1},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Windows/', 'samples': -1},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Linux/', 'samples': -1},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Windows/', 'samples': -1},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Linux/', 'samples': -1},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': -1},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': -1},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': -1},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': -1},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': -1},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': -1},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': -1},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': -1},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Windows/', 'samples': -1},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Linux/', 'samples': -1},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Windows/', 'samples': -1},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Linux/', 'samples': -1},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': -1},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': -1},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': -1},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': -1},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': -1},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': -1},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': -1},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': -1},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': -1},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': -1},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': -1},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': -1},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': -1},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': -1},        
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': -1},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': -1},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': -1},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': -1},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Windows/', 'samples': -1},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Linux/', 'samples': -1},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Windows/', 'samples': -1},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Linux/', 'samples': -1},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Windows/', 'samples': -1},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Linux/', 'samples': -1},
        }
    
    DATABASE_C_PRESENT={
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Windows/', 'samples': -1},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Linux/', 'samples': -1},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Windows/', 'samples': -1},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Linux/', 'samples': -1},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Windows/', 'samples': -1},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Linux/', 'samples': -1},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Windows/', 'samples': -1},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Linux/', 'samples': -1},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Windows/', 'samples': -1},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Linux/', 'samples': -1},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Windows/', 'samples': -1},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Linux/', 'samples': -1},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Windows/', 'samples': -1},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Linux/', 'samples': -1},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Windows/', 'samples': -1},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Windows/', 'samples': -1},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Linux/', 'samples': -1},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Windows/', 'samples': 0},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Linux/', 'samples': 0},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Windows/', 'samples': 0},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Linux/', 'samples': 0},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Windows/', 'samples': 0},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Linux/', 'samples': 0},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Windows/', 'samples': -1},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Linux/', 'samples': -1},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Windows/', 'samples': -1},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Linux/', 'samples': -1},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': -1},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': -1},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': -1},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': -1},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': -1},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Windows/', 'samples': -1},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Linux/', 'samples': -1},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Windows/', 'samples': -1},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Linux/', 'samples': -1},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': -1},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': -1},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': -1},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': -1},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': -1},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': -1},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': -1},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': -1},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': -1},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': -1},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': -1},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': -1},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': -1},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': -1},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': -1},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': -1},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': -1},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': -1},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': -1},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': -1},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Windows/', 'samples': 0},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Linux/', 'samples': 0},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Windows/', 'samples': 0},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Linux/', 'samples': 0},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Linux/', 'samples': 0},
        }
    
    DATABASE_C_NOTPRESENT = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Windows/', 'samples': 0},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Linux/', 'samples': 0},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Windows/', 'samples': 0},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Linux/', 'samples': 0},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Windows/', 'samples': 0},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Linux/', 'samples': 0},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Windows/', 'samples': 0},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Linux/', 'samples': 0},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Windows/', 'samples': 0},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Linux/', 'samples': 0},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Windows/', 'samples': 0},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Linux/', 'samples': 0},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Windows/', 'samples': 0},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Linux/', 'samples': 0},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Windows/', 'samples': 0},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Windows/', 'samples': 0},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Linux/', 'samples': 0},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Windows/', 'samples': 0},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Linux/', 'samples': -1},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Windows/', 'samples': -1},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Linux/', 'samples': -1},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Windows/', 'samples': 0},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Linux/', 'samples': -1},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 0},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 0},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 0},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 0},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 0},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 0},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 0},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 0},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 0},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 0},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 0},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 0},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 0},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 0},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 0},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 0},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 0},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 0},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 0},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 0},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 0},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 0},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 0},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 0},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 0},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 0},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 0},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 0},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 0},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 0},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 0},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 0},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 0},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Windows/', 'samples': -1},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Linux/', 'samples': -1},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Windows/', 'samples': 0},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Linux/', 'samples': -1},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Linux/', 'samples': -1},
    }
    
    DATABASE_C_RANSOMWARE = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Windows/', 'samples': 569},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Linux/', 'samples': 0},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Windows/', 'samples': 0},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Linux/', 'samples': 0},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Windows/', 'samples': 0},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Linux/', 'samples': 0},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Windows/', 'samples': 0},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Linux/', 'samples': 0},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Windows/', 'samples': 0},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Linux/', 'samples': 0},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Windows/', 'samples': 0},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Linux/', 'samples': 0},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Windows/', 'samples': 0},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Linux/', 'samples': 0},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Windows/', 'samples': 0},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Windows/', 'samples': 0},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Linux/', 'samples': 0},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Windows/', 'samples': 0},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Linux/', 'samples': 0},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Windows/', 'samples': 0},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Linux/', 'samples': 0},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Windows/', 'samples': 0},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Linux/', 'samples': 0},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 0},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 0},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 0},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 0},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 0},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 0},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 0},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 0},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 0},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 0},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 0},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 0},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 0},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 0},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 0},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 0},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 0},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 0},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 0},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 0},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 0},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 0},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 0},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 0},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 0},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 0},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 0},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 0},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 0},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 0},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 0},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 0},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 0},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Windows/', 'samples': 0},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Linux/', 'samples': 0},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Windows/', 'samples': 0},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Linux/', 'samples': 0},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Linux/', 'samples': 0},
        'RANSOMWARE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/8 - Ransomware/Windows/', 'samples': -1},
        'RANSOMWARE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/8 - Ransomware/Linux/', 'samples': -1},
    }
    
    DATABASE_C_RANSOMWAREBRUTEPASSWORD = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Windows/', 'samples': 1138},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Linux/', 'samples': 0},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Windows/', 'samples': 0},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Linux/', 'samples': 0},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Windows/', 'samples': 0},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Linux/', 'samples': 0},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Windows/', 'samples': 0},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Linux/', 'samples': 0},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Windows/', 'samples': 0},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Linux/', 'samples': 0},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Windows/', 'samples': 0},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Linux/', 'samples': 0},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Windows/', 'samples': 0},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Linux/', 'samples': 0},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Windows/', 'samples': 0},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Windows/', 'samples': 0},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Linux/', 'samples': 0},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Windows/', 'samples': 0},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Linux/', 'samples': 0},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Windows/', 'samples': 0},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Linux/', 'samples': 0},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Windows/', 'samples': 0},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Linux/', 'samples': 0},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 0},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 0},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 0},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 0},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 0},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 0},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 0},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 0},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 0},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 0},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 0},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 0},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 0},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 0},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 0},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 0},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 0},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 0},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 0},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 0},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 0},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 0},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 0},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 0},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 0},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 0},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 0},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 0},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 0},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 0},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 0},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 0},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 0},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Windows/', 'samples': 0},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Linux/', 'samples': 0},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Windows/', 'samples': 0},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Linux/', 'samples': 0},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Linux/', 'samples': 569},
        'RANSOMWARE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/8 - Ransomware/Windows/', 'samples': -1},
        'RANSOMWARE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/8 - Ransomware/Linux/', 'samples': -1},
    }
    
    DATABASE_C_BRUTEPASSWORD = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Windows/', 'samples': 569},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/1 - Idle/Linux/', 'samples': 0},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Windows/', 'samples': 0},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Netflix/Linux/', 'samples': 0},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Windows/', 'samples': 0},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/PlutoTV/Linux/', 'samples': 0},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Windows/', 'samples': 0},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/RuntimeTV/Linux/', 'samples': 0},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Windows/', 'samples': 0},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Vimeo/Linux/', 'samples': 0},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Windows/', 'samples': 0},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/2 - Streaming/Youtube/Linux/', 'samples': 0},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Windows/', 'samples': 0},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/Openttd/Linux/', 'samples': 0},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Windows/', 'samples': 0},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Windows/', 'samples': 0},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Backup/Linux/', 'samples': 0},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Windows/', 'samples': 0},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Compile/Linux/', 'samples': 0},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Windows/', 'samples': 0},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/Download/Linux/', 'samples': 0},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Windows/', 'samples': 0},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/4 - Other/SysUpgrade/Linux/', 'samples': 0},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 0},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 0},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 0},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 0},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 0},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 0},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 0},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 0},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 0},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 0},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 0},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 0},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 0},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 0},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 0},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 0},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 0},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 0},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 0},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 0},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 0},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 0},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 0},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 0},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 0},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 0},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 0},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 0},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 0},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 0},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 0},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 0},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 0},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Windows/', 'samples': 0},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginIn/Linux/', 'samples': 0},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Windows/', 'samples': 0},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BruteLoginOut/Linux/', 'samples': 0},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/7 - BruteForce/BrutePassword/Linux/', 'samples': 569},
        'RANSOMWARE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/8 - Ransomware/Windows/', 'samples': 0},
        'RANSOMWARE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_C/8 - Ransomware/Linux/', 'samples': 0},
    }
    
    
    TRAINSET_A={
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Windows/', 'samples': 13300},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Linux/', 'samples': 13300},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Windows/', 'samples': 1400},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Linux/', 'samples': 1400},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Windows/', 'samples': 1400},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Linux/', 'samples': 1400},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Windows/', 'samples': 1400},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Linux/', 'samples': 1400},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Windows/', 'samples': 1400},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Linux/', 'samples': 1400},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Windows/', 'samples': 1400},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Linux/', 'samples': 1400},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Windows/', 'samples':1400},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Linux/', 'samples': 1400},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Windows/', 'samples': 1400},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Linux/', 'samples': 1400},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Windows/', 'samples': 1400},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Linux/', 'samples': 1400},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Windows/', 'samples': 1400},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Linux/', 'samples': 1400},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Windows/', 'samples': 1400},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Linux/', 'samples': 1400},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Windows/', 'samples': 1400},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Linux/', 'samples': 1400},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 1400},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 1400},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 1400},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 1400},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 1400},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 1400},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 1400},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 1400},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 1400},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 1400},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 1400},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 1400},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 1400},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 1400},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 1400},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 1400},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 1400},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 1400},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 1400},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 1400},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 1400},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 1400},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 1400},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 1400},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 1400},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 1400},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 1400},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 1400},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 1400},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 1400},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 1400},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 1400},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 1400},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 1400},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 1400},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 1400},
        }
    TRAINSET_B={
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Windows/', 'samples': 13300},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Linux/', 'samples': 13300},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Windows/', 'samples': 1400},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Linux/', 'samples': 1400},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Windows/', 'samples': 1400},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Linux/', 'samples': 1400},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Windows/', 'samples': 1400},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Linux/', 'samples': 1400},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Windows/', 'samples': 1400},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Linux/', 'samples': 1400},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Windows/', 'samples': 1400},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Linux/', 'samples': 1400},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Windows/', 'samples':1400},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Linux/', 'samples': 1400},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Windows/', 'samples': 1400},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Linux/', 'samples': 1400},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Windows/', 'samples': 1400},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Linux/', 'samples': 1400},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Windows/', 'samples': 1400},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Linux/', 'samples': 1400},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 1400},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 1400},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 1400},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 1400},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 1400},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 1400},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 1400},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 1400},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 1400},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 1400},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 1400},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 1400},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 1400},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 1400},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 1400},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 1400},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 1400},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 1400},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 1400},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 1400},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 1400},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 1400},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 1400},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 1400},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 1400},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 1400},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 1400},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 1400},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 1400},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 1400},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 1400},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 1400},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 1400},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 1400},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 1400},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 1400},
        }
    TRAINSET_A_100K={
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Windows/', 'samples': 13500},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Linux/', 'samples': 13500},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Windows/', 'samples': 1400},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Linux/', 'samples': 1400},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Windows/', 'samples': 1400},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Linux/', 'samples': 1400},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Windows/', 'samples': 1400},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Linux/', 'samples': 1400},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Windows/', 'samples': 1400},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Linux/', 'samples': 1400},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Windows/', 'samples': 1400},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Linux/', 'samples': 1400},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Windows/', 'samples':0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Windows/', 'samples': 1500},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Linux/', 'samples': 1500},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Windows/', 'samples': 3000},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Windows/', 'samples': 1500},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Linux/', 'samples': 1500},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 1042},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 1042},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 1042},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 1042},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 1041},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 1041},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 2084},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 2083},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 2083},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 1042},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 1042},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 1042},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 1042},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 1041},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 1041},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 1042},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 1042},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 1042},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 1042},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 1041},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 1041},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 2084},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 2084},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 2083},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 2083},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 2083},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 2083},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 2084},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 2084},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 2083},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 2083},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 2083},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 2083},
        }
    
    TRAINSET_B_100K = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Windows/', 'samples': 13500},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Linux/', 'samples': 13500},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Windows/', 'samples': 1400},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Linux/', 'samples': 1400},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Windows/', 'samples': 1400},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Linux/', 'samples': 1400},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Windows/', 'samples': 1400},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Linux/', 'samples': 1400},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Windows/', 'samples': 1400},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Linux/', 'samples': 1400},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Windows/', 'samples': 1400},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Linux/', 'samples': 1400},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Windows/', 'samples': 1500},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Linux/', 'samples': 1500},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Windows/', 'samples': 3000},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Windows/', 'samples': 1500},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Linux/', 'samples': 1500},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 1042},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 1042},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 1042},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 1042},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 1041},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 1041},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 2084},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 2083},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 2083},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 1042},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 1042},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 1042},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 1042},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 1041},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 1041},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 1042},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 1042},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 1042},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 1042},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 1041},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 1041},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 2084},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 2084},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 2083},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 2083},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 2083},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 2083},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 2084},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 2084},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 2083},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 2083},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 2083},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 2083},
    }

    TRAINSET_A_50K = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Windows/', 'samples': 6750},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Linux/', 'samples': 6750},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Windows/', 'samples': 700},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Linux/', 'samples': 700},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Windows/', 'samples': 700},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Linux/', 'samples': 700},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Windows/', 'samples': 700},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Linux/', 'samples': 700},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Windows/', 'samples': 700},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Linux/', 'samples': 700},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Windows/', 'samples': 700},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Linux/', 'samples': 700},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Windows/', 'samples': 750},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Linux/', 'samples': 750},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Windows/', 'samples': 1500},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Windows/', 'samples': 750},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Linux/', 'samples': 750},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 521},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 521},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 521},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 521},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 521},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 520},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 1042},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 1042},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 1041},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 521},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 521},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 521},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 521},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 521},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 520},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 521},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 521},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 521},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 521},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 521},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 520},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 1042},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 1042},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 1042},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 1042},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 1041},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 1041},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 1042},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 1042},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 1042},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 1042},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 1041},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 1041},
    }
    
    TRAINSET_A_53K = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Windows/', 'samples': 7500},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Linux/', 'samples': 7500},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Windows/', 'samples': 700},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Linux/', 'samples': 700},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Windows/', 'samples': 700},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Linux/', 'samples': 700},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Windows/', 'samples': 700},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Linux/', 'samples': 700},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Windows/', 'samples': 700},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Linux/', 'samples': 700},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Windows/', 'samples': 700},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Linux/', 'samples': 700},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Windows/', 'samples': 750},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Linux/', 'samples': 750},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Windows/', 'samples': 1500},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Windows/', 'samples': 750},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Linux/', 'samples': 750},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 521},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 521},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 521},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 521},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 521},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 520},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 1042},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 1042},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 1041},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 521},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 521},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 521},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 521},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 521},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 520},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 521},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 521},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 521},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 521},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 521},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 520},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 1042},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 1042},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 1042},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 1042},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 1041},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 1041},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 1042},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 1042},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 1042},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 1042},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 1041},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 1041},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/ADDED_TO_DATABASE/DATABASE_A/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/ADDED_TO_DATABASE/DATABASE_A/7 - BruteForce/BrutePassword/Linux/', 'samples': 1500},
    }


    TRAINSET_B_50K = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Windows/', 'samples': 6750},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Linux/', 'samples': 6750},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Windows/', 'samples': 700},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Linux/', 'samples': 700},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Windows/', 'samples': 700},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Linux/', 'samples': 700},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Windows/', 'samples': 700},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Linux/', 'samples': 700},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Windows/', 'samples': 700},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Linux/', 'samples': 700},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Windows/', 'samples': 700},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Linux/', 'samples': 700},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Windows/', 'samples': 750},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Linux/', 'samples': 750},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Windows/', 'samples': 1500},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Windows/', 'samples': 750},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Linux/', 'samples': 750},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 521},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 521},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 521},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 521},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 521},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 520},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 1042},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 1042},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 1041},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 521},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 521},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 521},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 521},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 521},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 520},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 521},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 521},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 521},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 521},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 521},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 520},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 1042},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 1042},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 1042},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 1042},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 1041},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 1041},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 1042},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 1042},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 1042},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 1042},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 1041},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 1041},
    }

    TRAINSET_B_53K = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Windows/', 'samples': 7500},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Linux/', 'samples': 7500},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Windows/', 'samples': 700},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Linux/', 'samples': 700},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Windows/', 'samples': 700},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Linux/', 'samples': 700},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Windows/', 'samples': 700},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Linux/', 'samples': 700},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Windows/', 'samples': 700},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Linux/', 'samples': 700},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Windows/', 'samples': 700},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Linux/', 'samples': 700},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Windows/', 'samples': 750},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Linux/', 'samples': 750},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Windows/', 'samples': 1500},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Windows/', 'samples': 750},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Linux/', 'samples': 750},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 521},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 521},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 521},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 521},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 521},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 520},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 1042},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 1042},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 1041},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 521},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 521},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 521},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 521},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 521},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 520},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 521},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 521},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 521},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 521},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 521},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 520},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 1042},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 1042},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 1042},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 1042},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 1041},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 1041},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 1042},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 1042},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 1042},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 1042},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 1041},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 1041},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/ADDED_TO_DATABASE/DATABASE_B/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/ADDED_TO_DATABASE/DATABASE_B/7 - BruteForce/BrutePassword/Linux/', 'samples': 1500},
    }
    
    TESTSET_A_PRESENT_245K = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Windows/', 'samples': 28500},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Linux/', 'samples': 28500},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Windows/', 'samples': 3500},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Linux/', 'samples': 2700},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Windows/', 'samples': 4200},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Linux/', 'samples': 2600},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Windows/', 'samples': 4200},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Linux/', 'samples': 2800},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Windows/', 'samples': 4300},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Linux/', 'samples': 2800},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Windows/', 'samples': 9300},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Linux/', 'samples': 7000},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Windows/', 'samples': 12000},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Linux/', 'samples': 5000},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Windows/', 'samples': 7000},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Windows/', 'samples': 4000},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Linux/', 'samples': 4000},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Compile/Windows/', 'samples': 0},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Compile/Linux/', 'samples': 0},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Download/Windows/', 'samples': 0},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Download/Linux/', 'samples': 0},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/SysUpgrade/Windows/', 'samples': 0},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/SysUpgrade/Linux/', 'samples': 0},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 2800},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 2800},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 2800},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 2800},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 2800},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 2800},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 4000},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 4000},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 4000},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 2800},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 2800},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 2800},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 2800},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 2800},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 2800},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 2800},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 2800},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 2800},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 5000},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 2800},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 2800},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 4000},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 4000},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 4000},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 4000},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 4000},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 4000},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 4000},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 4000},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 4000},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 4000},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 4000},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 4000},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginIn/Windows/', 'samples': 0},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginIn/Linux/', 'samples': 0},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginOut/Windows/', 'samples': 0},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginOut/Linux/', 'samples': 0},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BrutePassword/Linux/', 'samples': 0},
    }
    
    TESTSET_A_NOTPRESENT_15K = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Windows/', 'samples': 0},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Linux/', 'samples': 0},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Windows/', 'samples': 0},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Linux/', 'samples': 0},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Windows/', 'samples': 0},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Linux/', 'samples': 0},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Windows/', 'samples': 0},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Linux/', 'samples': 0},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Windows/', 'samples': 0},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Linux/', 'samples': 0},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Windows/', 'samples': 0},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Linux/', 'samples': 0},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Windows/', 'samples': 0},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Linux/', 'samples': 0},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Windows/', 'samples': 0},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Windows/', 'samples': 0},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Linux/', 'samples': 0},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Compile/Windows/', 'samples': 0},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Compile/Linux/', 'samples': 2975},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Download/Windows/', 'samples': 1475},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Download/Linux/', 'samples': 1475},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/SysUpgrade/Windows/', 'samples': 0},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/SysUpgrade/Linux/', 'samples': 200},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 0},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 0},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 0},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 0},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 0},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 0},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 0},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 0},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 0},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 0},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 0},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 0},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 0},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 0},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 0},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 0},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 0},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 0},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 0},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 0},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 0},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 0},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 0},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 0},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 0},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 0},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 0},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 0},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 0},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 0},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 0},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 0},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 0},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginIn/Windows/', 'samples': 1475},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginIn/Linux/', 'samples': 1475},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginOut/Windows/', 'samples': 0},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BruteLoginOut/Linux/', 'samples': 2975},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/7 - BruteForce/BrutePassword/Linux/', 'samples': 2950},
    }
    
    TESTSET_B_PRESENT_245K = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Windows/', 'samples': 28500},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Linux/', 'samples': 28500},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Windows/', 'samples': 3500},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Linux/', 'samples': 2700},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Windows/', 'samples': 4200},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Linux/', 'samples': 2600},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Windows/', 'samples': 4200},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Linux/', 'samples': 2800},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Windows/', 'samples': 4300},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Linux/', 'samples': 2800},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Windows/', 'samples': 9300},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Linux/', 'samples': 7000},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Windows/', 'samples': 1000},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Linux/', 'samples': 1000},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Windows/', 'samples': 10000},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Linux/', 'samples': 4000},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Windows/', 'samples': 1000},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Windows/', 'samples': 7000},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Windows/', 'samples': 4000},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Linux/', 'samples': 4000},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Compile/Windows/', 'samples': 0},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Compile/Linux/', 'samples': 0},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Download/Windows/', 'samples': 0},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Download/Linux/', 'samples': 0},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/SysUpgrade/Windows/', 'samples': 0},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/SysUpgrade/Linux/', 'samples': 0},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 2800},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 2800},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 2800},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 2800},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 2800},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 2800},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 4000},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 4000},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 4000},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 2800},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 2800},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 2800},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 2800},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 2800},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 2800},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 2800},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 2800},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 2800},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 5000},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 2800},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 2800},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 4000},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 4000},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 4000},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 4000},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 4000},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 4000},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 4000},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 4000},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 4000},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 4000},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 4000},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 4000},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginIn/Windows/', 'samples': 0},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginIn/Linux/', 'samples': 0},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginOut/Windows/', 'samples': 0},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginOut/Linux/', 'samples': 0},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BrutePassword/Linux/', 'samples': 0},
    }
    
    TESTSET_B_NOTPRESENT_15K = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Windows/', 'samples': 0},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Linux/', 'samples': 0},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Windows/', 'samples': 0},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Linux/', 'samples': 0},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Windows/', 'samples': 0},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Linux/', 'samples': 0},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Windows/', 'samples': 0},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Linux/', 'samples': 0},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Windows/', 'samples': 0},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Linux/', 'samples': 0},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Windows/', 'samples': 0},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Linux/', 'samples': 0},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Windows/', 'samples': 0},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Linux/', 'samples': 0},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Windows/', 'samples': 0},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Windows/', 'samples': 0},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Linux/', 'samples': 0},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Compile/Windows/', 'samples': 0},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Compile/Linux/', 'samples': 2975},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Download/Windows/', 'samples': 1475},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Download/Linux/', 'samples': 1475},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/SysUpgrade/Windows/', 'samples': 0},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/SysUpgrade/Linux/', 'samples': 200},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 0},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 0},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 0},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 0},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 0},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 0},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 0},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 0},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 0},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 0},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 0},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 0},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 0},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 0},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 0},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 0},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 0},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 0},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 0},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 0},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 0},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 0},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 0},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 0},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 0},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 0},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 0},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 0},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 0},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 0},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 0},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 0},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 0},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginIn/Windows/', 'samples': 1475},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginIn/Linux/', 'samples': 1475},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginOut/Windows/', 'samples': 0},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginOut/Linux/', 'samples': 2975},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BrutePassword/Linux/', 'samples': 2950},
    }
    
    
    TESTSET_B_200K = {
        'IDLE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Windows/', 'samples': 27000},
        'IDLE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Linux/', 'samples': 27000},
        'NETFLIX-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Windows/', 'samples': 2800},
        'NETFLIX-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Linux/', 'samples': 2800},
        'PLUTOTV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Windows/', 'samples': 2800},
        'PLUTOTV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Linux/', 'samples': 2800},
        'RUNTIMETV-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Windows/', 'samples': 2800},
        'RUNTIMETV-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Linux/', 'samples': 2800},
        'VIMEO-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Windows/', 'samples': 2800},
        'VIMEO-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Linux/', 'samples': 2800},
        'YOUTUBE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Windows/', 'samples': 2800},
        'YOUTUBE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Linux/', 'samples': 2800},
        'COUNTERSTRIKE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Windows/', 'samples': -1},
        'COUNTERSTRIKE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Linux/', 'samples': -1},
        'DOTA-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Windows/', 'samples': -1},
        'DOTA-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Linux/', 'samples': -1},
        'OPENTTD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Windows/', 'samples': 3000},
        'OPENTTD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Linux/', 'samples': 3000},
        'SIMS-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Windows/', 'samples': -1},
        'SIMS-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Linux/', 'samples': -1},
        'BOINC-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Windows/', 'samples': 6000},
        'BOINC-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Linux/', 'samples': 0},
        'BACKUP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Windows/', 'samples': 3000},
        'BACKUP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Linux/', 'samples': 3000},
        'COMPILE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Compile/Windows/', 'samples': 0},
        'COMPILE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Compile/Linux/', 'samples': 0},
        'DOWNLOAD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Download/Windows/', 'samples': 0},
        'DOWNLOAD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Download/Linux/', 'samples': 0},
        'SYSUPGRADE-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/SysUpgrade/Windows/', 'samples': 0},
        'SYSUPGRADE-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/SysUpgrade/Linux/', 'samples': 0},
        'HPING3INTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 2084},
        'HPING3INTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 2084},
        'HPING3INUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 2084},
        'HPING3INUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 2084},
        'HPING3INICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 2084},
        'HPING3INICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 2080},
        'HPING3OUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 4168},
        'HPING3OUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 4168},
        'HPING3OUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 4168},
        'NPINGINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 2084},
        'NPINGINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 2084},
        'NPINGINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 2084},
        'NPINGINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 2084},
        'NPINGINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 2084},
        'NPINGINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 2084},
        'NPINGOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 2084},
        'NPINGOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 2084},
        'NPINGOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 2084},
        'NPINGOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 2084},
        'NPINGOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 2084},
        'NPINGOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 2080},
        'NMAPINTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 4168},
        'NMAPINTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 4168},
        'NMAPINUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 4168},
        'NMAPINUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 4168},
        'NMAPINICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 4164},
        'NMAPINICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 4164},
        'NMAPOUTTCP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 4168},
        'NMAPOUTTCP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 4168},
        'NMAPOUTUDP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 4168},
        'NMAPOUTUDP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 4168},
        'NMAPOUTICMP-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 4164},
        'NMAPOUTICMP-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 4164},
        'BRUTELOGININ-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginIn/Windows/', 'samples': 0},
        'BRUTELOGININ-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginIn/Linux/', 'samples': 0},
        'BRUTELOGINOUT-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginOut/Windows/', 'samples': 0},
        'BRUTELOGINOUT-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BruteLoginOut/Linux/', 'samples': 0},
        'BRUTEPASSWORD-WIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BrutePassword/Windows/', 'samples': 0},
        'BRUTEPASSWORD-LIN': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/7 - BruteForce/BrutePassword/Linux/', 'samples': 0},
    }
    
    TRAINSET_AB_50K = {
        'IDLE-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Windows/', 'samples': 3375},
        'IDLE-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/1 - Idle/Linux/', 'samples': 3375},
        'IDLE-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Windows/', 'samples': 3375},
        'IDLE-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/1 - Idle/Linux/', 'samples': 3375},

        'NETFLIX-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Windows/', 'samples': 350},
        'NETFLIX-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Netflix/Linux/', 'samples': 350},
        'NETFLIX-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Windows/', 'samples': 350},
        'NETFLIX-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Netflix/Linux/', 'samples': 350},
        
        'PLUTOTV-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Windows/', 'samples': 350},
        'PLUTOTV-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/PlutoTV/Linux/', 'samples': 350},
        'PLUTOTV-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Windows/', 'samples': 350},
        'PLUTOTV-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/PlutoTV/Linux/', 'samples': 350},

        'RUNTIMETV-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Windows/', 'samples': 350},
        'RUNTIMETV-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/RuntimeTV/Linux/', 'samples': 350},
        'RUNTIMETV-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Windows/', 'samples': 350},
        'RUNTIMETV-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/RuntimeTV/Linux/', 'samples': 350},

        'VIMEO-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Windows/', 'samples': 350},
        'VIMEO-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Vimeo/Linux/', 'samples': 350},
        'VIMEO-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Windows/', 'samples': 350},
        'VIMEO-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Vimeo/Linux/', 'samples': 350},
   
        'YOUTUBE-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Windows/', 'samples': 350},
        'YOUTUBE-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/2 - Streaming/Youtube/Linux/', 'samples': 350},
        'YOUTUBE-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Windows/', 'samples': 350},
        'YOUTUBE-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/2 - Streaming/Youtube/Linux/', 'samples': 350},

        'COUNTERSTRIKE-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/CounterStrike/Linux/', 'samples': 0},
        'COUNTERSTRIKE-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Windows/', 'samples': 0},
        'COUNTERSTRIKE-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/CounterStrike/Linux/', 'samples': 0},

        'DOTA-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/DOTA/Linux/', 'samples': 0},
        'DOTA-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Windows/', 'samples': 0},
        'DOTA-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/DOTA/Linux/', 'samples': 0},

        'OPENTTD-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Windows/', 'samples': 350},
        'OPENTTD-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/Openttd/Linux/', 'samples': 350},
        'OPENTTD-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Windows/', 'samples': 350},
        'OPENTTD-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/Openttd/Linux/', 'samples': 350},
        
        'SIMS-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/3 - Gaming/SIMS/Linux/', 'samples': 0},
        'SIMS-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Windows/', 'samples': 0},
        'SIMS-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/3 - Gaming/SIMS/Linux/', 'samples': 0},

        'BOINC-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Windows/', 'samples': 750},
        'BOINC-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/BOINC/Linux/', 'samples': 0},
        'BOINC-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Windows/', 'samples': 750},
        'BOINC-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/BOINC/Linux/', 'samples': 0},
        
        'BACKUP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Windows/', 'samples': 375},
        'BACKUP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/4 - Other/Backup/Linux/', 'samples': 375},
        'BACKUP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Windows/', 'samples': 375},
        'BACKUP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/4 - Other/Backup/Linux/', 'samples': 375},
        
        'HPING3INTCP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 261},
        'HPING3INTCP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 261},
        'HPING3INTCP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Windows/', 'samples': 260},
        'HPING3INTCP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/TCP/Linux/', 'samples': 260},
        
        'HPING3INUDP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 261},
        'HPING3INUDP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 261},
        'HPING3INUDP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Windows/', 'samples': 260},
        'HPING3INUDP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/UDP/Linux/', 'samples': 260},  
    
        'HPING3INICMP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 261},
        'HPING3INICMP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 260},
        'HPING3INICMP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Windows/', 'samples': 260},
        'HPING3INICMP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3IN/ICMP/Linux/', 'samples': 260},

        'HPING3OUTTCP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 521},
        'HPING3OUTTCP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Windows/', 'samples': 0},
        'HPING3OUTTCP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/TCP/Linux/', 'samples': 521},
        
        'HPING3OUTUDP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 521},
        'HPING3OUTUDP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Windows/', 'samples': 0},
        'HPING3OUTUDP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/UDP/Linux/', 'samples': 521},
        
        'HPING3OUTICMP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 521},        
        'HPING3OUTICMP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Windows/', 'samples': 0},
        'HPING3OUTICMP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/HPING3OUT/ICMP/Linux/', 'samples': 520},

        'NPINGINTCP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 261},
        'NPINGINTCP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 261},
        'NPINGINTCP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Windows/', 'samples': 260},
        'NPINGINTCP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/TCP/Linux/', 'samples': 260},

        'NPINGINUDP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 261},
        'NPINGINUDP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 261},
        'NPINGINUDP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Windows/', 'samples': 260},
        'NPINGINUDP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/UDP/Linux/', 'samples': 260},

        'NPINGINICMP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 261},
        'NPINGINICMP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 260},
        'NPINGINICMP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Windows/', 'samples': 260},
        'NPINGINICMP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGIN/ICMP/Linux/', 'samples': 260},

        'NPINGOUTTCP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 261},
        'NPINGOUTTCP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 261},
        'NPINGOUTTCP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Windows/', 'samples': 260},
        'NPINGOUTTCP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/TCP/Linux/', 'samples': 260},

        'NPINGOUTUDP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 261},
        'NPINGOUTUDP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 261},
        'NPINGOUTUDP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Windows/', 'samples': 260},
        'NPINGOUTUDP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/UDP/Linux/', 'samples': 260},

        'NPINGOUTICMP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 261},
        'NPINGOUTICMP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 260},
        'NPINGOUTICMP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Windows/', 'samples': 260},
        'NPINGOUTICMP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/5 - Dos/NPINGOUT/ICMP/Linux/', 'samples': 260},

        'NMAPINTCP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 521},
        'NMAPINTCP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 521},
        'NMAPINTCP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Windows/', 'samples': 521},
        'NMAPINTCP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/TCP/Linux/', 'samples': 521},
        
        'NMAPINUDP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 521},
        'NMAPINUDP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 521},
        'NMAPINUDP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Windows/', 'samples': 521},
        'NMAPINUDP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/UDP/Linux/', 'samples': 521},
        
        'NMAPINICMP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 521},
        'NMAPINICMP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 521},
        'NMAPINICMP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Windows/', 'samples': 520},
        'NMAPINICMP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPIN/ICMP/Linux/', 'samples': 520},
        
        'NMAPOUTTCP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 521},
        'NMAPOUTTCP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 521},
        'NMAPOUTTCP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Windows/', 'samples': 521},
        'NMAPOUTTCP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/TCP/Linux/', 'samples': 521},
        
        'NMAPOUTUDP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 521},
        'NMAPOUTUDP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 521},
        'NMAPOUTUDP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Windows/', 'samples': 521},
        'NMAPOUTUDP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/UDP/Linux/', 'samples': 521},


        'NMAPOUTICMP-WINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 521},
        'NMAPOUTICMP-LINA': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_A/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 521},
        'NMAPOUTICMP-WINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Windows/', 'samples': 520},
        'NMAPOUTICMP-LINB': {'path': '/home/grau/TFG/Definitiu/2 - Bases de dades/DATABASE_B/6 - PortScanning/NMAPOUT/ICMP/Linux/', 'samples': 520},
    }

    

    
    if known_data_to_be_loaded=='TEST':
        known_data_definition=TEST_A_SELECTED
        validation_data_definition=TEST_A_FULL
        
    elif known_data_to_be_loaded=='TEST1':
        known_data_definition=TEST_A_SELECTED
        validation_data_definition=TEST_1_FULL
        
    elif known_data_to_be_loaded=='TEST2':
        known_data_definition=TEST_A_SELECTED
        validation_data_definition=TEST_2_FULL
        
    elif known_data_to_be_loaded=='DATABASE_A':
        known_data_definition=DATABASE_A
        validation_data_definition=DATABASE_A

    elif known_data_to_be_loaded=='DATABASE_B':
        known_data_definition=DATABASE_B
        validation_data_definition=DATABASE_B
        
    elif known_data_to_be_loaded=='TRAINSET_A':
        known_data_definition=TRAINSET_A
        validation_data_definition=DATABASE_A

    elif known_data_to_be_loaded=='TRAINSET_B':
        known_data_definition=TRAINSET_B
        validation_data_definition=DATABASE_B
         
    elif known_data_to_be_loaded=='TRAINSET_A_100K':
        known_data_definition=TRAINSET_A_100K
        validation_data_definition=DATABASE_A

    elif known_data_to_be_loaded=='TRAINSET_B_100K':
        known_data_definition=TRAINSET_B_100K
        validation_data_definition=DATABASE_B
        
    elif known_data_to_be_loaded=='TRAINSET_A_50K':
        known_data_definition=TRAINSET_A_50K
        validation_data_definition=DATABASE_A

    elif known_data_to_be_loaded=='TRAINSET_B_50K':
        known_data_definition=TRAINSET_B_50K
        validation_data_definition=DATABASE_B
        
    elif known_data_to_be_loaded=='TRAINSET_AB_50K':
        known_data_definition=TRAINSET_AB_50K
        validation_data_definition=DATABASE_B
        
    elif known_data_to_be_loaded=='TRAINSET_A_53K':
        known_data_definition=TRAINSET_A_53K
        validation_data_definition=DATABASE_A

    elif known_data_to_be_loaded=='TRAINSET_B_53K':
        known_data_definition=TRAINSET_B_53K
        validation_data_definition=DATABASE_B
 
    else: 
        print(f'Unexpected known_data_to_be_loaded value: {known_data_to_be_loaded}')
        
        
    if validation_data_to_be_loaded=='TEST':
        validation_data_definition=TEST_A_FULL
        
    if validation_data_to_be_loaded=='TEST1':
        validation_data_definition=TEST_1_FULL

    if validation_data_to_be_loaded=='TEST2':
        validation_data_definition=TEST_2_FULL
                
    elif validation_data_to_be_loaded=='DATABASE_A':
        validation_data_definition=DATABASE_A

    elif validation_data_to_be_loaded=='DATABASE_B':
        validation_data_definition=DATABASE_B

    elif validation_data_to_be_loaded=='TESTSET_B_PRESENT_245K':
        validation_data_definition=TESTSET_B_PRESENT_245K

    elif validation_data_to_be_loaded=='TESTSET_A_PRESENT_245K':
        validation_data_definition=TESTSET_A_PRESENT_245K
    
    elif validation_data_to_be_loaded=='TESTSET_B_NOTPRESENT_15K':
        validation_data_definition=TESTSET_B_NOTPRESENT_15K

    elif validation_data_to_be_loaded=='TESTSET_A_NOTPRESENT_15K':
        validation_data_definition=TESTSET_A_NOTPRESENT_15K
        
    known_data_definition = {name: contents for name, contents in known_data_definition.items() if contents['samples'] != 0}
   
    # Get known data files information filtered by platform
    
    known_data_files_info=searchfilesinfolders(known_data_definition,extension_to_search,data_platform_to_be_used)
    
    validation_data_files_info=searchfilesinfolders(validation_data_definition,extension_to_search,data_platform_to_be_used)

    # Define folders and extension for unknown data
    
    if unknown_data_to_be_loaded=='TEST':
        unknown_data_definition=TEST_B_FULL

    elif unknown_data_to_be_loaded=='TEST1':
            unknown_data_definition=TEST_1_FULL
    
    elif unknown_data_to_be_loaded=='TEST2':
            unknown_data_definition=TEST_2_FULL

    
    elif unknown_data_to_be_loaded=='DATABASE_A':
            unknown_data_definition=DATABASE_A
    
    elif unknown_data_to_be_loaded=='DATABASE_B':
        unknown_data_definition=DATABASE_B
    
    elif unknown_data_to_be_loaded=='DATABASE_C':
        unknown_data_definition=DATABASE_C
        
    elif unknown_data_to_be_loaded=='DATABASE_C_PRESENT':
        unknown_data_definition=DATABASE_C_PRESENT
    
    elif unknown_data_to_be_loaded=='DATABASE_C_NOTPRESENT':
        unknown_data_definition=DATABASE_C_NOTPRESENT
    
    elif unknown_data_to_be_loaded=='DATABASE_C_RANSOMWARE':
        unknown_data_definition=DATABASE_C_RANSOMWARE
        
    elif unknown_data_to_be_loaded=='DATABASE_C_BRUTEPASSWORD':
        unknown_data_definition=DATABASE_C_BRUTEPASSWORD
    
    elif unknown_data_to_be_loaded=='DATABASE_C_RANSOMWAREBRUTEPASSWORD':
        unknown_data_definition=DATABASE_C_RANSOMWAREBRUTEPASSWORD
        
    elif unknown_data_to_be_loaded=='TESTSET_A_PRESENT_245K':
            unknown_data_definition=TESTSET_A_PRESENT_245K
    
    elif unknown_data_to_be_loaded=='TESTSET_B_PRESENT_245K':
            unknown_data_definition=TESTSET_B_PRESENT_245K
    
    elif unknown_data_to_be_loaded=='TESTSET_A_NOTPRESENT_15K':
            unknown_data_definition=TESTSET_A_NOTPRESENT_15K
    
    elif unknown_data_to_be_loaded=='TESTSET_B_NOTPRESENT_15K':
            unknown_data_definition=TESTSET_B_NOTPRESENT_15K
    
    else: 
        print(f'Unexpected unknown_data_to_be_loaded value: {known_data_to_be_loaded}')
    
   
    unknown_data_definition = {name: contents for name, contents in unknown_data_definition.items() if contents['samples'] != 0}
    
    # Get known data files information filtered by platform
    unknown_data_files_info=searchfilesinfolders(unknown_data_definition,extension_to_search,data_platform_to_be_used)
    
    # validation_data_definition=getvalidationdatadefinition(known_data_definition)
    
    return (known_data_files_info,unknown_data_files_info,validation_data_files_info,known_data_definition,unknown_data_definition,validation_data_definition)

def getalldata(data_files_info,known_data_definition,data_platform_to_be_used,parameters):
    all_data=pd.DataFrame()
    
    for file_info in data_files_info:
        temp=importdatafrom7zfile(file_info['global_path'],parameters)

        
        temp['filter_field'] = file_info['filter_field']
        all_data = pd.concat([all_data, temp])
    return (all_data)

def getdataset(all_data,data_definition,parameters,data_type='Train'):
    
    # Data temporality 
    all_data,parameters=addtemporality(all_data,data_type,parameters)
  
    # Data validation and exclusion 
    
    all_data=validatedata3(all_data,data_type,parameters)
    
    # Select defined number of samples
    selected_data = pd.DataFrame()

       
    for name, contents in data_definition.items():
        temporality_index = parameters['General']['temporality_index']
        if temporality_index == 1:
            temp = all_data[all_data['filter_field'] == name]
        elif temporality_index in range(2, 21):
            temp = all_data[all_data['filter_field_0'] == name]
        else:
            raise ValueError("Unexpected value for 'temporality_index'. Expected between 1 and 20")
        dataframe_size = len(temp)
        sample_size = min(dataframe_size, contents['samples']) if contents['samples'] != -1 else dataframe_size
        
        needed = contents['samples']
        more_samples = 2 * needed - dataframe_size
        if needed > dataframe_size:
            print(f'There are not enough samples in {name}-{contents}')
        print(f'In {name} - Needed: {needed} - Contents: {dataframe_size} - Add {more_samples} to ensure variability')
            
        
        if sample_size < dataframe_size:
            temp = temp.sample(n=sample_size)
            
        selected_data = pd.concat([selected_data, temp])
        selected_data.reset_index(drop=True, inplace=True)

        
    data_columns_to_process=parameters['General']['data_columns_to_process']
    final_data=selected_data[data_columns_to_process]
    if (parameters['General']['temporality_index']==1):
        final_class = pd.DataFrame()
        final_class['data_class']=selected_data['data_class']
        final_metadata=selected_data.drop(columns=data_columns_to_process + ['data_class',])
        
    elif (parameters['General']['temporality_index'] in range(2, 21)):
        final_class = pd.DataFrame()
        final_class['data_class']=selected_data['data_class_0']
        final_metadata=selected_data.drop(columns=data_columns_to_process + ['data_class_0',])
        final_metadata = selected_data.rename(columns={'data_class_0': 'data_class'})
    else: 
        raise ValueError("Unexpected value for 'temporality_index'. Expected between 1 and 20")

    return (final_data,final_class,final_metadata)
    

    
def getdata(data_files_info,known_data_definition,data_platform_to_be_used,parameters,data_type='Known'):
    all_data=pd.DataFrame()
    
    for file_info in data_files_info:
        temp=importdatafrom7zfile(file_info['global_path'],parameters)

        
        temp['filter_field'] = file_info['filter_field']
        all_data = pd.concat([all_data, temp])
        
        
    # Data temporality 
    all_data,parameters=addtemporality(all_data,data_type,parameters)
  
    # Data validation and exclusion 
    
    all_data=validatedata3(all_data,data_type,parameters)
    
    # Select defined number of samples
    selected_data = pd.DataFrame()

       
    for name, contents in known_data_definition.items():
        temporality_index = parameters['General']['temporality_index']
        if temporality_index == 1:
            temp = all_data[all_data['filter_field'] == name]
        elif temporality_index in range(2, 21):
            temp = all_data[all_data['filter_field_0'] == name]
        else:
            raise ValueError("Unexpected value for 'temporality_index'. Expected between 1 and 20")
        dataframe_size = len(temp)
        sample_size = min(dataframe_size, contents['samples']) if contents['samples'] != -1 else dataframe_size
        if sample_size < dataframe_size:
            temp = temp.sample(n=sample_size)
        selected_data = pd.concat([selected_data, temp])
        selected_data.reset_index(drop=True, inplace=True)

        
    data_columns_to_process=parameters['General']['data_columns_to_process']
    final_data=selected_data[data_columns_to_process]
    if (parameters['General']['temporality_index']==1):
        final_class = pd.DataFrame()
        final_class['data_class']=selected_data['data_class']
        final_metadata=selected_data.drop(columns=data_columns_to_process + ['data_class',])
        
    elif (parameters['General']['temporality_index'] in range(2, 21)):
        final_class = pd.DataFrame()
        final_class['data_class']=selected_data['data_class_0']
        final_metadata=selected_data.drop(columns=data_columns_to_process + ['data_class_0',])
        final_metadata = selected_data.rename(columns={'data_class_0': 'data_class'})
    else: 
        raise ValueError("Unexpected value for 'temporality_index'. Expected between 1 and 20")
          

    # Temporal validation data output   

    return (final_data,final_class,final_metadata)

def newcolumns(input_columns , n):
    if n == 1:
        return input_columns
    else:
        output_columns = []
        for i in range(n):
            suffix = f"_{i}"
            output_columns.extend([column + suffix for column in input_columns])
        return output_columns



def addtemporality(input_dataframe,data_type,parameters):
    
    if (parameters['General']['temporality_index'] in range(2, 21)):
        n=parameters['General']['temporality_index']
        num_rows_original, num_cols_original = input_dataframe.shape
        output_dataframe = pd.DataFrame()
        for i in range(n):
            padded_columns = []
            for col in input_dataframe.columns:
                new_column_name = f"{col}_{i}"
                remove_beginning = pd.DataFrame([["Remove"]] * i, columns=[new_column_name])
                remove_end = pd.DataFrame([["Remove"]] * (n - i - 1), columns=[new_column_name])
                col_data = input_dataframe[[col]].rename(columns={col: new_column_name}).reset_index(drop=True)
                padded_col = pd.concat([remove_beginning, col_data, remove_end], ignore_index=True)
                padded_columns.append(padded_col)
            df_concat = pd.concat(padded_columns, axis=1)
            output_dataframe = pd.concat([output_dataframe, df_concat], axis=1)
        output_dataframe = output_dataframe[~output_dataframe.apply(lambda row: row.astype(str).str.contains("Remove")).any(axis=1)].reset_index(drop=True)
        parameters['General']['data_columns_to_process']=newcolumns(parameters['General']['columns_to_get'],parameters['General']['temporality_index'])
        parameters['General']['metadata_columns_to_process']=newcolumns(parameters['General']['metadata_to_get'],parameters['General']['temporality_index'])
        
    elif (parameters['General']['temporality_index']==1):
        output_dataframe=input_dataframe.copy()
        parameters['General']['data_columns_to_process']=newcolumns(parameters['General']['columns_to_get'],parameters['General']['temporality_index'])
        parameters['General']['metadata_columns_to_process']=newcolumns(parameters['General']['metadata_to_get'],parameters['General']['temporality_index'])
    else:
        raise ValueError("Unexpected value for 'temporality_index'. Expected between 1 and 20")
    return output_dataframe,parameters


def generatenewcolumns(input_columns, n):
    if n == 0 or n == 1:
        return input_columns
    elif n >= 2:
        output_columns = []
        for i in range(n):
            suffix = f'_{i}'
            output_columns.extend([col + suffix for col in input_columns])
        return output_columns
    else:
        raise ValueError('Unexpected value of n')

        
def validatecolumns(all_data, columns_list, validation_column, target_column, tolerance=0.0):
    all_data[validation_column] = abs(all_data[columns_list].sum(axis=1) - all_data[target_column]) <= tolerance
    return all_data

def timevalidation(all_data, n):
    for i in range(n):
        col_name = f'iteration_time_validation_{i}' if n > 1 else 'iteration_time_validation'
        iteration_col = f'iteration_time_{i}' if n > 1 else 'iteration_time'
        all_data[col_name] = (all_data[iteration_col] >= 4.5) & (all_data[iteration_col] <= 5.5)
    return all_data

def processesvalidation(all_data, n):
    for i in range(n):
        validation_col = f'processes_validation_{i}' if n > 1 else 'processes_validation'
        kernel_col = f'kernel_processes_{i}' if n > 1 else 'kernel_processes'
        nonkernel_col = f'nonkernel_processes_{i}' if n > 1 else 'nonkernel_processes'
        total_col = f'total_processes_{i}' if n > 1 else 'total_processes'
        all_data[validation_col] = all_data[[kernel_col, nonkernel_col]].sum(axis=1) == all_data[total_col]
    return all_data

def cpuvalidation(all_data, n):
    for i in range(n):
        validation_col = f'cpu_validation_{i}' if n > 1 else 'cpu_validation'
        value_col = f'cpu_validation_value_{i}' if n > 1 else 'cpu_validation_value'
        columns = [f'{col}_{i}' if n > 1 else col for col in [
            'user_cpu', 'nice_cpu', 'system_cpu', 'idle_cpu', 'iowait_cpu',
            'irq_cpu', 'softirq_cpu', 'steal_cpu', 'guest_cpu', 'guest_nice_cpu',
            'interrupt_cpu', 'dpc_cpu'
        ]]
        all_data[value_col] = abs(all_data[columns].sum(axis=1) - 100) / 100
        all_data[validation_col] = all_data[value_col] <= 0.0025
    return all_data

def swapvalidation(all_data, n):
    for i in range(n):
        validation_col = f'swap_validation_{i}' if n > 1 else 'swap_validation'
        columns = [f'{col}_{i}' if n > 1 else col for col in [
            'swap_used_mem', 'swap_free_mem'
        ]]
        total_col = f'swap_total_mem_{i}' if n > 1 else 'swap_total_mem'
        all_data[validation_col] = abs(all_data[columns].sum(axis=1) - all_data[total_col]) / all_data[f'total_mem_{i}' if n > 1 else 'total_mem'] <= 0.001
    return all_data

def protocolfamilyvalidation(all_data, n):
    for i in range(n):
        validation_col = f'connections_protocol_family_validation_{i}' if n > 1 else 'connections_protocol_family_validation'
        columns = [f'{col}_{i}' if n > 1 else col for col in [
            'current_protocol_family_AF_INET', 'current_protocol_family_AF_INET6',
            'current_protocol_family_AF_UNIX', 'current_protocol_family_OTHER'
        ]]
        connections_col = f'current_connections_{i}' if n > 1 else 'current_connections'
        all_data[validation_col] = all_data[columns].sum(axis=1) == all_data[connections_col]
    return all_data

def protocoltypevalidation(all_data, n):
    for i in range(n):
        validation_col = f'connections_protocol_type_validation_{i}' if n > 1 else 'connections_protocol_type_validation'
        columns = [f'{col}_{i}' if n > 1 else col for col in [
            'current_protocol_type_SOCK_STREAM', 'current_protocol_type_SOCK_DGRAM',
            'current_protocol_type_SOCK_SEQPACKET', 'current_protocol_type_OTHER'
        ]]
        connections_col = f'current_connections_{i}' if n > 1 else 'current_connections'
        all_data[validation_col] = all_data[columns].sum(axis=1) == all_data[connections_col]
    return all_data

def countryvalidation(all_data, n):
    for i in range(n):
        validation_col = f'connections_country_validation_{i}' if n > 1 else 'connections_country_validation'
        columns = [f'{col}_{i}' if n > 1 else col for col in [
            'current_country_CN', 'current_country_RU', 'current_country_EMPTY_IP',
            'current_country_OTHER'
        ]]
        connections_col = f'current_connections_{i}' if n > 1 else 'current_connections'
        all_data[validation_col] = all_data[columns].sum(axis=1) == all_data[connections_col]
    return all_data

def continentvalidation(all_data, n):
    for i in range(n):
        validation_col = f'connections_continent_validation_{i}' if n > 1 else 'connections_continent_validation'
        columns = [f'{col}_{i}' if n > 1 else col for col in [
            'current_continent_AF', 'current_continent_AN', 'current_continent_AS',
            'current_continent_EU', 'current_continent_NA', 'current_continent_OC',
            'current_continent_SA', 'current_continent_EMPTY_IP', 'current_continent_OTHER'
        ]]
        connections_col = f'current_connections_{i}' if n > 1 else 'current_connections'
        all_data[validation_col] = all_data[columns].sum(axis=1) == all_data[connections_col]
    return all_data

def statusvalidation(all_data, n):
    for i in range(n):
        validation_col = f'connections_status_validation_{i}' if n > 1 else 'connections_status_validation'
        columns = [f'{col}_{i}' if n > 1 else col for col in [
            'current_status_CLOSE_WAIT', 'current_status_CLOSED', 'current_status_ESTABLISHED',
            'current_status_FIN_WAIT1', 'current_status_FIN_WAIT2', 'current_status_LISTEN',
            'current_status_NONE', 'current_status_SYN_SENT', 'current_status_TIME_WAIT',
            'current_status_LAST_ACK', 'current_status_CLOSING', 'current_status_SYN_RECV', 'current_status_OTHER'
        ]]
        connections_col = f'current_connections_{i}' if n > 1 else 'current_connections'
        all_data[validation_col] = all_data[columns].sum(axis=1) == all_data[connections_col]
    return all_data

def unexpectedothervaluesvalidation(all_data, n):
    for i in range(n):
        validation_col = f'unexpected_other_values_{i}' if n > 1 else 'unexpected_other_values'
        columns = [f'{col}_{i}' if n > 1 else col for col in [
            'current_protocol_type_OTHER', 'current_protocol_family_OTHER', 'current_status_OTHER'
        ]]
        all_data[validation_col] = all_data[columns].sum(axis=1) == 0
    return all_data



def performallvalidations(all_data, n):
    all_data = timevalidation(all_data, n)
    all_data = processesvalidation(all_data, n)
    all_data = cpuvalidation(all_data, n)
    all_data = swapvalidation(all_data, n)
    all_data = protocolfamilyvalidation(all_data, n)
    all_data = protocoltypevalidation(all_data, n)
    all_data = countryvalidation(all_data, n)
    all_data = continentvalidation(all_data, n)
    all_data = statusvalidation(all_data, n)
    all_data = unexpectedothervaluesvalidation(all_data, n)
    
    all_ok_df = pd.DataFrame()
    
    if n == 1:
        validation_columns = [
            'iteration_time_validation',
            'processes_validation',
            'cpu_validation',
            'swap_validation',
            'connections_protocol_family_validation',
            'connections_protocol_type_validation',
            'connections_country_validation',
            'connections_continent_validation',
            'connections_status_validation',
            'unexpected_other_values',
        ]
        

        existing_columns = [col for col in validation_columns if col in all_data.columns]
        if not existing_columns:
            print("Warning: No validation columns found in 'all_data'.")
        else:

            all_data['all_ok'] = all_data[existing_columns].all(axis=1)
    else:
        for i in range(n):

            validation_columns_for_i = [
                f'iteration_time_validation_{i}',
                f'processes_validation_{i}',
                f'cpu_validation_{i}',
                f'swap_validation_{i}',
                f'connections_protocol_family_validation_{i}',
                f'connections_protocol_type_validation_{i}',
                f'connections_country_validation_{i}',
                f'connections_continent_validation_{i}',
                f'connections_status_validation_{i}',
                f'unexpected_other_values_{i}',
            ]

            existing_columns = [col for col in validation_columns_for_i if col in all_data.columns]
            if not existing_columns:
                print(f"Warning: No validation columns found for i={i} in 'all_data'.")
                continue

            all_ok_df[f'all_ok_{i}'] = all_data[existing_columns].all(axis=1)
        

        all_data = pd.concat([all_data, all_ok_df], axis=1)

    if 'all_ok_0' in all_data.columns:
        all_data = all_data.rename(columns={'all_ok_0': 'all_ok'})
    
    return all_data


def printerrorsummary(all_data, n, num_registers, data_type):
    validation_fields = [
        'connections_continent_validation',
        'connections_country_validation',
        'connections_protocol_family_validation',
        'connections_protocol_type_validation',
        'connections_status_validation',
        'cpu_validation',
        'iteration_time_validation',
        'processes_validation',
        'swap_validation',
        'unexpected_other_values'
    ]
    
    for field in validation_fields:
        combined_field = f"{field}_combined"
        all_data[combined_field] = True
        for i in range(n):
            suffix = f"_{i}" if i > 0 else ""
            col_name = f"{field}{suffix}"
            if col_name in all_data.columns:
                all_data[combined_field] &= all_data[col_name]

    print(f'The data defined as {data_type} have {num_registers} records')
    print('Of these:')
    for field in validation_fields:
        combined_field = f"{field}_combined"
        incidences = (~all_data[combined_field]).sum()
        field_name = field.replace("connections_", "").replace("_validation", "").replace("_", " ")
        print(f'{incidences} records have an issue in the field {field_name}')


def validatedata3(all_data, data_type, parameters):
    all_data_to_process = all_data.copy()
    original_columns = all_data.columns
    num_registers = len(all_data_to_process)

    n = parameters['General']['temporality_index']
    all_data_to_process = performallvalidations(all_data_to_process, n)
    
    all_data_error_df = all_data_to_process[~all_data_to_process.filter(like='all_ok').all(axis=1)]
 
    output_file_name = parameters['General']['results_files_path'] + parameters['General']['output_file_name_format'] + '-DataErrors-' + data_type + '.xlsx'
    all_data_error_df.to_excel(output_file_name, index=False)

    compressed_file_name = parameters['General']['compressed_file_name']  
    addtocompressedfile(compressed_file_name, output_file_name)

    printerrorsummary(all_data_to_process, n, num_registers, data_type)

    all_data_correct_df = all_data_to_process[all_data_to_process.filter(like='all_ok').all(axis=1)]
    all_output_data = all_data_correct_df.loc[:, original_columns]
    
    parameters['Columns'] = original_columns

    return all_output_data