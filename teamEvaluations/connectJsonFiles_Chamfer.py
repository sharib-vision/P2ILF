#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 21:23:03 2023

@author: scssali
"""

import json
import csv
import os
import pandas as pd

teamNames = ['output_BHL', 'output_NCT', 'output_UCL', 'output_VIP','output_VOR','output_grasp']

dirfolder = '/Users/scssali/Dataset/p2ilf_final_evaluation_ALI/docker_evaluation/evaluationCodes/'

for k in range(0,6):
    team = teamNames[k]
    dirfolderTeam = os.path.join(dirfolder, team)
    
    # # List of input JSON files
    # json_files = ['eval_test_patient4_3.json', 'eval_test_patient4_4.json', 'eval_test_patient4_7.json', 'eval_test_patient4_11.json',
    #               'eval_test_patient4_17.json', 'eval_test_patient4_20.json', 'eval_test_patient4_21.json', 'eval_test_patient4_22.json',
    #               'eval_test_patient11_2.json','eval_test_patient11_3.json', 'eval_test_patient11_4.json', 'eval_test_patient11_5.json',
    #               'eval_test_patient11_6.json', 'eval_test_patient11_7.json', 'eval_test_patient11_8.json', 'eval_test_patient11_9.json']
    
    
    json_files = ['eval_test_v4_ChamferDist_patient4_3.json', 'eval_test_v4_ChamferDist_patient4_4.json', 'eval_test_v4_ChamferDist_patient4_7.json', 'eval_test_v4_ChamferDist_patient4_11.json',
                  'eval_test_v4_ChamferDist_patient4_17.json', 'eval_test_v4_ChamferDist_patient4_20.json', 'eval_test_v4_ChamferDist_patient4_21.json', 'eval_test_v4_ChamferDist_patient4_22.json',
                  'eval_test_v4_ChamferDist_patient11_2.json','eval_test_v4_ChamferDist_patient11_3.json', 'eval_test_v4_ChamferDist_patient11_4.json', 'eval_test_v4_ChamferDist_patient11_5.json',
                  'eval_test_v4_ChamferDist_patient11_6.json', 'eval_test_v4_ChamferDist_patient11_7.json', 'eval_test_v4_ChamferDist_patient11_8.json', 'eval_test_v4_ChamferDist_patient11_9.json']
    
    
    # json_files = ['eval_test_DSC_patient4_3.json', 'eval_test_DSC_patient4_4.json', 'eval_test_DSC_patient4_7.json', 'eval_test_DSC_patient4_11.json',
    #               'eval_test_DSC_patient4_17.json', 'eval_test_DSC_patient4_20.json', 'eval_test_DSC_patient4_21.json', 'eval_test_DSC_patient4_22.json',
    #               'eval_test_DSC_patient11_2.json','eval_test_DSC_patient11_3.json', 'eval_test_DSC_patient11_4.json', 'eval_test_DSC_patient11_5.json',
    #               'eval_test_DSC_patient11_6.json', 'eval_test_DSC_patient11_7.json', 'eval_test_DSC_patient11_8.json', 'eval_test_DSC_patient11_9.json']
    
    
    # List to store all headers
    all_headers = []
    
    
    os.chdir(dirfolderTeam)
    
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            
            data['dist_r_l_SL'] = [ round(elem, 2) for elem in data['dist_r_l_SL'] ]
            data['dist_HFD_RL'] = [ round(elem, 2) for elem in data['dist_HFD_RL'] ]
            for item in data:
                headers = list(data.keys())
                for header in headers:
                    if header not in all_headers:
                        all_headers.append(header)
                        
                        
    # Loop through each JSON file and extract headers
    # for file in json_files:
    #     with open(file, 'r') as f:
    #         data = json.load(f)
    #         # headers = list(data[0].keys())
    #         # all_headers.extend(headers)
    
    # # Remove duplicates from all_headers list
    # all_headers = list(set(all_headers))
    # all_headers = list(set(data.keys()))
    
    # header2D = list(data.values().mapping.get('P2ILF').get('2D_contour').keys())
    # header3D = list(data.values().mapping.get('P2ILF').get('3D_contour').keys())
    # headerReg = list(data.values().mapping.get('P2ILF').get('P2ILF_task2').keys())
    
    # all_headers = list(data.mapping.get('P2ILF').get('2D_contour').keys(), data.mapping.get('P2ILF').get('3D_contour').keys(),
                       
    # all_headers = header2D + header3D + headerReg
    
    # Write headers to output CSV file 
    
   
    with open(os.path.join(dirfolder, team + '_output_v4_chamfer.csv'), 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(all_headers)
        R_chamfer = []
        L_chamfer = []

        # Loop through each JSON file and write data to CSV file
        for file in json_files:
            with open(file, 'r') as f1:
                data = json.load(f1)
                data['dist_r_l_SL'] = [ round(elem, 2) for elem in data['dist_r_l_SL'] ]
                data['dist_HFD_RL'] = [ round(elem, 2) for elem in data['dist_HFD_RL'] ]
                R_chamfer.append(data['dist_HFD_RL'][0])
                L_chamfer.append(data['dist_HFD_RL'][1])
                for item in data:
                    print(item)
                    row = []
                    for header in all_headers:
                        print(header)
                        # if header in item:
                        row.append(data[header])
                        
                        # else:
                            # row.append('')
                writer.writerow(row)
        df = pd.DataFrame({'Chamfer_Ridge':R_chamfer, 'Chamfer_Ligament':L_chamfer})
        
        df.to_csv(os.path.join(dirfolder, team + '_output_DSC_RL.csv'))  
        # df.to_csv(os.path.join(dirfolder, team + '_output_v4_chamfer_Chamfer_RL.csv'))  
        # print(R_chamfer)
        # print(L_chamfer)
    print('Data has been written to output.csv file.')
    
    
    # A_VOR = [[0.65, 0.76, 0.13],
    # [0.43, 0.7, 0.15],
    # [0.67, 0.66, 0.14],
    # [0.59, 0.52, 0.14],
    # [0.68, 1.0, 0.14],
    # [0.54, 0.23, 0.22],
    # [0.39, 0.0, 0.27],
    # [0.29, 0.0, 0.46],
    # [0.76, 0.21, 0.61],
    # [0.61, 0.19, 0.57],
    # [0.85, 0.19, 0.62],
    # [0.79, 0.22, 0.62],
    # [1.0, 0.25, 0.57],
    # [1.05, 0.34, 0.49],
    # [0.76, 0.39, 0.66],
    # [1.0, 0.21, 0.59]]
    
    # A_VIP = [[0.77, 0.37, 0.22],
    # [0.57, 0.35, 0.26],
    # [0.74, 0.56, 0.27],
    # [0.58, 0.39, 0.28],
    # [0.44, 1.0, 0.37],
    # [1.0,1.0, 0.32],
    # [1.0, 1.0, 0.28],
    # [0.28, 1.0, 0.56],
    # [0.59, 0.19, 0.67],
    # [0.58, 0.12, 0.61],
    # [0.68, 0.21, 0.6],
    # [0.7, 0.26, 0.59],
    # [0.81, 0.29, 0.67],
    # [0.84, 0.34, 0.5],
    # [0.65, 0.32, 0.65],
    # [1.0, 0.23, 0.65]]
    
    
    # A_UCL = [[0.61, 0.38, 0.19],
    # [0.62, 0.43, 0.18],
    # [0.62, 0.47, 0.16],
    # [0.72, 0.3, 0.15],
    # [1.0, 0.24, 0.35],
    # [0.61, 0.23, 0.25],
    # [0.81, 1.0, 0.28],
    # [1.0, 1.0, 0.44],
    # [0.53, 0.18, 0.39],
    # [0.46, 0.2, 0.26],
    # [0.6, 0.12, 0.37],
    # [0.55, 0.1, 0.34],
    # [0.61, 0.1, 0.34],
    # [0.79, 0.12, 0.24],
    # [0.81, 0.11, 0.4],
    # [1.0, 0.11, 0.4]]
    
    
    # A_NCT = [[0.33, 0.9, 0.1],
    # [0.65, 1.0, 0.1],
    # [0.69, 0.85, 0.11],
    # [0.7, 1.0, 0.24],
    # [0.4, 1.0, 0.12],
    # [0.58, 0.52, 0.2],
    # [0.34, 0.0, 0.32],
    # [0.4, 0.0, 0.7],
    # [0.43, 0.22, 0.23],
    # [0.37, 0.11, 0.15],
    # [0.44, 0.24, 0.21],
    # [0.53, 0.27, 0.22],
    # [0.54, 0.41, 0.24],
    # [0.56, 0.44, 0.11],
    # [0.63, 0.21, 0.29],
    # [0.46, 0.16, 0.3]]
    
    # A_BHL = [[0.67, 0.5, 0.18],
    # [0.56, 0.6, 0.18],
    # [0.77, 0.49, 0.12],
    # [0.76, 0.43, 0.39],
    # [1.0, 1.0, 0.38],
    # [0.48, 0.23, 0.29],
    # [0.36, 1.0, 0.35],
    # [1.0, 1.0, 0.59],
    # [0.5, 0.14, 0.45],
    # [0.63, 0.22, 0.37],
    # [0.63, 0.13, 0.37],
    # [0.59, 0.28, 0.38],
    # [0.79, 0.3, 0.36],
    # [0.65, 0.58, 0.31],
    # [0.81, 0.57, 0.45],
    # [0.82, 0.57, 0.44]]
    
# np.nanmean(A, axis=0)