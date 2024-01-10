#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 21:23:03 2023

@author: scssali
"""

import json
import csv
import os

teamNames = ['output_BHL', 'output_NCT', 'output_UCL', 'output_VIP','output_VOR','output_grasp']

dirfolder = '/Users/scssali/Dataset/center-unseen/evaluationCodes/'
dirfolder = '/Users/scssali/Dataset/p2ilf_final_evaluation_ALI/docker_evaluation/evaluationCodes/'

team = teamNames[4]
dirfolderTeam = os.path.join(dirfolder, team)

# List of input JSON files
json_files = ['eval_test_patientp1_0005.json', 'eval_test_patientp1_0009.json', 'eval_test_patientp1_0095.json', 'eval_test_patientp1_0508.json',
              'eval_test_patientp1_0514.json', 'eval_test_patientp1_0533.json', 'eval_test_patientp1_0684.json', 'eval_test_patientp1_1155.json',
              'eval_test_patientp1_1160.json','eval_test_patientp2_0135.json', 'eval_test_patientp2_0136.json', 'eval_test_patientp2_0442.json',
              'eval_test_patientp2_0465.json', 'eval_test_patientp2_0639.json', 'eval_test_patientp2_0795.json', 'eval_test_patientp2_0852.json', 'eval_test_patientp2_1075.json']

json_files = ['eval_test_DSC_patient4_3.json', 'eval_test_DSC_patient4_4.json', 'eval_test_DSC_patient4_7.json', 'eval_test_DSC_patient4_11.json',
              'eval_test_DSC_patient4_17.json', 'eval_test_DSC_patient4_20.json', 'eval_test_DSC_patient4_21.json', 'eval_test_DSC_patient4_22.json',
              'eval_test_DSC_patient11_2.json','eval_test_DSC_patient11_3.json', 'eval_test_DSC_patient11_4.json', 'eval_test_DSC_patient11_5.json',
              'eval_test_DSC_patient11_6.json', 'eval_test_DSC_patient11_7.json', 'eval_test_DSC_patient11_8.json', 'eval_test_DSC_patient11_9.json']

# List to store all headers
all_headers = []


os.chdir(dirfolderTeam)

for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)
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
with open(os.path.join(dirfolder, team + '_output_DSC.csv'), 'w', newline='') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(all_headers)

    # Loop through each JSON file and write data to CSV file
    for file in json_files:
        with open(file, 'r') as f1:
            data = json.load(f1)
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

print('Data has been written to output.csv file.')
