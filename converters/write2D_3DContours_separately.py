#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 20:35:30 2022

@author: sharib

@ For Type 2 challenge producing two separate 2D and 3D contour files 
(refer here: https://p2ilf.grand-challenge.org/submission-instructions/)

 - writes 2d-liver-contours.json and 3d-liver-contours.json separately - Please note that this depends on json converted file from xml
 - You can directly write your predictions in the current file format - please check if you have all entries correct
"""


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="segmentation metrics - 2D", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--combined2D_3DContour", type=str, default="../samplexmlprovided/patient2_1.json", help="combined contours")
    parser.add_argument("--separateFiles", type=str, default="../samplexmlprovided/", help="converter json 2D and 3D - separated files")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    import json
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    
    args = get_args()
    
    inputFile = args.combined2D_3DContour
    filename = inputFile.split('/')[-1].split('.')[0]
    f = open(inputFile)
    data = json.load(f)
    
    array1 = []
    
    # For 2D only 
    for k in range(0,data['contours']['numOfContours'][0]):
        # 2D contours will be all - ridge, ligament and silhoutte
        
        # preambles: numOfContours, 
        
        # 1) Define contour type
        cType = data['contours']['contour'][k]['contourType']
        
        # 2) Define image points: {"x": {"type": "array", "items": {"type": "integer"}}, "y": {"type": "array", "items": {"type": "integer"}}
        imagePointsX = data['contours']['contour'][k]['imagePoints']['x']
        imagePointsY = data['contours']['contour'][k]['imagePoints']['y']
        array1.append( {"contourType": cType, "imagePoints": {'x': imagePointsX, 'y': imagePointsY} })
        
    my_dictionary = {
        "numOfContours": data['contours']['numOfContours'][0], "contour": array1 }
    
    jsonFileName=os.path.join(args.separateFiles, filename+'_2D-contours.json')
    # in your docker for each image you will use: jsonFileName = /output/2d-liver-contours.json 

    fileObj= open(jsonFileName, "a")
    # fileObj.write("\n")
    json.dump(my_dictionary, fileObj)
    fileObj.close()
    

    # 3) Define model points
    array1 = []
    indexContour = 0
    for k in range(0,data['contours']['numOfContours'][0]):
        
        # preambles: numOfContours, 
        
        # 1) Define contour type
        cType = data['contours']['contour'][k]['contourType']
        if cType != 'Silhouette':
            modelPoints = data['contours']['contour'][k]['modelPoints']['vertices']
            vertices = data['contours']['contour'][k]['modelPoints']['vertices']
            array1.append( {"contourType": cType, "modelPoints": {'vertices': vertices} })
            indexContour+=1
            
    my_dictionary = {
        "numOfContours": indexContour, "contour": array1 }
    
    jsonFileName=os.path.join(args.separateFiles, filename+'_3D-contours.json') 
    # in your docker for each image you will use: jsonFileName = /output/3d-liver-contours.json 
 
    fileObj= open(jsonFileName, "a")
    # fileObj.write("\n")
    json.dump(my_dictionary, fileObj)
    fileObj.close() 
 
    # {"type": "object", "required": ["numOfContours", "contour"], "properties": {"contour": {"type": "array", "items": {"type": "object", "required": ["contourType", "modelPoints"], "properties": {"contourType": {"type": "string"}, "modelPoints": {"type": "object", "required": ["vertices"], "properties": {"vertices": {"type": "array", "items": {"type": "integer"}}}}}}}, "numOfContours": {"type": "integer"}}}
     # "modelPoints": {"type": "object", "required": ["vertices"], "properties": {"vertices": {"type": "array", "items": {"type": "integer"}}
    
    
        