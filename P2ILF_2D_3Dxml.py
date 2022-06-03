#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 19:11:26 2022

@author: sharib
@P2ILF challenge 
"""

''' Converting to xml file - read and write @ P2ILF challenge'''

import xml.etree.ElementTree as ET
import xml.dom.minidom

'''
TODO 1: Read and write the X and Y co-ordinates and write sequencially for each image number 
    - 		<contourType>Silhouette</contourType>
    		<imagePoints>
			<numOfPoints>1305</numOfPoints> <x> </x> <y> </y>
            <contourType>Silhouette</contourType>
            		<imagePoints>
        			<numOfPoints>300</numOfPoints> <x> </x> <y> </y>
                    
                    
    - 		<contourType>Ridges</contourType>
    		<imagePoints>
			<numOfPoints>1305</numOfPoints> <x> </x> <y> </y>
            <contourType>Ridges</contourType>
            		<imagePoints>
        			<numOfPoints>300</numOfPoints> <x> </x> <y> </y>
                    
    - 
                    
    
'''


'''Get elements by Tag contour inside tag: contours
XML structure: 
Contours
  Contour
     | -- ridges/ligament
     | -- image points (NofPoints)
     | -- vertices (NofPoints)
     | -- 
'''

def write_imagePoints(ET, doc1, df):
    ET.SubElement(doc1, "numOfPoints").text =  str(len(df['x']))
    ET.SubElement(doc1, "x").text =  str(', '.join(map(str, df['x'].to_numpy().flatten().tolist())))
    ET.SubElement(doc1, "y").text =  str(', '.join(map(str, df['y'].to_numpy().flatten().tolist())))
    ET.ElementTree(doc1)
    return ET
    
def write_3Dvertices(ET, doc2, vertices_, indx):
    ET.SubElement(doc2, "numOfPoints").text = str(len(convert_string_2_arrayInts(vertices_[indx])))
    ET.SubElement(doc2, "vertices").text = str(vertices_[indx])
    ET.ElementTree(doc2)
    return ET
    

def convert_string_2_arrayInts(vertices_):
    NpointsArraySplit = vertices_.split(",")
    desired_array = [int(numeric_string) for numeric_string in NpointsArraySplit]
    return desired_array
                
def get_args():
    
    import argparse
    parser = argparse.ArgumentParser(description="P2ILF @MICCAI challenge 2022", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # test data
    parser.add_argument("--csvDir", type=str, default="/Users/sharib/Datasets/P2ILF-challenge-dataset/P2ILF-challenge2022/annotations_miccai_challenge/2D_landmarks_P2ILF/P2ILF-2Dseg_dataset/2DLandmarks_CSV_patients1-7/", help="csv file annotations)")
    parser.add_argument("--orginalImageDir", type=str, default="/Users/sharib/Datasets/P2ILF-challenge-dataset/P2ILF-challenge2022/annotations_miccai_challenge/2D_landmarks_P2ILF/P2ILF-2Dseg_dataset/images_1-5/", help="original image, jpg")
    parser.add_argument("--xmlFile3DDirectory", type=str, default="/Users/sharib/Datasets/P2ILF-challenge-dataset/P2ILF-challenge2022/annotations_miccai_challenge/2D_landmarks_P2ILF/P2ILF-2Dseg_dataset/annotations_miccai_challenge_patients1-5/", help="combined html")
    parser.add_argument("--newxmlFile2D_3DDirectory", type=str, default="/Users/sharib/Datasets/P2ILF-challenge-dataset/P2ILF-challenge2022/annotations_miccai_challenge/2D_landmarks_P2ILF/P2ILF-2Dseg_dataset/P2ILF_2D-3D_annotations_Patient1-5/", help="combined csv")
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import numpy as np
    import glob
    import os
    import re
    import pandas as pd
    
    args = get_args()
    
    os.makedirs(args.newxmlFile2D_3DDirectory, exist_ok=True)
    
    # categoryList = ['suspected', 'hgd', 'cancer']
    categoryList = ['ridge', 'silhouette', 'line', 'boundary']
    
    patientInfo = ['patient1', 'patient2', 'patient3', 'patient4', 'patient5']
    
    ext = ['*.jpg']
    imageNo = 0
    for filename in sorted(glob.glob(args.orginalImageDir + '/'+ ext[0], recursive = True)):
        
        file=filename.split('/')[-1]
        fileNameOnly = file.split('.')[0]
        patientID = fileNameOnly.split('_')[0]
        
        # finding numbers in the name
        temp = re.findall(r'\d+', file)
        res = list(map(int, temp))
            
        
        ''' read 3D vertices in the xml files'''
        XmlFile_3D = os.path.join(args.xmlFile3DDirectory, patientID, str(res[1]), 'contours_model.xml')
        XmlFile_3D_values = xml.dom.minidom.parse(XmlFile_3D)
        
        contourNums = XmlFile_3D_values.getElementsByTagName("contour") #contourEach.tagName
        root = ET.Element("contours")
        
        ET.SubElement(root, "numOfContours").text = str(len(contourNums))
        
        ctypeD_names=[]
        vertices_ = []
        for i in range(0, len(contourNums)):
            print(i)
            contourEach = contourNums[i] 
            ctypeD  = contourEach.getElementsByTagName("contourType")[0].lastChild._data
            ctypeD_names.append(ctypeD)
            vertices_.append(contourEach.getElementsByTagName("vertices")[0].lastChild._data)
            
        
        
        if imageNo == '14':
            print('Debug')
        
        print('contour types in 3D annotated:', ctypeD_names)
        ''' read CSV files to write on xml file'''
        nClasses_annotated = len(glob.glob1(args.csvDir,fileNameOnly+"_*.csv"))
        fileList = glob.glob1(args.csvDir,fileNameOnly+"_*.csv")
        
        ''' create new html file'''
        root1 = ET.Element("contours")
        ET.SubElement(root1, "numOfContours").text = str(len(fileList))
        
        
    
        lrval = 0
        indices = [0]
        for i in range (0, nClasses_annotated):
            f = fileList[i].split('_')[-1]
            
            # TODO: explicit sorting to get it 0, 1 and then 2 ... in case multiple exists
            if f.split('.')[0] == 'line':
                doc = ET.SubElement(root1, "contour")
                ET.SubElement(doc, "contourType").text = 'Ligament'
                
                fileNameCSV = fileList[i]
                df = pd.read_csv(os.path.join(args.csvDir,fileNameCSV))
                
                doc1 = ET.SubElement(doc, "imagePoints")
                ET = write_imagePoints(ET, doc1, df)
        
                doc2 = ET.SubElement(doc, "modelPoints")
                indx = ctypeD_names.index('Ligament')
                
                ET = write_3Dvertices(ET, doc2, vertices_, indx)
                ET.ElementTree(doc)
             
            # Only 2D points for this
            elif f.split('.')[0] == 'silhouette': #
                doc = ET.SubElement(root1, "contour")
                ET.SubElement(doc, "contourType").text = 'Silhouette'
                
                fileNameCSV = fileList[i]
                df = pd.read_csv(os.path.join(args.csvDir,fileNameCSV))
                
                doc1 = ET.SubElement(doc, "imagePoints")
                ET = write_imagePoints(ET, doc1, df)
                ET.ElementTree(doc)
            
            
            elif f.split('.')[0] == 'ridge': #
            # Step1: put    <contour> <contourType>Ridge</contourType> <imagePoints> <numOfPoints>566</numOfPoints> <x> <y>
                doc = ET.SubElement(root1, "contour")
                ET.SubElement(doc, "contourType").text = 'Ridge'
                
                fileNameCSV = fileList[i]
                df = pd.read_csv(os.path.join(args.csvDir,fileNameCSV))
                
                doc1 = ET.SubElement(doc, "imagePoints")
                ET = write_imagePoints(ET, doc1, df)
        
                if (lrval <= len(indices)-1):
                    doc2 = ET.SubElement(doc, "modelPoints")
                    indx = ctypeD_names.index('Ridge')
                    indices = [index for index, element in enumerate(ctypeD_names) if element == 'Ridge']
                    ET = write_3Dvertices(ET, doc2, vertices_, indx)
                    ET.ElementTree(doc)
                    lrval=lrval+1
            # TODO: write to the xml file and then take the 3D points from the above xml read file
            
        
        ET.ElementTree(root1)
        tree = ET.ElementTree(root1)
        ET.indent(tree, space="\t", level=0)
        fileNameXML2D_3d = os.path.join(args.newxmlFile2D_3DDirectory, fileNameOnly+'.xml')
        tree.write(fileNameXML2D_3d, encoding="utf-8", xml_declaration=True)
        
