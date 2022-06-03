#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 19:11:26 2022

@author: sharib
"""

''' Converting to xml file - read and write @ P2ILF challenge'''

import xml.etree.ElementTree as ET
import xml.dom.minidom


file = 'contours_sample.xml'
doc = xml.dom.minidom.parse(file)


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
contourNums = doc.getElementsByTagName("contour") #contourEach.tagName
root = ET.Element("contours")

# xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ", encoding="utf-8")
ET.SubElement(root, "numOfContours").text = str(len(contourNums))

for i in range(0, len(contourNums)):
    print(i)
    contourEach = contourNums[i] 
    ctypeD  = contourEach.getElementsByTagName("contourType")[0].lastChild._data
    
    # Image points (x, y)
    imagePoints_length = contourEach.getElementsByTagName("numOfPoints")[0].lastChild._data
    x1 = contourEach.getElementsByTagName("imagePoints")[0].getElementsByTagName("x")[0]
    x1_val =  x1.lastChild._data
    print('x:', x1.lastChild._data)
    y1 = contourEach.getElementsByTagName("imagePoints")[0].getElementsByTagName("y")[0]
    y1_val =  y1.lastChild._data
    print('y:', y1.lastChild._data)
    
    # Get how many number of points each contour has ? - if only then no 3D correspondence 
    Npoints = len(contourEach.getElementsByTagName('numOfPoints'))
    
    if Npoints>1:
       nvertices = contourEach.getElementsByTagName("numOfPoints")[1].lastChild._data
       model3D =  contourEach.getElementsByTagName("vertices")
       model3D_vertices = model3D[0].lastChild._data

    doc = ET.SubElement(root, "contour")
    
    ET.SubElement(doc, "contourType").text = ctypeD
    doc1 = ET.SubElement(doc, "imagePoints")
    ET.SubElement(doc1, "numOfPoints").text =  imagePoints_length
    ET.SubElement(doc1, "x").text =  x1.lastChild._data
    ET.SubElement(doc1, "y").text =  y1.lastChild._data
    ET.ElementTree(doc1)
    if  Npoints>1:
        doc2 = ET.SubElement(doc, "modelPoints")
        ET.SubElement(doc2, "numOfPoints").text = nvertices
        ET.SubElement(doc2, "vertices").text = model3D_vertices
    
    
    ET.ElementTree(doc)
    # root1.append(root)
    
tree = ET.ElementTree(root)
# 
ET.indent(tree, space="\t", level=0)
tree.write("test_writer.xml", encoding="utf-8")



# # from lxml import etree
# # xml_object = xml.etree.ElementTree.tostring(root, pretty_print=True, xml_declaration=True,encoding='UTF-8')
# from lxml import etree
# xmlstr = etree.tostring(root, pretty_print=True, xml_declaration=True,encoding='UTF-8')

# with open("Test.xml", "w", encoding='utf-8') as f:
#     f.write(xmlstr)
    
    
    
# tree = ET.ElementTree(root)
# tree.write("test_writer.xml")