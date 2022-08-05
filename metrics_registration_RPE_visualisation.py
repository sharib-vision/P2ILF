#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 21:56:56 2022

@author: sharib

For @P2ILF challenge 2022 - visualisation - Hausdorff distance between curves 2D and projected 3D onto 2D with K and RT -

Version 1.0

- Please note that this code is written for the Phantom dataset where K and RT are provided
- You will have to estimate RT to be able to compute reprojection error in patient data or TRE or HFD distance between registered and GT mesh or contour points
- Current version: does not use deformed model


Requirements:
pip install torch-geometric
pip install openc3d


"""

import open3d as o3d
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString
import copy
from  torch_geometric.io import read_obj


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def draw_geometries_pick_points(geometries):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)
        vis.run()
    vis.destroy_window()

def visualize(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

def readtxtFile(filename):
    with open(filename) as f:
        lines = f.readlines()
    return lines
    
def stringtoMatrixK(Kvals):
    Knew=[]
    for i in range(1, len(Kvals)):
       vals = Kvals[i].split('\n')[0].split(' ')
       
       Knew.append([float(vals[0]), float(vals[1]), float(vals[2])])
       
    return np.asmatrix(Knew)
        
def stringtoMatrixRT(Kvals):
    Knew=[]
    Tnew = []
    for i in range(1, len(Kvals)):
       vals = Kvals[i].split('\n')[0].split(' ')
       
       Knew.append([float(vals[0]), float(vals[1]), float(vals[2])])
       Tnew.append(float(vals[3]))
       
    return np.asmatrix(Knew), np.asmatrix(Tnew)

# convert string to array 
def splitStringtoArray(f):
    map(int, f.split(','))

    '''
    Input:

        objectPoints Array of object points, Nx3/Nx1x3/1xNx3 array or cell array of 3-element vectors {[x,y,z],...}, where N is the number of points in the view.
        rvec Rotation vector or matrix (3x1/1x3 or 3x3). See cv.Rodrigues for details.
        tvec Translation vector (3x1/1x3).
        cameraMatrix Camera matrix 3x3, A = [fx 0 cx; 0 fy cy; 0 0 1].
        
        Note: Ridge is always from the left to right
            
    '''
    
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Reprojection error @ P2ILF challenge @ MICCAI 2022", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ModelRef", type=str, default="/Users/sharib/Datasets/example_p2ilf_metrics/phantom_annotations_P2ILF/referenceOriModel/originalMesh.obj", help="Original 3D model")
    parser.add_argument("--Model", type=str, default="deformedMesh/deformedMesh.obj", help="Original 3D model")
  
    parser.add_argument("--camerainstrinics", type=str, default="K.txt", help="supply K ")
    parser.add_argument("--cameraextrinsic", type=str, default="RT.txt", help="R|T")
    
    parser.add_argument("--ridgeLigamentLandmarks2D_3D", type=str, default="contours/contours.xml", help="2D and 3D contours")
    parser.add_argument("--originalImage", type=str, default="04_undistorted.png", help="original 2D image")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    basePath = '/Users/sharib/Datasets/example_p2ilf_metrics/phantom_annotations_P2ILF/phantomMesh02/'
    imagenumber = 'image04'
    
    args = get_args()
    Kread = readtxtFile(os.path.join(basePath,imagenumber, args.camerainstrinics))
    RTread = readtxtFile(os.path.join(basePath, imagenumber, args.cameraextrinsic))
    
    K = stringtoMatrixK(Kread)
    RT = stringtoMatrixRT(RTread)
    
    #Load 3D model Reference
    textured_mesh_GT = read_obj(args.ModelRef)
    vertices_GT = np.asarray(textured_mesh_GT.pos)
    triangles_GT = np.asarray(textured_mesh_GT.face)
    
    # Load 3D model deformed 
    # TODO: provide deformed model! 
    # textured_mesh_deformed = read_obj(os.path.join(basePath, args.Model))
    # vertices_def = np.asarray(textured_mesh_deformed.pos)
    # triangles_def = np.asarray(textured_mesh_deformed.face)
    
    # Read image for plotting the projected 3D points on to the undistorted model
    im = cv2.imread(os.path.join(basePath, imagenumber, args.originalImage))
    plt.imshow(im)
    plt.show()


    # ----> Are they registered?
    transformApply = 0
    target_mesh = o3d.io.read_triangle_mesh(os.path.join(basePath, args.Model))
    source_mesh =  o3d.io.read_triangle_mesh(args.ModelRef)
    
    if transformApply:
        source_mesh.rotate(RT[0], center=(0, 0, 0))
        source_mesh.translate((RT[1][0,0], RT[1][0,1], RT[1][0,2]))
    draw_registration_result(target_mesh, source_mesh, np.identity(4))
    
    # Reprojection Error - entire 3D vertices
    mean_error = 0
    X = []
    Y = []
    XY = []
    
    for i in range(len(vertices_GT)):
        imgpoints2, _ = cv2.projectPoints(vertices_GT[i], RT[0], RT[1], K, np.float64([]))
        # print(imgpoints2)
        X.append(int(imgpoints2[0][0][0]))
        Y.append(int(imgpoints2[0][0][1]))
        XY.append([int(imgpoints2[0][0][1]), int(imgpoints2[0][0][0])])
        cv2.circle(im, (int(imgpoints2[0][0][0]), int(imgpoints2[0][0][1])), 8, (255,255,255), 3 )
  
    # cv2.polylines(im, [np.flip(XY).astype('int32').reshape((-1, 1, 2))], False, (255,255,255), 2)     
    
    # Reprojection Error - only between ridges and ligament 3D vertices
    contours = os.path.join(basePath, imagenumber, args.ridgeLigamentLandmarks2D_3D)
    import xml.dom.minidom
    import xml.etree.ElementTree as ET
    XmlFile_3D_values = xml.dom.minidom.parse(contours)
    
    contourNums = XmlFile_3D_values.getElementsByTagName("contour") #contourEach.tagName
    root = ET.Element("contours")
    
    ET.SubElement(root, "numOfContours").text = str(len(contourNums))
    
    ctypeD_names=[]
    vertices_ = []
    x_ = []
    y_ = []
    xy = []
    for i in range(0, len(contourNums)):
        # print(i)
        contourEach = contourNums[i] 
        ctypeD  = contourEach.getElementsByTagName("contourType")[0].lastChild._data
        
        if ctypeD !='Silhouette':
            ctypeD_names.append(ctypeD)
            vertices_.append(list(map(int, contourEach.getElementsByTagName("vertices")[0].lastChild._data.split(','))))
            x_.append(list(map(int, contourEach.getElementsByTagName("x")[0].lastChild._data.split(','))))
            y_.append(list(map(int, contourEach.getElementsByTagName("y")[0].lastChild._data.split(','))))
            
            # xy.append([list(map(int, contourEach.getElementsByTagName("x")[0].lastChild._data.split(','))),list(map(int, contourEach.getElementsByTagName("y")[0].lastChild._data.split(',')))])


    
    X_3D_2D = []
    Y_3D_2D = []
    XY_3D_2D = []
    for idx in range(0, len(ctypeD_names)):
        vertex3D = vertices_[idx]
        if ctypeD_names[idx] == 'Ligament':
            color = (0,0,255)
        else:
            color = (255,0,0)
            
        polyLineVals = []
        for l in range(0, len(vertex3D)):
            imgpoints2, _ = cv2.projectPoints(vertices_GT[vertex3D[l]], RT[0], RT[1], K, np.float64([]))
            # print(imgpoints2)
            X_3D_2D.append(int(imgpoints2[0][0][0]))
            Y_3D_2D.append(int(imgpoints2[0][0][1]))
            
            polyLineVals.append([int(imgpoints2[0][0][1]), int(imgpoints2[0][0][0])])
            cv2.circle(im, (int(imgpoints2[0][0][0]), int(imgpoints2[0][0][1])), 8, color, 3 )
             
            # TODOL
            # error = cv2.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            # mean_error += error
            
        XY_3D_2D.append(polyLineVals)
        cv2.polylines(im, [np.flip(polyLineVals).astype('int32').reshape((-1, 1, 2))], False, color, 8)  
     
    
    '''3D visulisation
      OR - Uncomment if you want to do 3D visualisation but no contours in current version
     Press - W for wireframe visualisation
    '''
    coord = o3d.geometry.TriangleMesh().create_coordinate_frame()
    # mesh = o3d.io.read_triangle_mesh(os.path.join(basePath, args.Model))
    mesh = o3d.io.read_triangle_mesh(args.ModelRef)
    
    print("Painting the mesh")
    mesh.paint_uniform_color([1, 0.706, 0])
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=20, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([mesh , mesh_frame])
    
    '''
        distance between two closest points - after projection between 3D and 2D points - generate a curve after projecting and - closet point to a give projected 3D points! - HFD
        distance between boundaries - HFD 
    '''
    hfd = []
    for k in range(0, len(XY_3D_2D)):
        line1 = LineString(XY_3D_2D[k])
        line2 = LineString(np.vstack((y_[k], x_[k])).T)
        dHfd = line1.hausdorff_distance(line2)
        hfd.append(dHfd)
        print('Contour distance between 2D and 3D projection of {} is:  {}'.format(ctypeD_names[k] , dHfd))
    
    # Plot the 2D contours points
    for l in range (0, len(x_)):
        if ctypeD_names[l] == 'Ligament':
            color = (0,0,125)
        else:
            color = (125,0,0)
        for i in range(len(x_[l])):
            cv2.circle(im, (int(x_[l][i]), int(y_[l][i])), 8, color, 3 )


    plt.imshow(im)
    plt.show()
    

    
    

        
        
        
        
        