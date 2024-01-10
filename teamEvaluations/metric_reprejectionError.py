#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 21:00:45 2022

@author: sharib 

Thanks to Yamid for helping with 3D liver model registration 
Read: https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf
"""

import open3d as o3d
# import os
import numpy as np
import cv2
# import matplotlib.pyplot as plt
# import pandas as pd
from shapely.geometry import LineString
import copy


from  torch_geometric.io import read_obj
# import pymesh

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

def cameraIntrinsicMat(data_params):
    Knew=[]
    
    Knew.append([float(data_params['fx']), float(data_params['skew']), float(data_params['cx'])])
    Knew.append([ 0, float(data_params['fy']), float(data_params['cy'])])
    Knew.append([float(data_params['p1']), float(data_params['p2']), 1])
    
    return np.asmatrix(Knew)


def RPE(ModelRegistered, contour_gt_3D, contour_gt_2D , K, fileName, im):
    import meshio
     
    # textured_mesh_Reg = meshio.read(ModelRegistered)
    # vertices_Reg = textured_mesh_Reg.points
    
    textured_mesh_Reg = read_obj(ModelRegistered)
    vertices_Reg = np.asarray(textured_mesh_Reg.pos)

    # triangles_Reg = np.asarray(textured_mesh_Reg.face)


    ctypeD_names=[]
    vertices_ = []
    x_ = []
    y_ = []
    xy = []
    k = 0
    for i in range(0, contour_gt_3D['numOfContours']):
        # print(i)
        contourEach = contour_gt_3D['contour'][i] 
        ctypeD  = contourEach["contourType"]
        
        contourEach_2D = contour_gt_2D['contour'][i] 
        ctypeD_2D  = contourEach_2D["contourType"]
        if ctypeD_2D == 'Silhouette' or k > i:
            k = i+1
            contourEach_2D = contour_gt_2D['contour'][k] 
            ctypeD_2D  = contourEach_2D["contourType"]
            k = k+1
        
        
        if (ctypeD == ctypeD_2D):
            ctypeD_names.append(ctypeD)
            vertices_.append(contourEach['modelPoints']["vertices"])
            x_.append(contourEach_2D['imagePoints']["x"])
            y_.append(contourEach_2D['imagePoints']["y"])

    X = []
    Y = []
    XY = []
    RotMat = np.identity(3)
    TMat = np.matrix([[ 0.0, 0.0, 0.0]])
    for i in range(len(vertices_Reg)):
        imgpoints2, _ = cv2.projectPoints(vertices_Reg[i], RotMat, TMat, K, np.float64([]))
        # print(imgpoints2)
        X.append(int(imgpoints2[0][0][0]))
        Y.append(int(imgpoints2[0][0][1]))
        XY.append([int(imgpoints2[0][0][1]), int(imgpoints2[0][0][0])])
        cv2.circle(im, (int(imgpoints2[0][0][0]), int(imgpoints2[0][0][1])), 8, (255,255,255), 3 )
        
        
    X_3D_2D = []
    Y_3D_2D = []
    XY_3D_2D = []
    

    # TMat = np.matrix([[ -float(data_params['cx']), -float(data_params['cy']), 0.0]])
    
    # Check if its registered other wise these will be not correct
    
    for idx in range(0, len(ctypeD_names)):
        vertex3D = vertices_[idx]
        if ctypeD_names[idx] == 'Ligament':
            color = (0,0,255)
        else:
            color = (255,0,0)
            
        polyLineVals = []
        for l in range(0, len(vertex3D)):
            imgpoints2, _ = cv2.projectPoints(vertices_Reg[vertex3D[l]], RotMat, TMat, K, np.float64([]))
            # print(imgpoints2)
            X_3D_2D.append(int(imgpoints2[0][0][0]))
            Y_3D_2D.append(int(imgpoints2[0][0][1]))
            
            polyLineVals.append([int(imgpoints2[0][0][1]), int(imgpoints2[0][0][0])])
            cv2.circle(im, (int(imgpoints2[0][0][0]), int(imgpoints2[0][0][1])), 8, color, 3 )
            # print(polyLineVals)
             
            # TODOL
            # error = cv2.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            # mean_error += error
            
        XY_3D_2D.append(polyLineVals)
        cv2.polylines(im, [np.flip(polyLineVals).astype('int32').reshape((-1, 1, 2))], False, color, 8)
        cv2.imwrite(fileName, im)
        
    hfd = []
    for k in range(0, len(XY_3D_2D)):
        line1 = LineString(XY_3D_2D[k])
        line2 = LineString(np.vstack((y_[k], x_[k])).T)
        dHfd = line1.hausdorff_distance(line2)
        hfd.append(dHfd)
    if len(hfd)<2:
        hfd.append(-1.0)
    # return np.mean(hfd).astype('float')
    return hfd

    
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="----- Reprojection error @ P2ILF challenge -----", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input registered model and camera parameters
    # /output/transformed-3d-liver-model.obj
    parser.add_argument("--ModelRegistered", type=str, default="/Users/sharib/Datasets/P2ILF_verified_test_PatientData/metric_codes/output/transformed-3d-liver-model_v2.obj", help="Original 3D model")
    # /input/acquisition-camera-metadata.json
    parser.add_argument("--cameraparameters", type=str, default="/Users/sharib/Datasets/P2ILF_verified_test_PatientData/metric_codes/cameraparams/calibration.json", help="supply K ")
    
    
    # This is done by us as we have GT data: GT contours 
    parser.add_argument("--contours_2D_gt", type=str, default="/Users/sharib/Datasets/P2ILF_verified_test_PatientData/metric_codes/output/patient2_1_2D-contours.json", help="2D contours")
    parser.add_argument("--contours_3D_gt", type=str, default="/Users/sharib/Datasets/P2ILF_verified_test_PatientData/metric_codes/output/patient2_1_3D-contours.json", help="2D contours")
    
    
    #keep this as it is!
    parser.add_argument("--outputFileName", type=str, default="./output/metric_RPE.json", help="2D and 3D contours")
    
    # parser.add_argument("--originalImage", type=str, default="/Users/sharib/Datasets/P2ILF_verified_test_PatientData/metric_codes/images_labels/patient2_1.jpg", help="original 2D image")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    import json
    
    args = get_args()
    
    # im = cv2.imread(args.originalImage)
    # plt.imshow(im)
    # plt.show()
    # textured_mesh_Reg_2 = read_obj(args.ModelRegistered)
    # textured_mesh_Reg = meshio.read(args.ModelRegistered)
    
    gt_2d_contours = args.contours_2D_gt
    gt_3d_contours = args.contours_3D_gt
    
    camera_params = args.cameraparameters
    f = open(camera_params)
    data_params = json.load(f)
    K = cameraIntrinsicMat(data_params)
    
    # 2D contours
    f = open(gt_2d_contours)
    contour_gt_2D = json.load(f)
    
    # 3D contour vertices : at which the evaluation will be done
    f = open(gt_3d_contours)
    contour_gt_3D = json.load(f)
    
    hfd = RPE(args.ModelRegistered, contour_gt_3D, contour_gt_2D , K)

        # print('Contour distance between 2D and 3D projection of {} is:  {}'.format(ctypeD_names[k] , dHfd))
    
    # Plot the 2D contours points
    # for l in range (0, len(x_)):
    #     if ctypeD_names[l] == 'Ligament':
    #         color = (0,0,125)
    #     else:
    #         color = (125,0,0)
        # for i in range(len(x_[l])):
        #     cv2.circle(im, (int(x_[l][i]), int(y_[l][i])), 8, color, 3 )

    # plt.imshow(im)
    # plt.show()

    my_dictionary = {"P2ILF_task2":{"RPE":{"HfD": np.mean(hfd).astype('float')}
                },
                     }
           
    print(my_dictionary) 
       
    # jsonFileName=os.path.join('./output/', 'RPE_metricvalues' + '.json')
    
    jsonFileName=args.outputFileName
    from misc_eval  import EndoCV_misc
    EndoCV_misc.write2json(jsonFileName, my_dictionary)  

