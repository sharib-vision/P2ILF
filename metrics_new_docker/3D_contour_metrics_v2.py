#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 00:27:04 2022

@author: sharib
"""

import os
import numpy as np

def Hausdorff_dist(vol_a,vol_b):
    dist_lst = []
    for idx in range(len(vol_a)):
        dist_min = 1000.0        
        for idx2 in range(len(vol_b)):
            dist= np.linalg.norm(vol_a[idx]-vol_b[idx2])
            if dist_min > dist:
                dist_min = dist
        dist_lst.append(dist_min)
    return np.max(dist_lst)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="3D contour distance metric - task 1 @ P2ILF challenge @ MICCAI 2022", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ModelRef", type=str, default="../input/3d-liver-model.obj", help="Original 3D model - GT reference model")
    parser.add_argument("--ridgeLigamentLandmarks_3D_GT", type=str, default="../evaluation_GT/patient2_1_3D-contours.json", help="2D and 3D contours")
    parser.add_argument("--ridgeLigamentLandmarks_3D_eval", type=str, default="../output/3d-liver-contours.json", help="2D and 3D contours")
    args = parser.parse_args()
    return args

def findContoursfromJSONVertices(contours_file3D):
    f = open(contours_file3D)
    data = json.load(f)
    
    contourNums = data["numOfContours"]
    ctypeD_names=[]
    vertices_ = []
    for i in range(0, contourNums):
        # print(i)
        contourEach = data['contour'][i]
        ctypeD  = contourEach["contourType"]
        ctypeD_names.append(ctypeD)
        vertices_.append(contourEach['modelPoints']["vertices"])
    return contourNums, vertices_, ctypeD_names


 
def distance(cords3D_gt, cords3D_eval):
    import math
    d = math.sqrt(math.pow(cords3D_gt[0] - cords3D_eval[0], 2) +
                math.pow(cords3D_gt[1] - cords3D_eval[1], 2) +
                math.pow(cords3D_gt[2] - cords3D_eval[2], 2)* 1.0)
    return d

if __name__ == "__main__":
    
    from  torch_geometric.io import read_obj
    import open3d as o3d
    import json
    from misc import EndoCV_misc
    
    args = get_args()
    
    textured_mesh_GT = read_obj(args.ModelRef)
    vertices_GT = np.asarray(textured_mesh_GT.pos)
    contours_GT = os.path.join( args.ridgeLigamentLandmarks_3D_GT)
    
    
    filename = contours_GT.split('/')[-1].split('.')[0]
    contourNumsGT, vertices_GT_contour, ctypeD_names = findContoursfromJSONVertices(contours_GT)
    
    #evaluation vertices
    contours_eval = os.path.join(args.ridgeLigamentLandmarks_3D_eval)
    contourNums_eval, vertices_eval_contour, ctypeD_names_eval = findContoursfromJSONVertices(contours_eval)
    
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    
    'Comment: make sure you have correct order -  Ridges from left to right and Ligament from top to bottom'
    
    dist_NN=[]
    dist_HFD = []
    
    '''Flag for participants with no prediction on 3D contours'''
    if contourNums_eval == 0:
        print('3D contour file is empty - task not completed' )
        dist_NN = [1000, 1000, 1000]
        dist_HFD = [1000, 1000, 1000]
    else:
        if len(ctypeD_names_eval) < len(ctypeD_names):
            print('variable predictions found with evaluation less than GT')
        
        ''' Note only first few predictions will be taken into account (Ridge -> L to R) - if you predict more then this will be discarded'''
        for idx in range(0, len(ctypeD_names)):
            if (ctypeD_names[idx] == ctypeD_names_eval[idx] and ctypeD_names_eval[idx]=='Ridge'):
                vertex3D_GT = vertices_GT_contour[idx]
                vertex3D_eval = vertices_eval_contour[idx]
                dist = []
               
                cords3D_gt = []
                for k in range(0, len(vertex3D_GT)):
                    cords3D_gt.append(vertices_GT[vertex3D_GT[k]])
                
                # find distance from nearest neighbor in GT
                pcd.points = o3d.utility.Vector3dVector(cords3D_gt)
                distances_GT = pcd.compute_nearest_neighbor_distance()
                avg_dist_GT = np.mean(distances_GT)
                # print(avg_dist_GT)
                # o3d.visualization.draw_geometries([pcd])
                    
                cords3D_eval = []
                for k in range(0, len(vertex3D_eval)):
                    cords3D_eval.append(vertices_GT[vertex3D_eval[k]])  
                
                # find distance from nearest neighbor in predicted
                pcd = o3d.geometry.PointCloud()   
                pcd.points = o3d.utility.Vector3dVector(cords3D_eval)
                # o3d.visualization.draw_geometries([pcd])
                distances_eval = pcd.compute_nearest_neighbor_distance()
                avg_dist_eval = np.mean(distances_eval)
                # print(avg_dist_eval)
                
                # Note: the distance difference will penalise the score : min -> best (ideally 0)
                distNN_diff = np.asarray(avg_dist_eval)-np.asarray(avg_dist_GT)
                dist_NN.append(distNN_diff)
                print(distNN_diff)
                valHFD = Hausdorff_dist(cords3D_gt, cords3D_eval)
                dist_HFD.append(valHFD)
            elif (ctypeD_names[idx] == 'Ligament'):
                # print(ctypeD_names[idx])
                try:
                    idx2 = ctypeD_names_eval.index('Ligament')
                    vertex3D_GT = vertices_GT_contour[idx]
                    vertex3D_eval = vertices_eval_contour[idx2]
                    dist = []
                   
                    cords3D_gt = []
                    for k in range(0, len(vertex3D_GT)):
                        cords3D_gt.append(vertices_GT[vertex3D_GT[k]])
                    
                    # find distance from nearest neighbor in GT
                    pcd.points = o3d.utility.Vector3dVector(cords3D_gt)
                    distances_GT = pcd.compute_nearest_neighbor_distance()
                    avg_dist_GT = np.mean(distances_GT)
                    # print(avg_dist_GT)
                    # o3d.visualization.draw_geometries([pcd])
                        
                    cords3D_eval = []
                    for k in range(0, len(vertex3D_eval)):
                        cords3D_eval.append(vertices_GT[vertex3D_eval[k]])  
                    
                    # find distance from nearest neighbor in predicted
                    pcd = o3d.geometry.PointCloud()   
                    pcd.points = o3d.utility.Vector3dVector(cords3D_eval)
                    # o3d.visualization.draw_geometries([pcd])
                    distances_eval = pcd.compute_nearest_neighbor_distance()
                    avg_dist_eval = np.mean(distances_eval)
                    # print(avg_dist_eval)
                    
                    # Note: the distance difference will penalise the score : min -> best (ideally 0)
                    distNN_diff = np.asarray(avg_dist_eval)-np.asarray(avg_dist_GT)
                    dist_NN.append(distNN_diff)
                    print(distNN_diff)
                    valHFD = Hausdorff_dist(cords3D_gt, cords3D_eval)
                    dist_HFD.append(valHFD)
                    
                except:
                    print("Ligament not predicted found")
        
    my_dictionary = {"P2ILF_3D_contours":{"Distances":{"distNN_diff": np.mean(dist_NN).astype('float'), "distHF": np.mean(dist_HFD).astype('float')}
                },
                     }
                   
    jsonFileName=os.path.join('../output/',  filename + '_metricvalues' + '.json')
    
    EndoCV_misc.write2json(jsonFileName, my_dictionary)  
    