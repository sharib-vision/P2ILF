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
    parser.add_argument("--ModelRef", type=str, default="/Users/sharib/Documents/docker_evaluation/GT/patient11_3D-liver-model.obj", help="Original 3D model - GT reference model")
    parser.add_argument("--ridgeLigamentLandmarks_3D_GT", type=str, default="/Users/sharib/Documents/docker_evaluation/GT/patient11_2_3D-contours.json", help="2D and 3D contours")
    parser.add_argument("--ridgeLigamentLandmarks_3D_eval", type=str, default="/Users/sharib/Documents/docker_evaluation/pred/3d-liver-contours.json", help="2D and 3D contours")
    parser.add_argument("--outputFileName", type=str, default="./output/metric_3D.json", help="2D and 3D contours")
    args = parser.parse_args()
    return args

def findContoursfromJSONVertices(contours_file3D):
    import json
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

def metric3Dcompute(ModelRef, contours_3D_GT, contours_3D_eval):
    # from  torch_geometric.io import read_obj
    import meshio
    import open3d as o3d
    
    textured_mesh_Reg = meshio.read(ModelRef)
    # textured_mesh_GT = read_obj(ModelRef)
    # vertices_GT = np.asarray(textured_mesh_GT.pos)
    vertices_GT = textured_mesh_Reg.points
    # contours_GT = os.path.join()
    
    
    # filename = contours_GT.split('/')[-1].split('.')[0]
    contourNumsGT, vertices_GT_contour, ctypeD_names = findContoursfromJSONVertices(contours_3D_GT)
    
    #evaluation vertices
    
    contourNums_eval, vertices_eval_contour, ctypeD_names_eval = findContoursfromJSONVertices(contours_3D_eval)
    
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
    # else:
    #     if len(ctypeD_names_eval) < len(ctypeD_names):
    #         print('variable predictions found with evaluation less than GT')
            
    else:
        
        ''' Note only first few predictions will be taken into account (Ridge -> L to R) - if you predict more then this will be discarded'''
        
        try:
            # Concatenate ridge contours for GT and eval data:
            cords3D_ridge_gt = []
            cords3D_ridge_eval = []
            for idx in range(0, len(ctypeD_names)):
                if (ctypeD_names[idx]=='Ridge'):
                    vertex3D_GT = vertices_GT_contour[idx]
                    for k in range(0, len(vertex3D_GT)):
                        cords3D_ridge_gt.append(vertices_GT[vertex3D_GT[k]])
            
            for idx in range(0, len(ctypeD_names_eval)):
                if (ctypeD_names_eval[idx]=='Ridge'):
                    vertex3D_eval = vertices_eval_contour[idx]
                    for k in range(0, len(vertex3D_eval)):
                        cords3D_ridge_eval.append(vertices_GT[vertex3D_eval[k]])
                        
            # find distance from nearest neighbor in GT
            pcd.points = o3d.utility.Vector3dVector(cords3D_ridge_gt)
            distances_GT = pcd.compute_nearest_neighbor_distance()
            avg_dist_GT = np.mean(distances_GT)
            # print(avg_dist_GT)
            # o3d.visualization.draw_geometries([pcd])
            
            # find distance from nearest neighbor in predicted
            pcd = o3d.geometry.PointCloud()   
            pcd.points = o3d.utility.Vector3dVector(cords3D_ridge_eval)
            # o3d.visualization.draw_geometries([pcd])
            distances_eval = pcd.compute_nearest_neighbor_distance()
            avg_dist_eval = np.mean(distances_eval)
            # print(avg_dist_eval)
            
            # Note: the distance difference will penalise the score : min -> best (ideally 0)
            distNN_diff = np.abs(np.asarray(avg_dist_eval)-np.asarray(avg_dist_GT))
            dist_NN.append(distNN_diff)
           
            valHFD = Hausdorff_dist(cords3D_ridge_gt, cords3D_ridge_eval)
            dist_HFD.append(valHFD)
           
            
        except:
            print("Ridge not found in either GT or predicted data")
          
            
        try:
            lig_flag = 0
            for idx in range(0, len(ctypeD_names)):
                cords3D_ligament_gt = []
                cords3D_ligament_eval = []
                # This will execute only if ligament is present in the GT
                if (ctypeD_names[idx]=='Ligament'):
                    # Concatenate ligaments contours for GT and eval data:
                    # for idx in range(0, len(ctypeD_names)):

                    vertex3D_GT = vertices_GT_contour[idx]
                    
                    for k in range(0, len(vertex3D_GT)):
                        cords3D_ligament_gt.append(vertices_GT[vertex3D_GT[k]])
                    
                    for idx in range(0, len(ctypeD_names_eval)):
                        if (ctypeD_names_eval[idx]=='Ligament'):
                            lig_flag = 1
                            vertex3D_eval = vertices_eval_contour[idx]
                            for k in range(0, len(vertex3D_eval)):
                                cords3D_ligament_eval.append(vertices_GT[vertex3D_eval[k]])

                    # Deals with nan --> we take these into account only if both GT and prediction have this -- 
                    if lig_flag:      
                        # find distance from nearest neighbor in GT
                        pcd.points = o3d.utility.Vector3dVector(cords3D_ligament_gt)
                        distances_GT = pcd.compute_nearest_neighbor_distance()
                        avg_dist_GT = np.mean(distances_GT)
                        # print(avg_dist_GT)
                        # o3d.visualization.draw_geometries([pcd])
                        
                        # find distance from nearest neighbor in predicted
                        pcd = o3d.geometry.PointCloud()   
                        pcd.points = o3d.utility.Vector3dVector(cords3D_ligament_eval)
                        # o3d.visualization.draw_geometries([pcd])
                        distances_eval = pcd.compute_nearest_neighbor_distance()
                        avg_dist_eval = np.mean(distances_eval)
                        # print(avg_dist_eval)
                        
                        # Note: the distance difference will penalise the score : min -> best (ideally 0)
                        distNN_diff = np.abs(np.asarray(avg_dist_eval)-np.asarray(avg_dist_GT))
                        dist_NN.append(distNN_diff)
                        print(distNN_diff)
                        valHFD = Hausdorff_dist(cords3D_ligament_gt, cords3D_ligament_eval)
                        dist_HFD.append(valHFD)
                
        except:
            print("Ligament not found in either GT or predicted data")
                
        print(dist_HFD) 
        print(distNN_diff)
        return np.mean(dist_NN).astype('float'), np.mean(dist_HFD).astype('float')
    
if __name__ == "__main__":
    
    
    # import meshio

    import json
    from misc_eval import EndoCV_misc
    
    args = get_args()
    
    import meshio
    import open3d as o3d
    

    
    contours_3D_GT = args.ridgeLigamentLandmarks_3D_GT
    contours_3D_eval = os.path.join(args.ridgeLigamentLandmarks_3D_eval)
    
    
    textured_mesh_Reg = meshio.read(args.ModelRef)
    
    avg_dist_NN, avg_dist_HFD = metric3Dcompute(args.ModelRef, contours_3D_GT, contours_3D_eval)
        
    my_dictionary = {"P2ILF_3D_contours":{"Distances":{"distNN_diff": avg_dist_NN, "distHF": avg_dist_HFD}
                },
                     }
               
    print(my_dictionary)    
    jsonFileName = args.outputFileName
    # os.path.join('./output/',  filename + '_metricvalues' + '.json')
    EndoCV_misc.write2json(jsonFileName, my_dictionary)  
    