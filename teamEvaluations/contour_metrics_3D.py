#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 00:27:04 2022

@author: sharib

K-NN approach window size 5
BHL:
max distance: 136.76375
mean: 67.90934

NCT:
max distance: 136.76375
mean: 21.731981

UCL:
max: 33.749794
mean: 11.077423


"""

import os
import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
from pytorch3d.ops import box3d_overlap



def Hausdorff_dist(vol_a,vol_b):
    dist_lst = []
    for idx in range(len(vol_a)):
        dist_min = 0.0        
        for idx2 in range(len(vol_b)):
            dist= np.linalg.norm(vol_a[idx]-vol_b[idx2])
            if dist_min > dist:
                dist_min = dist
        dist_lst.append(dist_min)
    return np.max(dist_lst)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="3D contour distance metric - task 1 @ P2ILF challenge @ MICCAI 2022", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #GTs provided by us - should match uuid of image TODO
    parser.add_argument("--ModelRef", type=str, default="/Users/scssali/Dataset/p2ilf_final_evaluation_ALI/P2ILF_test_leaderboard/GroundTruth/patient4_3D-liver-model.obj", help="Original 3D model - GT reference model")
    parser.add_argument("--ridgeLigamentLandmarks_3D_GT", type=str, default="/Users/scssali/Dataset/p2ilf_final_evaluation_ALI/P2ILF_test_leaderboard/GroundTruth/patient4_4_3D-contours.json", help="2D and 3D contours")
   
    # /input/"job_pk"/output/3d-liver-contours.json
    parser.add_argument("--ridgeLigamentLandmarks_3D_eval", type=str, default="/Users/scssali/Dataset/p2ilf_final_evaluation_ALI/P2ILF_test_leaderboard/output/output_VOR/test1_sample_7/3d-liver-contours.json", help="2D and 3D contours")
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

def vis_save(filename, pcd, typeGT_pred):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename + typeGT_pred + '.png')
    vis.destroy_window()
 
def distance(cords3D_gt, cords3D_eval):
    import math
    d = math.sqrt(math.pow(cords3D_gt[0] - cords3D_eval[0], 2) +
                math.pow(cords3D_gt[1] - cords3D_eval[1], 2) +
                math.pow(cords3D_gt[2] - cords3D_eval[2], 2)* 1.0)
    return d

def metric3Dcompute(ModelRef, contours_3D_GT, contours_3D_eval, filename):
    from  torch_geometric.io import read_obj
    import meshio

    from pytorch3d.ops import box3d_overlap
    # textured_mesh_Reg = meshio.read(ModelRef)
    textured_mesh_GT = read_obj(ModelRef)
    vertices_GT = np.asarray(textured_mesh_GT.pos)
    # vertices_GT = textured_mesh_Reg.points
    # contours_GT = os.path.join()
    
    
    # filename = contours_GT.split('/')[-1].split('.')[0]
    contourNumsGT, vertices_GT_contour, ctypeD_names = findContoursfromJSONVertices(contours_3D_GT)
    
    #evaluation vertices
    contourNums_eval, vertices_eval_contour, ctypeD_names_eval = findContoursfromJSONVertices(contours_3D_eval)
    pcd = o3d.geometry.PointCloud()
    
    
    # print('countour points:', len(vertices_eval_contour[0]))
    # print('countour points GT:', len(vertices_GT_contour[0]))
    # pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    
    
    'Comment: make sure you have correct order -  Ridges from left to right and Ligament from top to bottom'
    
    dist_NN=[]
    dist_HFD = []
    vis = 0  #Visualisation flag
    
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
            if vis:
                # o3d.visualization.draw_geometries([pcd])
                vis_save(filename, pcd, '_GT_ridge')
                
                # find distance from nearest neighbor in predicted
                pcd = o3d.geometry.PointCloud()   
                pcd.points = o3d.utility.Vector3dVector(cords3D_ridge_eval)
                vis_save(filename, pcd, '_pred_ridge')
                # o3d.visualization.draw_geometries([pcd])
            
            distances_eval = pcd.compute_nearest_neighbor_distance()
            avg_dist_eval = np.mean(distances_eval)
            # print(avg_dist_eval)
            
            # Note: the distance difference will penalise the score : min -> best (ideally 0)
            distNN_diff = np.abs(np.asarray(avg_dist_eval)-np.asarray(avg_dist_GT))
            dist_NN.append(distNN_diff)
           
            # Take a max of the one sided squared  distances to get the two sided Hausdorff distance - v2
            valHFD = pcu.hausdorff_distance(np.array(cords3D_ridge_gt), np.array(cords3D_ridge_eval))
            
            # Compute one-sided squared Hausdorff distances -v3
            valHFD = pcu.one_sided_hausdorff_distance(np.array(cords3D_ridge_eval), np.array(cords3D_ridge_gt))[0]

            # dists_a_to_b, corrs_a_to_b = pcu.k_nearest_neighbors(np.array(cords3D_ridge_eval), np.array(cords3D_ridge_gt), 5)
            
            dists_a_to_b, corrs_a_to_b = pcu.k_nearest_neighbors(np.array(cords3D_ridge_gt), np.array(cords3D_ridge_eval), 1)
            print(dists_a_to_b.max())
            print(dists_a_to_b.mean())
            
            chamfer_dist = pcu.chamfer_distance(np.array(cords3D_ridge_gt), np.array(cords3D_ridge_eval))
            print('chamfer distance Ridge-->', chamfer_dist)

            
            # intersection_vol, iou_3d = box3d_overlap(np.array(cords3D_ridge_gt), np.array(cords3D_ridge_eval))
            
            valHFD  = chamfer_dist
            # valHFD = Hausdorff_dist(cords3D_ridge_gt, cords3D_ridge_eval) 
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
                        if vis:
                            # o3d.visualization.draw_geometries([pcd])
                            vis_save(filename, pcd, '_gt_lig')
                            
                            # find distance from nearest neighbor in predicted
                            pcd = o3d.geometry.PointCloud()   
                            pcd.points = o3d.utility.Vector3dVector(cords3D_ligament_eval)
                            vis_save(filename, pcd, '_pred_lig')
                            # o3d.visualization.draw_geometries([pcd])
                        
                        distances_eval = pcd.compute_nearest_neighbor_distance()
                        avg_dist_eval = np.mean(distances_eval)
                        # print(avg_dist_eval)
                        
                        # Note: the distance difference will penalise the score : min -> best (ideally 0)
                        distNN_diff = np.abs(np.asarray(avg_dist_eval)-np.asarray(avg_dist_GT))
                        dist_NN.append(distNN_diff)
                        print(distNN_diff)
                        
                        # Take a max of the one sided squared  distances to get the two sided Hausdorff distance -v2
                        valHFD = pcu.hausdorff_distance(np.array(cords3D_ligament_gt), np.array(cords3D_ligament_eval))
                        
                        # Compute one-sided squared Hausdorff distances -v3
                        valHFD = pcu.one_sided_hausdorff_distance(np.array(cords3D_ligament_eval), np.array(cords3D_ligament_gt))[0]
                        
                        
                        # 
                        dists_a_to_b, corrs_a_to_b = pcu.k_nearest_neighbors(np.array(cords3D_ligament_eval), np.array(cords3D_ligament_gt), 5)
                        print(dists_a_to_b.max())
                        print(dists_a_to_b.mean())
                        
                        
                        chamfer_dist = pcu.chamfer_distance(np.array(cords3D_ligament_gt), np.array(cords3D_ligament_eval))
                        print('chamfer distance Ligament-->', chamfer_dist)
                        valHFD = chamfer_dist
                        
                        # valHFD  = dists_a_to_b.mean()
                        # valHFD = Hausdorff_dist(cords3D_ligament_gt, cords3D_ligament_eval)
                        dist_HFD.append(valHFD)

                
        except:
            print("Ligament not found in either GT or predicted data")
            
        if len(dist_HFD)<2:
            dist_NN.append(-1.0)
            dist_HFD.append(-1.0)
        print(dist_HFD) 
        print(distNN_diff)
        return dist_NN, dist_HFD
        # return np.mean(dist_NN).astype('float'), np.mean(dist_HFD).astype('float')
    
if __name__ == "__main__":
    
    
    # import meshio

    import json
    from misc_eval import EndoCV_misc
    
    args = get_args()
    
   
    contours_3D_GT = args.ridgeLigamentLandmarks_3D_GT
    contours_3D_eval = os.path.join(args.ridgeLigamentLandmarks_3D_eval)
    
    avg_dist_NN, avg_dist_HFD = metric3Dcompute(args.ModelRef, contours_3D_GT, contours_3D_eval, filename='test')
    
    print(avg_dist_NN)
    print(avg_dist_HFD)
        
    # my_dictionary = {"P2ILF_3D_contours":{"Distances":{"distNN_diff": avg_dist_NN, "distHF": avg_dist_HFD}
    #             },
    #                  }
               
    # print(my_dictionary)    
    # jsonFileName = args.outputFileName
    # os.path.join('./output/',  filename + '_metricvalues' + '.json')
    # EndoCV_misc.write2json(jsonFileName, my_dictionary)  
    