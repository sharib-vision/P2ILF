#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 22:38:49 2022

@author: sharib

2D Countour evaluation (prediction 2D contours versus GT contours)

Input:
    Two json files - prediction and GT 
    
output: 
    json metric file
"""

color_ridge = (0,0,255)
color_ligament = (255,0,0)
color_silhouette =  (0,255,255)

import json
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import torch
from metric_functions import computeScoreFrancois, computeScoreFrancoisSimplified, convertGT_toOneHotEncoding, thinPrediction, precision, sensitivity, computeClassificationMetrics, symDist2
from misc_eval import EndoCV_misc


def precision_new(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)
    
def recall_new(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)



def data_2_dilated_lines(data, imgLabel, thickness):
    for contour in data['contour']:
        contourType = contour['contourType']
        imgPointsX = contour['imagePoints']['x']
        imgPointsY = contour['imagePoints']['y']
        imgPtsXY = [j for j in zip(imgPointsX, imgPointsY)]
        imgPointsXY = np.array(imgPtsXY)
        # Generate image curve from XY points:
        imgPointsXY = imgPointsXY.reshape((-1, 1, 2))
        isClosed = False
 
        # Set contour color
        color = (0,0,0)
        if contourType == 'Ridge':
            color = color_ridge
        elif contourType == 'Ligament':
            color = color_ligament
        elif contourType == 'Silhouette':
            color = color_silhouette
         
        # Line thickness
        imgLabel = cv2.polylines(imgLabel, [imgPointsXY],
                              isClosed, color, thickness)
        
    return imgLabel
        
# def compute2DContours(data_params, data_gt, data_pred):
def compute2DContours  (data_params, data_gt, data_pred, filename, image):
    from metric_functions import convertGT_toOneHotEncoding, thinPrediction, precision, sensitivity, computeClassificationMetrics, symDist2

    import numpy as np
    nclass = 4
    ## Dilate GT data 
    ## Dilate GT data 
    imgLabel_GT = np.zeros((int(data_params['height']), int(data_params['width']),3))
    imgLabel_GT = data_2_dilated_lines(data_gt, imgLabel_GT, 12) # thickness - gives the offset
    T_one_hot_GT, GTimageTest = convertGT_toOneHotEncoding(imgLabel_GT, nclass)
    
    
    imgLabel_eval = np.zeros((int(data_params['height']), int(data_params['width']),3))
    imgLabel_eval = data_2_dilated_lines(data_pred, imgLabel_eval, 12) 
    T_one_hot_eval, evalimageTest = convertGT_toOneHotEncoding(imgLabel_eval, nclass)
    evalimageTest[evalimageTest[:,:,:] > 1] = 255
   
    binary_edges = evalimageTest 
    # > 0
    # i/p: contour_input, n_class, area_threshold_hole=5, min_size_elements=5)
    b1 = thinPrediction(binary_edges[:,:,0], 3, 5, 5) # L
    b2 = thinPrediction(binary_edges[:,:,1], 3, 5, 5) # S
    b3 = thinPrediction(binary_edges[:,:,2], 3, 5, 5) # R
    
    
    list_metrics = [precision, sensitivity] 
    # Note some how the results are flipped - so in the output with this gives the first value as recall and second value as precision
    # list_metrics = [sensitivity, precision]
  
    # i/p: computeClassificationMetrics(input, target, n_class, list_metrics,skip_class0=True):
    metrics_Ligament, DSC_L, JC_L, pl = computeClassificationMetrics((b1>0), T_one_hot_GT[:,:,2]>0, 2 , list_metrics, skip_class0=True)
    metrics_Ligament= metrics_Ligament[0].detach().cpu().numpy()
    
    metrics_SL, DSC_SL, JC_SL, psl = computeClassificationMetrics(b2>0, T_one_hot_GT[:,:,3]>0, 2 , list_metrics, skip_class0=True)
    metrics_SL= metrics_SL[0].detach().cpu().numpy()
    
    metrics_Ridge, DSC_R, JC_R, pr= computeClassificationMetrics(b3>0, T_one_hot_GT[:,:,1]>0, 2 , list_metrics, skip_class0=True)
    metrics_Ridge= metrics_Ridge[0].detach().cpu().numpy()
    
    metricssym_Ligament = symDist2((b1 > 0).type(torch.float32), T_one_hot_GT[:,:,2], reduction=True)
    metricssym_SL = symDist2((b2 > 0).type(torch.float32), T_one_hot_GT[:,:,3], reduction=True)
    metricssym_Ridge = symDist2((b3 > 0).type(torch.float32),T_one_hot_GT[:,:,1], reduction=True)
      
    # Compute full score according to François et al (2022):
    # thresh_dt = 10
    # metricsdist_Ligament = computeScoreFrancois(T_one_hot_eval[:,:,2], T_one_hot_GT[:,:,2], int(data_params['width']), int(data_params['height']), 2, thresh_dt)
    # metricsdist_SL = computeScoreFrancois(T_one_hot_eval[:,:,3], T_one_hot_GT[:,:,3], int(data_params['width']), int(data_params['height']), 2, thresh_dt)
    # metricsdist_Ridge = computeScoreFrancois(T_one_hot_eval[:,:,1], T_one_hot_GT[:,:,1], int(data_params['width']), int(data_params['height']), 2, thresh_dt)
    
    # Compute full score according to François et al (2022):
    thresh_dt = 10
    # metricsdist_Ligament = computeScoreFrancois(T_one_hot_eval[:,:,2], T_one_hot_GT[:,:,2], int(data_params['width']), int(data_params['height']), 2, thresh_dt)
    # metricsdist_SL = computeScoreFrancois(T_one_hot_eval[:,:,3], T_one_hot_GT[:,:,3], int(data_params['width']), int(data_params['height']), 2, thresh_dt)
    # metricsdist_Ridge = computeScoreFrancois(T_one_hot_eval[:,:,1], T_one_hot_GT[:,:,1], int(data_params['width']), int(data_params['height']), 2, thresh_dt)
    
    
    metricsdist_Ligament = computeScoreFrancoisSimplified(T_one_hot_eval[:,:,2], T_one_hot_GT[:,:,2], int(data_params['width']), int(data_params['height']),thresh_dt)
    metricsdist_SL = computeScoreFrancoisSimplified(T_one_hot_eval[:,:,3], T_one_hot_GT[:,:,3], int(data_params['width']), int(data_params['height']), thresh_dt)
    metricsdist_Ridge = computeScoreFrancoisSimplified(T_one_hot_eval[:,:,1], T_one_hot_GT[:,:,1], int(data_params['width']), int(data_params['height']), thresh_dt)
    
    # metricssym_Ligament = 0
    # metricssym_SL=0
    # metricssym_Ridge = 0
    
    return [metrics_Ridge, DSC_R, JC_R, pr], [metrics_Ligament, DSC_L, JC_L, pl], [metrics_SL, DSC_SL, JC_SL, psl], metricsdist_Ridge, metricsdist_Ligament, metricsdist_SL, metricssym_Ligament, metricssym_SL, metricssym_Ridge
    # return metrics_Ridge, metrics_Ligament, metrics_SL, metricsdist_Ridge, metricsdist_Ligament, metricsdist_SL

    
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="segmentation metrics - 2D", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--contours_2D_pred", type=str, default="../../../final-test2022/output_BHL/test1_sample_1/2d-liver-contours.json", help="combined contours")
    parser.add_argument("--contours_2D_gt", type=str, default="/Users/scssali/Dataset/p2ilf_final_evaluation_ALI/P2ILF_test_leaderboard/GroundTruth/patient4_11_2D-contours.json", help="converter json 2D and 3D - separated files")
    parser.add_argument("--cameraparameters", type=str, default="../../P2ILF_test_leaderboard/input/test1_sample_1/acquisition-camera-metadata.json", help="for extracting height and width")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    from misc_eval import EndoCV_misc
    args = get_args()
    nclass = 4
    # Get predicted and GT data
    pred_2d_contours = args.contours_2D_pred
    gt_2d_contours = args.contours_2D_gt
    camera_params = args.cameraparameters
    
    fileName =  args.contours_2D_gt.split('/')[-1].split('.')[0]
    
    # inputs
    f = open(pred_2d_contours)
    data_pred = json.load(f)
    
    f = open(gt_2d_contours)
    data_gt = json.load(f)
    
    f = open(camera_params)
    data_params = json.load(f)
    
    ## Dilate GT data 
    imgLabel_GT = np.zeros((int(data_params['height']), int(data_params['width']),3))
    imgLabel_GT = data_2_dilated_lines(data_gt, imgLabel_GT, 12) # thickness - gives the offset
    T_one_hot_GT, GTimageTest = convertGT_toOneHotEncoding(imgLabel_GT, nclass)
    
    
    imgLabel_eval = np.zeros((int(data_params['height']), int(data_params['width']),3))
    imgLabel_eval = data_2_dilated_lines(data_pred, imgLabel_eval, 12) 
    T_one_hot_eval, evalimageTest = convertGT_toOneHotEncoding(imgLabel_eval, nclass)
    evalimageTest[evalimageTest[:,:,:] > 1] = 255
   
    binary_edges = evalimageTest 
    # > 0
    # i/p: contour_input, n_class, area_threshold_hole=5, min_size_elements=5)
    b1 = thinPrediction(binary_edges[:,:,0], 3, 5, 5) # L
    b2 = thinPrediction(binary_edges[:,:,1], 3, 5, 5) # S
    b3 = thinPrediction(binary_edges[:,:,2], 3, 5, 5) # R
    
    
    list_metrics = [precision, sensitivity] 
    # Note some how the results are flipped - so in the output with this gives the first value as recall and second value as precision
    # list_metrics = [sensitivity, precision]

    # i/p: computeClassificationMetrics(input, target, n_class, list_metrics,skip_class0=True):
    metrics_Ligament, DSC_L, JC_L, pl = computeClassificationMetrics((b1>0), T_one_hot_GT[:,:,2]>0, 2 , list_metrics, skip_class0=True)
    metrics_Ligament= metrics_Ligament[0].detach().cpu().numpy()
    
    metrics_SL, DSC_SL, JC_SL, psl = computeClassificationMetrics(b2>0, T_one_hot_GT[:,:,3]>0, 2 , list_metrics, skip_class0=True)
    metrics_SL= metrics_SL[0].detach().cpu().numpy()
    
    metrics_Ridge, DSC_R, JC_R, pr= computeClassificationMetrics(b3>0, T_one_hot_GT[:,:,1]>0, 2 , list_metrics, skip_class0=True)
    metrics_Ridge= metrics_Ridge[0].detach().cpu().numpy()
    
    metricssym_Ligament = symDist2((b1 > 0).type(torch.float32), T_one_hot_GT[:,:,2], reduction=True)
    metricssym_SL = symDist2((b2 > 0).type(torch.float32), T_one_hot_GT[:,:,3], reduction=True)
    metricssym_Ridge = symDist2((b3 > 0).type(torch.float32),T_one_hot_GT[:,:,1], reduction=True)
    
    # Compute full score according to François et al (2022):
    thresh_dt = 10
    # metricsdist_Ligament = computeScoreFrancois(T_one_hot_eval[:,:,2], T_one_hot_GT[:,:,2], int(data_params['width']), int(data_params['height']), 2, thresh_dt)
    # metricsdist_SL = computeScoreFrancois(T_one_hot_eval[:,:,3], T_one_hot_GT[:,:,3], int(data_params['width']), int(data_params['height']), 2, thresh_dt)
    # metricsdist_Ridge = computeScoreFrancois(T_one_hot_eval[:,:,1], T_one_hot_GT[:,:,1], int(data_params['width']), int(data_params['height']), 2, thresh_dt)
    
    
    metricsdist_Ligament = computeScoreFrancoisSimplified(T_one_hot_eval[:,:,2], T_one_hot_GT[:,:,2], int(data_params['width']), int(data_params['height']),thresh_dt)
    metricsdist_SL = computeScoreFrancoisSimplified(T_one_hot_eval[:,:,3], T_one_hot_GT[:,:,3], int(data_params['width']), int(data_params['height']), thresh_dt)
    metricsdist_Ridge = computeScoreFrancoisSimplified(T_one_hot_eval[:,:,1], T_one_hot_GT[:,:,1], int(data_params['width']), int(data_params['height']), thresh_dt)
    
    my_dictionary = {"P2ILF_2D_contours":{"Classification":{"Ridge": [metrics_Ridge[0], metrics_Ridge[1], DSC_R, JC_R],
                "Ligament":  [metrics_Ligament[0], metrics_Ligament[1], DSC_L, JC_L],
                   "SL": [metrics_SL[0], metrics_SL[1], DSC_SL, JC_SL]
                   },
                                          "symDiff": {
                                              "Ridge": metricssym_Ridge, "Ligament": metricssym_Ligament, "SL": metricssym_SL},
                                          "symDist": {
                                              "Ridge": metricsdist_Ridge, "Ligament": metricsdist_Ligament, "SL": metricsdist_SL}
                    },
                }  
    
                   
    jsonFileName=os.path.join('./output/',  fileName + '_metricvalues' + '.json')
    
    EndoCV_misc.write2json(jsonFileName, my_dictionary)    
             
    
    
    # # This is only for checking visually!!!
    # pathOutput = './images_labels/'
    # if not os.path.exists(pathOutput):
    #     os.mkdir(pathOutput)
    # imageNameSplit = fileName+'.jpg'
    # imagePathOutput = pathOutput + imageNameSplit
    # cv2.imwrite(imagePathOutput, imgLabel_GT)
    
    # imageNameSplit = fileName+'_eval.jpg'
    # imagePathOutput = pathOutput + imageNameSplit
    # cv2.imwrite(imagePathOutput, imgLabel_eval)
    # print('Saving image ' + imagePathOutput)

    
    
    