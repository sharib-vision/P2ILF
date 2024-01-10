#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 23:37:12 2022

@author: sharib
"""

import numpy as np
import torch
from skimage import morphology
from scipy import ndimage
import sys
    

import matplotlib.pyplot as plt

print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)


def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def IoU(tp,fp,tn,fn):
    if tp +fp + fn == 0:
        return 0
    return tp / (tp+fp+fn)

def IoUClass(confusion_matrix, id_class):
    tp, fp, tn, fn = confusionMatrixClass(confusion_matrix, id_class)
    return IoU(tp, fp, tn, fn)

def sensitivity(tp,fp,tn,fn):
    if tp+fn == 0:
        return 0
    return tp / (tp+fn)

def sensitivityClass(confusion_matrix, id_class):
    tp, fp, tn, fn = confusionMatrixClass(confusion_matrix, id_class)
    return sensitivity(tp, fp, tn, fn)

def precision(tp,fp,tn,fn):
    if tp + fp == 0:
        return 0
    return tp / (tp+fp)

def precisionClass(confusion_matrix, id_class):
    tp, fp, tn, fn = confusionMatrixClass(confusion_matrix, id_class)
    return precision(tp, fp, tn, fn)


# Distance transform
def computeTDT(contour_image, threshold=None,norm=True):
    """
    contour:[h,w] with values in {0,1} 1 means contours
    return tdt in [0,1]
    """
    if threshold is None:
        threshold = 255
    # if contour contain no contour pixel
    if np.sum(contour_image) == 0:
        return threshold * np.ones_like(contour_image,dtype=np.float64)

    # requires contour to value 0
    inversedContour = 1 - contour_image
    dt = ndimage.distance_transform_edt(inversedContour)
    dt = np.float32(dt)


    #apply threshold
    dt[dt > threshold] = threshold
    if norm:
        dt = dt  / threshold
    return dt

def distContour2Dt(contour, dt):
    assert contour.shape == dt.shape

    dist = contour.to(torch.float64) * dt.to(torch.float64)
    return dist



def symDist2(P,G, P_dt=None, G_dt=None, threshold_dt=50, reduction=False, norm='gt'):
    if P_dt is None:
        P_dt = torch.from_numpy(computeTDT(P.numpy(), threshold_dt,norm=False))
    if G_dt is None:
        G_dt = torch.from_numpy(computeTDT(G.numpy(), threshold_dt,norm=False))

    dPG = distContour2Dt(P, G_dt)
    dGP = distContour2Dt(G, P_dt)
    # Normalization by =====> 
    if norm == 'gt':
        N = 1 if G.sum() == 0 else G.sum().item()
    elif norm == 'both':
        N = 1 if G.sum() + P.sum() == 0 else (P.sum() + G.sum()).item()
    elif norm == 'respective':
        # norm according to respective number of elemnts in
        norm_G = 1 if G.sum() == 0 else G.sum().item()
        norm_P = 1 if G.sum() == 0 else P.sum().item()
        distSym = dPG / P + dGP / G
        if reduction:
            return torch.sum(distSym).item()
        return distSym
    elif norm == 'none':
        N=2

    # for gt and both norm cases:
    distSym = (dPG + dGP) / (N*threshold_dt)
    if reduction:
        return torch.sum(distSym).item()
    return distSym

def distanceMatching(A,B,dmax):
    # check if A is not empty
    if A.sum().item() == 0:
        return 0.0, 0.0, 0.0, False
    #compute dt
    B_dt = torch.from_numpy(computeTDT(B.numpy(),dmax,norm=False))

    dAB = distContour2Dt(A, B_dt)
    N_tot = A.sum().item()
    mask_dmax = dAB >= dmax # n non_match
    mask_match_and_zero = dAB < dmax # n match + zero

    n_non_match = float(mask_dmax.sum().item())
    n_match = N_tot - n_non_match
    distance_tot = dAB[mask_match_and_zero].sum().item()

    return distance_tot, n_non_match, n_match, True

def symetricDistanceMatching(A,B,dmax):
    """
    if output = 'all':
        return (D_AB+D_BA) /2, distance_AB, distance_BA, miss_A, miss_B
    else
        return D, miss_A, miss_B
    """
    # get non normed results
    D_AB, miss_A, match_A, A_not_empty = distanceMatching(A,B,dmax)

    D_BA, miss_B, match_B, B_not_empty = distanceMatching(B,A,dmax)

    if (not A_not_empty) and (not B_not_empty): # if both False
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # compute d_AB and d_BA norm:
    if D_AB > 0:
        D_AB_norm = D_AB / (1.0 * match_A)
    else:
        D_AB_norm = 0.0 # = d_AB
    if D_BA:
        D_BA_norm = D_BA / (1.0* match_B)
    else:
        D_BA_norm = 0. # = d_BA

    D = (D_AB + D_BA)/(1.0 * (A_not_empty + B_not_empty)) # denom can't be = 0 cause A+B empty is dealt before

    return D, D_AB_norm, D_BA_norm, miss_A, miss_B, match_A, match_B

def computeSymetricDistanceMatching(input, target, thresh_dt=None):
    if thresh_dt==None:
        thresh_dt = 50

    n_class = 2
    # distance combined normed
    dist_sym = torch.zeros((n_class-1,1), dtype=torch.float64)
    # distance pred to gt normed by n_pred_match
    dist_pred = torch.zeros((n_class-1,1), dtype=torch.float64)
    # distance gt to pred normed by n_gt_match
    dist_gt = torch.zeros((n_class-1,1), dtype=torch.float64)
    miss_pred = torch.zeros((n_class-1,1), dtype=torch.float64)
    miss_gt = torch.zeros((n_class-1,1), dtype=torch.float64)
    match_pred = torch.zeros((n_class-1,1), dtype=torch.float64)
    match_gt = torch.zeros((n_class-1,1), dtype=torch.float64)
    out = symetricDistanceMatching(input, target,dmax=thresh_dt)
    # for i in range(1,n_class): # skip class0
    #     out = symetricDistanceMatching(input[i, :], target[i,:],dmax=thresh_dt)
    i = 1
    dist_sym[i-1,0] = out[0]
    dist_pred[i-1,0] = out[1]
    dist_gt[i-1,0] = out[2]
    miss_pred[i-1,0] = out[3]
    miss_gt[i-1,0] = out[4]
    match_pred[i-1,0] = out[5]
    match_gt[i-1,0] = out[6]
        
    return dist_sym, dist_pred, dist_gt, miss_pred, miss_gt, match_pred, match_gt

def computeScoreDataset(sym_dist_matching, n_miss_gt, n_match_gt, n_outliers, n_pixels, dmax):
    score_match = sym_dist_matching
    n_tot_gt =n_miss_gt + n_match_gt
    score_outliers = dmax * n_outliers / (n_pixels - 2*dmax*n_tot_gt)
    score_gt_miss = dmax * n_miss_gt / n_tot_gt
    score = score_match + score_outliers + score_gt_miss
    return score

def computeScoreFrancois(contours_one_hot, target_one_hot, w_image, h_image, n_class, dmax):
    symetric_distances_matching = torch.zeros((1, n_class-1, 1), dtype=torch.float64)
    distance_matching_pred = torch.zeros((1, n_class-1,1), dtype=torch.float64)
    distance_matching_gt = torch.zeros((1, n_class-1,1), dtype=torch.float64)
    n_miss_gt_all = torch.zeros((1, n_class-1,1), dtype=torch.float64)
    n_miss_pred_all = torch.zeros((1, n_class-1,1), dtype=torch.float64)
    n_match_pred_all= torch.zeros((1, n_class-1,1), dtype=torch.float64)
    n_match_gt_all= torch.zeros((1, n_class-1,1), dtype=torch.float64)
    
    #  compute symetric distance matching
    sym_dist_match, d_pred, d_gt, miss_pred, miss_gt, match_pred, match_gt = computeSymetricDistanceMatching(contours_one_hot.squeeze(), target_one_hot, dmax)
    
    distance_matching_pred[0] += d_pred * match_pred
    distance_matching_gt[0] += d_gt * match_gt
    n_miss_gt_all[0] += miss_gt
    n_miss_pred_all[0] += miss_pred
    n_match_pred_all[0] += match_pred
    n_match_gt_all[0] += match_gt
    
    N_images = 1
    n_pixels = h_image * w_image
    
    # compute symetric chamfer distance normed by 2 x N_gt_match
    symetric_distances_matching = (distance_matching_pred + distance_matching_gt) / (2.0 * n_match_gt_all)
    distance_matching_pred /= n_match_pred_all # norm by all pred inliers pixels
    distance_matching_gt /= n_match_gt_all # norm by all gt matched pixels
    n_miss_gt_all /= N_images # norm by total images
    n_miss_pred_all /= N_images # norm by total images
    n_match_pred_all /= N_images # norm by total images
    n_match_gt_all /= N_images # norm by total images
    full_score = computeScoreDataset(symetric_distances_matching, n_miss_gt_all, n_match_gt_all, n_miss_pred_all, n_pixels, dmax)
    
    return full_score

def computeScoreFrancoisSimplified(contours_one_hot, target_one_hot, w_image, h_image, dmax):
    
    A = contours_one_hot.squeeze()
    B = target_one_hot.squeeze()
    epsilon = 0.0000001
 
    D_AB, miss_A, match_A, A_not_empty = distanceMatching(A,B,dmax)

    D_BA, miss_B, match_B, B_not_empty = distanceMatching(B,A,dmax)

    n_pixel_image = A.shape[0] * A.shape[1]
    n_tot_gt = miss_B+match_B
    D_match = (D_AB + D_BA) / (2.0 * n_tot_gt+ epsilon)
    D_miss_gt = dmax * miss_B / (1.0*n_tot_gt+ epsilon)
    D_outliers = dmax * miss_A / (n_pixel_image - (2*dmax*n_tot_gt)+ epsilon)
    score = D_match + D_miss_gt + D_outliers

    return score


def thinPrediction(contour_input, n_class, area_threshold_hole=5, min_size_elements=5):
    """
    Thin prediction contour:
    Input should be torch Tensor
    """
    if isinstance(contour_input, torch.Tensor):
        contour_input = contour_input.numpy()
    # work on copy
    contour = contour_input.copy()

    if n_class > 1:
        for i in range(1,n_class):

            mask = contour == i
            c = morphology.remove_small_holes(mask, area_threshold=area_threshold_hole)
            c = morphology.remove_small_objects(c, min_size=min_size_elements)
            thin_mask = morphology.skeletonize(c)
            # erase
            contour[mask] = 0
            # draw skeleton
            contour[thin_mask] = i
        return torch.from_numpy(contour)
    

def confusionMatrix(input,target, n_class):
    """
    Compute confusion matrix for tensor torch
    https://en.wikipedia.org/wiki/Confusion_matrix

    input: [H, W]
    target: [H,W]

    return confusion_matrix[n_class, n_class]
    predicted_class 0-axis,
    actual class 1-axis
    """
    # assert input.dim() == 2, "Input is not 2 dim tensor"
    # assert target.dim() == 2,"Target is not 2 dim tensor"
    matrix = torch.zeros((n_class, n_class))

    for i_true in range(n_class):
        target_i = target == i_true
        for j_predict in range(n_class):
            # extract predicted class j
            input_j = input == j_predict

            S = torch.sum(target_i[input_j])
            matrix[j_predict,i_true] = S.item()

    return matrix

def confusionMatrixClass(confusion_matrix, id_class):
    """
    From confusion matrix, return associated postive and negative confusion
    retuns: true_pos, false_pos, true_neg, false_neg
    """
    # actual class that is predicted class (cm[id,id] element)
    true_positive = confusion_matrix[id_class, id_class]
    # actual class - true positive
    false_positive = torch.sum(confusion_matrix[id_class, :]) - true_positive
    # predicted class - true positive
    false_negative = torch.sum(confusion_matrix[:, id_class]) - true_positive
    # rest of confusion matrix
    true_negative = torch.sum(confusion_matrix[:]) - true_positive - false_negative - false_positive
    
    print(true_negative)
    print(false_positive)
    print(false_negative)
    print(true_positive)
    
    print('Precision_implemented:', true_positive/(true_positive+false_positive))
    print('Recall_implemented:', true_positive/(true_positive+false_negative))
    
    DSC = (2*true_positive)/((true_positive+false_positive) + (true_positive+false_negative))
    print('DSC:', DSC)

    return true_positive.item(), false_positive.item(), true_negative.item(), false_negative.item()


def computeClassificationMetrics(input, target, n_class, list_metrics, skip_class0=True):
    '''
    

    Parameters
    ----------
    input : Prediction
        DESCRIPTION.
    target : GT
        DESCRIPTION.
    n_class : TYPE
        DESCRIPTION.
    list_metrics : TYPE
        DESCRIPTION.
    skip_class0 : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    metrics : TYPE
        DESCRIPTION.
    DSC : TYPE
        DESCRIPTION.
    JC : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.

    '''

    from sklearn.metrics import confusion_matrix
    p =0
    # , ConfusionMatrixDisplay
    if (input.flatten()).sum().item()!=0 or (target.flatten()).sum().item()!=0:
        tn, fp, fn, tp  = confusion_matrix((target).flatten(), (input).flatten()).ravel()
        print(tn, fp, fn, tp)
        DSC = (2*tp)/((tp+fp) + (tp+fn))
        print('DSC_Sklearn:', DSC)
            
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        d = 2*p*r/(p+r)
        print('Precision:', tp/(tp+fp))
        print('Recall:', tp/(tp+fn))
        print('DSC:', d)
        
    
    else:
        DSC = 1
        print('DSC_Sklearn:', DSC)
    # cm = confusion_matrix((input).flatten(),  (target).flatten())
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.show()

    
    from sklearn.metrics import jaccard_score
    JC = jaccard_score((target).flatten(), (input).flatten())
    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    print('JC-sklearn:', JC)
    
    # compute confusion matrix:
    confusion_matrix = confusionMatrix(input, target, n_class)
    print(confusion_matrix)
    l_confusion=[]
    for i in range(n_class):
        if skip_class0 and i == 0:
            continue
        l_confusion.append(confusionMatrixClass(confusion_matrix, i))

    metrics = torch.zeros((len(l_confusion), len(list_metrics)),dtype=torch.float64)
    
    for i,confusion in enumerate(l_confusion):
        for j,f in enumerate(list_metrics):

            m = f(*confusion)
            metrics[i,j] = m
            
            # after checking it turns out that the first is recall and next is precision
    return metrics, DSC, JC , p  

def convertToOneHot(T, n_class):
    """
    T is [B,H,W]
    one_hot is [B,C,H,W]
    """
    if T.dim() == 3:
        T_one_hot = torch.nn.functional.one_hot(T.long(), n_class) # new_shape = shape+[n_class]
        target = T_one_hot.permute(0,3,1,2) # [B, N, H, W]

    elif T.dim() == 2:
        T_one_hot = torch.nn.functional.one_hot(T.long(),n_class)
        target = T_one_hot.permute(2,0,1) # [N, H, W]
    return target

def convertGT_toOneHotEncoding(GTimage, nclass): 
    GTimage[GTimage[:,:,:] > 1] = 255
    GTimage[GTimage[:,:,:] <= 1] = 0
    
    GTimageTest=np.zeros(GTimage.shape)
    GTimageTestCollapse=np.zeros((GTimage.shape[0],GTimage.shape[1]))
    # Ligament (B G R)
    #  BGR: convert to background: 0 (0,0,0); class silhouette: 1 (0, 255,255); class ridges: 2 (0,255,0) and class ligament: 3 (0,0,255)
    GTimageTest[:,:,0] = GTimage[:,:,0]
    GTimageTest[:,:,1] = GTimage[:,:,1]
    GTimageTest[:,:,2] = GTimage[:,:,2]-GTimage[:,:,1]
    
    # GTimageTest[GTimageTest[:,:,2]<=200] = 0
    GTimageTest[:,:,2][GTimageTest[:,:,2]==255] = 1 # Ridge
    GTimageTest[:,:,0][GTimageTest[:,:,0]==255] = 2 # Ligament
    GTimageTest[:,:,1][GTimageTest[:,:,1]==255] = 3 # Silhouette

    
    # GTimageTestCollapse = GTimageTest[:,:,0] + GTimageTest[:,:,1] + GTimageTest[:,:,2]
    idx = np.where(GTimageTest[:,:,2]==1)
    GTimageTestCollapse[idx] = 1
    idx = np.where(GTimageTest[:,:,0]==2)
    GTimageTestCollapse[idx] = 2
    idx = np.where(GTimageTest[:,:,1]==3)
    GTimageTestCollapse[idx] = 3
        
    GTimageTestCollapse = torch.tensor(GTimageTestCollapse.astype('uint8'))
    
    T_one_hot_GT = torch.nn.functional.one_hot(GTimageTestCollapse.type(torch.int64), nclass)
    # target = T_one_hot_GT.permute(2,0,1) # [N, H, W]
    return T_one_hot_GT, GTimageTest