#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 01:00:00 2022

@author: sharib
final metrics on leaderboard - 
"""



def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Leaderboard values listing @ P2ILF challenge @ MICCAI 2022", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--contour_2D_metrics", type=str, default="./output/contour_2D_metricvaleues.json", help="Original 3D model")
    parser.add_argument("--contour_3D_metrics", type=str, default="./output/contours_3D_metricvalues.json", help="Original 3D model")
    parser.add_argument("--RPE_metrics", type=str, default="./output/RPE_metricvalues.json", help="Registration reprojection error ")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import json
    from pathlib import Path
    import os
    import meshio

    import numpy as np
    # from evaluation_docker import load_predictions_json
    from metrics_2DContours_eval import compute2DContours
    
    # ALl outputs will be written as separate job instances
    # /input/"job_pk"/output/2d-liver-contours.json
    # /input/"job_pk"/output/3d-liver-contours.json
    # /input/“job_pk"/output/transformed-3d-liver-model.obj
        
    args = get_args()

    flag3D_2D_Reg = 1
    flag3D_contour = 1
    dockercase = True
    
    metrics_Ridge_pr = []
    metrics_lig_pr = []
    metrics_SL_pr = []
    
    avg_dist = []
    
    dist_3D = []
    dist_3D_Hfd=[]

    cnt = 0
    
    if dockercase:
        fname = "/input/predictions.json"
        with open(Path(fname), "r") as f:
            entries = json.load(f)
        
        if isinstance(entries, float):
            raise TypeError(f"entries of type float for file: {fname}")
    else:
            
        entries =[{"pk": "22d796c5-e4a2-4efb-9be2-4791650d55e8","inputs":{"image":"patient4_3.mha"}}]
        # filename = cases[i].split('.')[0]
    print('entries to prediction json file', entries)
    for e in entries:
        job_pk = e["pk"]
        # job_pk =  "22d796c5-e4a2-4efb-9be2-4791650d55e8"
                 
        
        if dockercase:
            contour_2D_pred = '/input/'+ job_pk+'/output/2d-liver-contours.json'
            contour_3D_pred = '/input/'+ job_pk+'/output/3d-liver-contours.json'
            reg_model = '/input/'+ job_pk+'/output/transformed-3d-liver-model.obj'
            

            textured_mesh_Reg = meshio.read(reg_model)
            if int(textured_mesh_Reg.points[0][0])==0: flag3D_2D_Reg=0
            
            f = open(contour_3D_pred)
            contour_pred_3D = json.load(f)
            if int(contour_pred_3D["numOfContours"])==0: flag3D_contour=0
            
            
            inputs = e["inputs"]
            name = None
            for input in inputs:
                if input["interface"]["slug"] == "laparoscopic-image":
                    name = str(input["image"]["name"])
                    break  # expecting only a single input
            if name is None:
                raise ValueError(f"No filename found for entry: {e}")

            fileName = name
            print(fileName)
        else:
            fileName = e["inputs"]["image"]
            # patientID = n
            print(fileName)
            contour_2D_pred = Path('/home/p2ilf2022/app/pred-v1/2d-liver-contours.json')
            reg_model =  Path('/home/p2ilf2022/app/dummy/transformed-3d-liver-model.obj')
            textured_mesh_Reg = meshio.read(reg_model)
            if int(textured_mesh_Reg.points[0][0])==0: flag3D_2D_Reg=0

            contour_3D_pred = Path('/home/p2ilf2022/app/dummy/3d-liver-contours.json')
            f = open(contour_3D_pred)
            contour_pred_3D = json.load(f)
            if int(contour_pred_3D["numOfContours"])==0: flag3D_contour=0

            flag3D_2D_Reg = 1
            flag3D_contour = 1
            
        print('path to predicted countour 2D:', contour_2D_pred)
        print('path to predicted countour 3D:', contour_3D_pred)
        print('path to transformed 2d-3d reg:', reg_model)  
        print("case for reg: {}, and for 3D pred: {}".format(flag3D_2D_Reg, flag3D_contour))
        '''Get GT data'''
        patientFile = fileName.split('.mha')[0]
        contour_2D_GT = os.path.join('/home/p2ilf2022/app/p2ilf_groundTruth-v1/', patientFile+'_2D-contours.json')
        camera_params = os.path.join('/home/p2ilf2022/app/p2ilf_groundTruth-v1/', patientFile.split('_')[0] +'_acquisition-camera-metadata.json')
        meshFile = os.path.join('/home/p2ilf2022/app/p2ilf_groundTruth-v1/', patientFile.split('_')[0] +'_3D-liver-model.obj')
        contour_3D_pred = os.path.join('/home/p2ilf2022/app/p2ilf_groundTruth-v1/', patientFile +'_3D-contours.json')
        
        
        f = open(contour_2D_pred)
        data_pred = json.load(f)
        
        f = open(contour_2D_GT)
        data_gt = json.load(f)
        
        f = open(camera_params)
        data_params = json.load(f)
        
        metrics_Ridge, metrics_Ligament, metrics_SL, metricsdist_Ridge, metricsdist_Ligament, metricsdist_SL = compute2DContours(data_params, data_gt, data_pred)
        
        # 2D ones
        metrics_Ridge_pr.append(metrics_Ridge[0])
        metrics_lig_pr.append(metrics_Ligament[0])
        metrics_SL_pr.append(metrics_SL[0])
        
        dist = (metricsdist_Ridge+metricsdist_Ligament+metricsdist_SL)/3
        avg_dist.append(dist)
        
        print('vals for 2D contour for metrics_Ridge_pr, metrics_lig_pr,  metrics_SL_pr, and dist: {}, {}, {}, {}'.format(metrics_Ridge_pr,metrics_lig_pr,metrics_SL_pr,dist))
        
        # compute for 3D contours
        if flag3D_contour:
            # from  torch_geometric.io import read_obj
            from contour_metrics_3D import metric3Dcompute
            from metric_reprejectionError import cameraIntrinsicMat, RPE
            contour_3D_GT = os.path.join('/home/p2ilf2022/app/p2ilf_groundTruth-v1/', patientFile+'_3D-contours.json')
            # meshFile =  os.path.join('/home/p2ilf2022/app/p2ilf_groundTruth-v1/',  patientFile.split('_')[0]+'_3D-liver-model.obj')
            

    #       Average distances
            avg_dist_NN, avg_dist_HFD = metric3Dcompute(meshFile, contour_3D_GT, contour_3D_pred)
            dist_3D.append(avg_dist_NN)
            dist_3D_Hfd.append(avg_dist_HFD)

            print('vals for 3D contour for dist_NN and dist_HfD: {}, {}'.format(avg_dist_NN, dist_3D_Hfd))

       

        if flag3D_2D_Reg: 

            contour_2D_GT = os.path.join('/home/p2ilf2022/app/p2ilf_groundTruth-v1/', patientFile+'_2D-contours.json')
            contour_3D_GT = os.path.join('/home/p2ilf2022/app/p2ilf_groundTruth-v1/', patientFile+'_3D-contours.json')
            
            gt_2d_contours = contour_2D_GT
            gt_3d_contours = contour_3D_GT
            K = cameraIntrinsicMat(data_params)
            
            # 2D contours
            f = open(gt_2d_contours)
            contour_gt_2D = json.load(f)
            
            # 3D contour vertices : at which the evaluation will be done
            f = open(gt_3d_contours)
            contour_gt_3D = json.load(f)
            
            hfd = RPE(reg_model, contour_gt_3D, contour_gt_2D , K)

            print('val for 2d-3d reg: {}'.format(hfd))
    
    # 3D accuracy
    if flag3D_contour==0:
        dist_3D = -1
        dist_3D_Hfd = -1
    else:
        dist_3D = np.mean(dist_3D)
        dist_3D_Hfd = np.mean(dist_3D_Hfd)

    
    
    # 32D-3D registration RPE (in pixels)
    if flag3D_2D_Reg==0:
        RPEval =  -1
    else:
        # print(hfd)
        RPEval = np.mean(hfd)
        
    
    precision = 0
    SymDistVal_average = 0
    # try:
    SymDistVal_average = np.mean(avg_dist)
    precision = (np.mean(metrics_Ridge_pr) + np.mean(metrics_lig_pr) + np.mean(metrics_SL_pr))/3
    # except:
    #     print('2D evaluation error')
            
    my_dictionary = {"P2ILF": {"2D_contour":{"av_precision": precision, 'SymDistVal_average': SymDistVal_average}
                , "3D_contour":{"distNN_diff": dist_3D, "distHF":dist_3D_Hfd}, 
                "P2ILF_task2":{"RPE": RPEval},
                     }
                }
    print(my_dictionary)                  
    jsonFileName=os.path.join('/output/', 'metrics' + '.json')
    from misc_eval import EndoCV_misc
    EndoCV_misc.write2json(jsonFileName, my_dictionary)  
    
    
    
    
