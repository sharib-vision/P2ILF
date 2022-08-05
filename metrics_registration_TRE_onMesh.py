#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 20:45:54 2022

@author: sharib
"""

# import open3d as o3d
import os
import numpy as np
from  torch_geometric.io import read_obj


import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="RFA: semantic segmentation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--groundtruthMesh", type=str, default="meshGT.obj", help="supply .obj GT mesh")
    parser.add_argument("--deformedMesh", type=str, default="originalMesh_views12345678_ct1_wrp1_arap0_alterscheme3.obj", help="supply deformed mesh (    originalMesh_views12345678_ct1_wrp0_arap0_alterscheme3.obj; originalMesh_views12345678_ct1_wrp1_arap0_alterscheme3)")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    
    registration_path = './registration_example/registrations/seq1/view0/'
    
    args = get_args()
    
    '''
         Ground truth mesh 
    '''
    textured_mesh_GT  = read_obj(os.path.join(registration_path, args.groundtruthMesh))
    print(textured_mesh_GT) 

    # print(np.asarray(textured_mesh_GT.vertices))
    
    '''
        Registered 3D mesh
    '''
    evaluation_path = os.path.join(registration_path, args.deformedMesh)
    registeredNodesContours = read_obj(evaluation_path)
    print(registeredNodesContours)
    

    
    '''
        Compute TRE using all 3D mesh vertices
    '''
    vertices_GT = np.asarray(textured_mesh_GT.pos.detach().cpu().numpy())
    vertices_Registered = np.asarray(registeredNodesContours.pos.detach().cpu().numpy())
    val_dist_pervertices = np.sqrt(((vertices_GT.T - vertices_Registered.T)**2).sum(axis = 0))
    
    meanTRE = np.mean(val_dist_pervertices)
    stdTRE =  np.std(val_dist_pervertices)
    
    print('mean TRE: {} +/- {} mm'.format(meanTRE,stdTRE))
    