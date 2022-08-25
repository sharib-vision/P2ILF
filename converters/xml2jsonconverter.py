#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 09:26:26 2022

@author: sharib

xml to json converter
pip install xmltodict

-- converter for 2D and 3D contours

Please note that you will need to provide separate 2D and 3D contours files during challenge - for this refer to write2D_3DContours_separately.py

- You might have to adapt to your code and you can do shorter way to do this using some functions in these files!
"""

import os
import numpy as np
import json
unicode = str
        
class Decoder2(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)  # result = super(Decoder, self).decode(s) for Python 2.x
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str) or isinstance(o, unicode):
            try:
                return list(map(int, map(float, o.split(','))))
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {k: self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o
        
        
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="formater - xml to json - task 1 @ P2ILF challenge @ MICCAI 2022", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
    parser.add_argument("--xmlfile", type=str, default="../samplexmlprovided/", help="2D and 3D contours")
    parser.add_argument("--jsonfile", type=str, default="../samplexmlprovided/", help="2D and 3D contours")
    # parser.add_argument("--originalImage", type=str, default="04_undistorted.png", help="original 2D image")
    args = parser.parse_args()
    return args

def splitStringtoArray(f):
   
    return  map(int, f.split(','))

if __name__ == "__main__":
    import xmltodict
    import pprint
    import glob
    
    args = get_args()

    directory = args.xmlfile
    jsonfileDir =  args.jsonfile
    #list(splitStringtoArray(A['contours']['contour'][0]['imagePoints']['x']))

    for filename in glob.iglob(f'{directory}/*xml'):
        print(filename)
        filenameJson= filename.split('/')[-1].split('.')[0]+'.json'
        filenameXml = filename

        with open(filenameXml) as fd:
            doc = xmltodict.parse(fd.read())
        
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(json.dumps(doc))
        
        with open(os.path.join(jsonfileDir, filenameJson), 'w', encoding='utf-8') as f:
            json.dump( json.loads(json.dumps(doc), cls=Decoder2), f, ensure_ascii=False, indent=4)
        

    
    