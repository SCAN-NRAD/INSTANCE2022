import glob
import os
import nibabel as nib
import numpy as np

cases = glob.glob('/str/nas/INSTANCE2022/train_2/data/*.nii.gz')
res = np.zeros((len(cases),3))
for i, case in enumerate(cases):
    print(i)
    image = nib.load(case)
    res[i] = np.array(image.affine.diagonal()[:3])

np.save('instance_resolutions.npy',res)
    
    
    
