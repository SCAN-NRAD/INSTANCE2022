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
    
#%%
import numpy as np
resolutions = np.abs(np.load('instance_resolutions.npy') )
rounded  = np.round(433*abs(resolutions))/433 #mario code
print(f'unique resolutions: {len(np.unique(resolutions,axis=0))}') 
print(f'unique rounded resolutions (mario version): {len(np.unique(rounded,axis=0))}') 
print(f'unique rounded resolutions (3 decimals): {len(np.unique(np.round(resolutions,decimals=1),axis=0))}') 


# %%
