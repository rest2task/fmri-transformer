from monai.transforms import LoadImage
import torch
import os
import time
from multiprocessing import Process, Queue
import nibabel as nib
import glob

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def read_data(subj_name, load_root, save_root, scaling_method='minmax', fill_zeroback=True):
    print(f'### Processing: {subj_name}, scaling_method={scaling_method}', flush=True)
    path = os.path.join(load_root, subj_name, 'rfMRI_REST1_LR_hp2000_clean.nii.gz')
    try:
        # load each nifti file
        data, meta = LoadImage()(path)
    except:
        print(f'{subj_name} read data fails.')
        return None
    
    print(subj_name, data.shape, end='')
    
    #change this line according to your file names
    save_dir = os.path.join(save_root, subj_name)
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
    
    # change this line according to your dataset
    data = data[:, 14:-7, :, :]
    # width, height, depth, time
    # Inspect the fMRI file first using your visualization tool. 
    # Limit the ranges of width, height, and depth to be under 96. Crop the background, not the brain regions. 
    # Each dimension of fMRI registered to MNI space (2mm) is expected to be around 100.
    # You can do this when you load each volume at the Dataset class, including padding backgrounds to fill dimensions under 96.

    background = (data==0)
    
    if scaling_method == 'z-norm':
        global_mean = data[~background].mean()
        global_std = data[~background].std()
        data_temp = (data - global_mean) / global_std
    elif scaling_method == 'minmax':
        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())
    

    data_global = torch.empty(data.shape)
    data_global[background] = data_temp[~background].min() if not fill_zeroback else 0 
    # data_temp[~background].min() is expected to be 0 for scaling_method == 'minmax', and minimum z-value for scaling_method == 'z-norm'
    data_global[~background] = data_temp[~background]

    # save volumes one-by-one in fp16 format.
    data_global = data_global.type(torch.float16)
    print(' ->', data_global.shape)
    data_global_split = torch.split(data_global, 1, 3)
    for i, TR in enumerate(data_global_split):
        torch.save(TR.clone(), os.path.join(save_dir, f"frame_{i}.pt"))


def read_data_v1(path, save_dir):
    print("processing: " + path, flush=True)
    fmri_path = os.path.join(path, 'rfMRI_REST1_LR_hp2000_clean.nii.gz') 
    data, metadata = LoadImage()(fmri_path) #torch.Tensor(nib.load(path).get_fdata()) #LoadImage()(path)
    
    #change this line according to your file names
    os.makedirs(save_dir, exist_ok=True)
    
    # change this line according to your dataset
    data = data[7:85, 7:103, 0:84, :]
    # width, height, depth, time
    # Inspect the fMRI file first using your visualization tool. 
    # Limit the ranges of width, height, and depth to be under 96. Crop the background, not the brain regions. 
    # Each dimension of fMRI registered to MNI space (2mm) is expected to be around 100.
    # You can do this when you load each volume at the Dataset class, including padding backgrounds to fill dimensions under 96.
   
    background = (data <=0) # change this because filtered UKB data has minus values
    
    valid_voxels = data[~background].numel()
    global_mean = data[~background].mean()
    global_std = data[~background].std()
    global_max = data[~background].max()
    # global min should be zero

    data[background] = 0

    # save volumes one-by-one in fp16 format.
    data_global = data.type(torch.float16)
    data_global_split = torch.split(data_global, 1, 3)
    for i, TR in enumerate(data_global_split):
        torch.save(TR.clone(), os.path.join(save_dir, "frame_" + str(i) + ".pt"))
    
    # save global stat of fMRI volumes
    checkpoint = {
    'valid_voxels': valid_voxels,
    'global_mean': global_mean,
    'global_std': global_std,
    'global_max': global_max
    }
    torch.save(checkpoint, os.path.join(save_dir,"global_stats.pt"))

if __name__=='__main__':
    start_time = time.time()

    load_root = './HCP_1200/' # This folder should have fMRI files in nifti format with subject names.
    save_root = './HCP_SwiFT/'
    scaling_method = 'minmax' # choose either 'z-norm'(default) or 'minmax'.

    # make result folders
    filenames = os.listdir(load_root)
    os.makedirs(os.path.join(save_root, 'img'), exist_ok = True)
    os.makedirs(os.path.join(save_root, 'metadata'), exist_ok = True) # locate your metadata file at this folder 
    save_img_root = os.path.join(save_root, 'img')
    
    finished_samples = os.listdir(save_root)
    queue = Queue() 
    count = 0
    for filename in sorted(filenames):
        subj_name = filename
        # extract subject name from nifti file. [:-7] rules out '.nii.gz'
        # we recommend you use subj_name that aligns with the subject key in a metadata file.

        expected_seq_length = 1200 # Specify the expected sequence length of fMRI for the case your preprocessing stopped unexpectedly and you try to resume the preprocessing.
        
        # change the line below according to your folder structure
        curr_len = len(glob.glob(os.path.join(save_img_root, subj_name, '*.pt')))
        if curr_len < expected_seq_length: # preprocess if the subject folder does not exist, or the number of pth files is lower than expected sequence length. 
            try:
                count += 1
                '''
                p = Process(target=read_data, args=(subj_name, load_root, save_img_root, scaling_method))
                p.start()
                if count % 1 == 0:
                    p.join()
                '''
                read_data(subj_name, load_root, save_img_root, scaling_method)
            except Exception:
                print('Encountered problem with ' + filename)
                print(Exception)
        else:
            print(f'Skip: {subj_name}, len={curr_len}')

    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')    
