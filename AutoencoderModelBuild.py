import zipfile
import SimpleITK as sitk
import numpy as np
from model2 import build_model  
import glob  
from scipy.ndimage import zoom  
import re  

dataset_path = "/home/isaacsmith/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/"
# print ("Unzipping...")
# zfile = zipfile.ZipFile(dataset_path)
# zfile.extractall()
# print ("Done.")

def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def resize(img, shape, mode='constant', orig_shape=(155, 240, 240)):
    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[0]/orig_shape[0],
        shape[1]/orig_shape[1], 
        shape[2]/orig_shape[2]
    )
    
    return zoom(img, factors, mode=mode)


def preprocess(img, out_shape=None):
    if out_shape is not None:
        img = resize(img, out_shape, mode='constant')
    mean = img.mean()
    std = img.std()
    return (img - mean) / std


def preprocess_label(img, out_shape=None, mode='nearest'):
    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = img == 2  # Peritumoral Edema (ED)
    et = img == 4  # GD-enhancing Tumor (ET)
    
    if out_shape is not None:
        ncr = resize(ncr, out_shape, mode=mode)
        ed = resize(ed, out_shape, mode=mode)
        et = resize(et, out_shape, mode=mode)

    return np.array([ncr, ed, et], dtype=np.uint8)

# Get a list of files for all modalities individually
print ("Collecting all the data for each modality...")
t1 = glob.glob('/home/isaacsmith/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1.nii')
t2 = glob.glob('/home/isaacsmith/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii')
flair = glob.glob('/home/isaacsmith/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii')
t1ce = glob.glob('/home/isaacsmith/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii')
seg = glob.glob('/home/isaacsmith/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii')  # Ground Truth
seg.extend(glob.glob('/home/isaacsmith/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*Segm.nii'))
print ("Sample from t1 files:\n{}\n\n".format(t1[:2]))
print ("Done.")

pat = re.compile('.*_(\w*)\.nii\.gz')

print ("Generating the dictionary of all data paths...")
data_paths = [{
    pat.findall(item)[0]:item
    for item in items
}
for items in list(zip(t1, t2, t1ce, flair, seg))]
print ("Sample:\n{}\n\n".format(data_paths[:2]))
print ("Done.")


# input_shape = (4, 160, 192, 128) # This overflows memor -- TODO
input_shape = (4, 64, 64, 64)
output_channels = 3
data = np.empty((len(data_paths),) + input_shape, dtype=np.float32)
labels = np.empty((len(data_paths), output_channels) + input_shape[1:], dtype=np.uint8)


import math

print ("Building the data and labels arrays...")
for i, imgs in enumerate(data_paths):
    try:
        data[i] = np.array([preprocess(read_img(imgs[m]), input_shape[1:]) for m in ['t1', 't2', 't1ce', 'flair']], dtype=np.float32)
        labels[i] = preprocess_label(read_img(imgs['seg']), input_shape[1:])[None, ...]
        print ("Added data successfully ({}/{})".format(i, len(data_paths)))
    except Exception as e:
        print(f'Something went wrong with {imgs["t1"]}, skipping...\n Exception:\n{str(e)}')
        continue
print ("Done.")

print ("Building model...")
model = build_model(input_shape=input_shape, out_channels=3)
print ("Done.")

print ("Fitting model to data...")
model.fit(data, [labels, data], epochs=300)
print ("Done.")

print ("Writing model...")
model.save('/home/isaacsmith/Models/300epochs.h5', save_format='h5')
print ("Done.")
