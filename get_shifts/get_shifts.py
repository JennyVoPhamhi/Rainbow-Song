from skimage.io import imread, imsave
from skimage import feature, transform
from scipy.ndimage import fourier_shift
import numpy as np
import pandas as pd
import os
from os.path import isdir, isfile, join

def get_dirs(mypath):
    return sorted([f for f in os.listdir(mypath) if isdir(join(mypath, f))])
def get_files(mypath):
    return sorted([f for f in os.listdir(mypath) if isfile(join(mypath, f))])

columns = ['experiment', 'img_set', 'filename', 'y_shift', 'x_shift']
df_result = pd.DataFrame(columns=columns)
experiments = get_dirs('./cci/')

for experiment_ind in range(len(experiments)):
    print('experiment: %s / %s' % (str(experiment_ind), str(len(experiments))))
    experiment = experiments[experiment_ind]
    
    cci_images_path = join('./cci/', experiment, 'CCI Images')
    img_sets = get_dirs(cci_images_path)
        
    for img_set_ind in range(len(img_sets)):
        print('\timg_set: %s / %s' % (str(img_set_ind), str(len(img_sets))))
        img_set = img_sets[img_set_ind]
        
        img_set_path = join(cci_images_path, img_set)
        files = get_files(img_set_path)
        background_imgs = [f for f in files if 'background-' in f]
        if len(background_imgs)==0: continue
        imgs_to_align = [f for f in files if 'image-' in f]
        if len(imgs_to_align)==0: continue
        names = [f[11:-4] for f in background_imgs]
        
        for name_ind in range(len(names)):
            print('\t\tname: %s / %s' % (str(name_ind), str(len(names))))
            name = names[name_ind]
            
            bg = imread(join(img_set_path, 'background-%s.tif' % name))
            imgToAlign = imread(join(img_set_path, 'image-%s.tif' % name))
            shift, error, diffphase = feature.register_translation(bg, imgToAlign)
            y_shift, x_shift = shift
            # alignedImage = fourier_shift(np.fft.fftn(imgToAlign), shift)
            # alignedImage = np.fft.ifftn(alignedImage)
            # alignedImage = alignedImage.real
            # imsave(join(img_set_path, 'aligned-%s.tif' % name), alignedImage.astype(np.uint16))
            row = {'experiment': experiment,
                   'img_set': img_set,
                   'filename': name,
                   'y_shift': y_shift,
                   'x_shift': x_shift,
                  }
            df_result = df_result.append(row, ignore_index=True)
df_result.to_csv('shifts.csv', index=None)