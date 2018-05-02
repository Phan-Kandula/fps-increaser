import numpy as np
import time
from PIL import Image
import glob
import os


def load_to_arr(file_path):
    '''
    arr = []
    for img in glob.glob(file_path + '/*.png'):
        im = Image.open(img)
        arr.append(im)
    list = np.array([np.array(Image.open(img)) for img in glob.glob(file_path +
        '*.png')])
        '''
    list = np.array([np.array(Image.open(file_path + "frame_" + str(i) + ".png")).flatten() for i in range(500)])
    return list

if __name__ == '__main__':
   start_time = time.time()
   list = load_to_arr(os.getcwd() + '/fs1_images/')
   np.save(os.getcwd() + '/train_new_set1.npy',list)
   print("--- %s seconds ---" % (time.time() - start_time))
