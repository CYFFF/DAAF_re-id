import glob
import os
from random import shuffle
import random

def write_file_list(image_list, filename):
    with open(filename, 'w') as output_file:
        for line in image_list:
            output_file.write(line + '\n')

def image_labels_in_folder(folder_path, parent_folder):
    identity_images = []
   
    for filename in glob.glob(folder_path + '/*.jpg'):
        basename = os.path.basename(filename)
        identity = basename[:4]
        identity_images.append(identity + "," + parent_folder + '/' + basename[:-4])

    return identity_images

def main():
    train_list = image_labels_in_folder('/data1/chenyf/Market-1501-keypoints', 'Market-1501-keypoints')
    write_file_list(train_list, 'market_train.csv')
	
if __name__=='__main__':
	main()
