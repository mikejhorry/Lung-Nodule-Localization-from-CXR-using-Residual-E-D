import os
import argparse
import numpy as np
import shutil
import json
import random

# NOTE: To use the default parameters, execute this from the main directory of
# the package.

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--data_dir",
    type=str,
    default="../data/image_dataset",
    help=("Raw data path. Expects 3 or 4 subfolders with classes")
)
ap.add_argument(
    "-m",
    "--mask_dir",
    type=str,
    default="../data/image_dataset",
    help=("Raw data path. Expects 3 or 4 subfolders with classes")
)

ap.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default="/data/mjhorry/Experiments/JSRT Nodule Segmentation/input/nodule_files/JSRT/crossval",
    help=("Output path where images for cross validation will be stored.")
)
ap.add_argument(
    "-s",
    "--splits",
    type=int,
    default=5,
    help="Number of folds for cross validation"
)
args = vars(ap.parse_args())

NUM_FOLDS = args['splits']
DATA_DIR = args['data_dir']
MASK_DIR = args['mask_dir']

OUTPUT_DIR = args['output_dir']

# MAKE DIRECTORIES
for split_ind in range(NUM_FOLDS):
    # make directory for this split
    fold_path = os.path.join(OUTPUT_DIR, 'fold' + str(split_ind))
    fold_images_path = os.path.join(OUTPUT_DIR, 'fold' + str(split_ind),'images')
    fold_masks_path = os.path.join(OUTPUT_DIR, 'fold' + str(split_ind),'masks')
    
    split_path = os.path.join(OUTPUT_DIR, 'split' + str(split_ind))    
    split_images_path = os.path.join(OUTPUT_DIR, 'split' + str(split_ind), 'images')
    split_masks_path = os.path.join(OUTPUT_DIR, 'split' + str(split_ind),'masks')
    predicted_path = os.path.join(OUTPUT_DIR, 'split' + str(split_ind),'predicted')

    if not os.path.exists(split_path):
        os.makedirs(split_path)
    if not os.path.exists(split_images_path):
        os.makedirs(split_images_path)
    if not os.path.exists(split_masks_path):
        os.makedirs(split_masks_path)        
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)        
    if not os.path.exists(fold_images_path):
        os.makedirs(fold_images_path)
    if not os.path.exists(fold_masks_path):
        os.makedirs(fold_masks_path)
    if not os.path.exists(predicted_path):
        os.makedirs(predicted_path)

# MAKE SPLIT
copy_dict = {}
uni_images = []
       
for in_file in os.listdir(DATA_DIR):
	if in_file[0] == ".":
		continue
	else:
		# this is an image
		print(in_file.split(".")[0])
		uni_images.append(in_file.split(".")[0])

# construct dict of file to fold mapping
inner_dict = {}

# consider images
split_counter = 0
for k, uni in enumerate([uni_images]):
	unique_files = np.unique(uni)
	
print(unique_files)
	
	# each file allocate a number 0 to 9 in order - reset after 9
	#inner_dict[f] = i
for f in unique_files:
	inner_dict[f] = split_counter
	split_counter = split_counter + 1
	if(split_counter == NUM_FOLDS):
		split_counter = 0




''' original code
for k, uni in enumerate([uni_images]):
	unique_files = np.unique(uni)
#	random.shuffle(unique_files)
	
	# s is number of files in one split
	s = len(unique_files) // NUM_FOLDS
	for i in range(NUM_FOLDS):
		for f in unique_files[i * s:(i + 1) * s]:
			inner_dict[f] = i

	# distribute the rest randomly
#	for f in unique_files[NUM_FOLDS * s:]:
#		inner_dict[f] = np.random.choice(np.arange(5))
'''		
print(inner_dict)		
				
'''
copy_dict[classe] = inner_dict
'''
for in_file in os.listdir(DATA_DIR):
	fold_to_put = inner_dict[in_file.split(".")[0]]
	split_path = os.path.join(
		OUTPUT_DIR, 'split' + str(fold_to_put),'images'
	)
	mask_path = os.path.join(
		OUTPUT_DIR, 'split' + str(fold_to_put), 'masks'
	)
	predict_path = os.path.join(
		OUTPUT_DIR, 'split' + str(fold_to_put), 'predicted'
	)
	
	print(os.path.join(DATA_DIR, in_file))
	print(split_path)
	shutil.copy(os.path.join(DATA_DIR, in_file), split_path)
	shutil.copy(os.path.join(MASK_DIR, in_file), mask_path)
	shutil.copy(os.path.join(MASK_DIR, in_file), predict_path)
	
		
	# create the fold directories
for split_ind in range(NUM_FOLDS):
	# copy all files except split_ind
	for in_file in os.listdir(DATA_DIR):
	    split = inner_dict[in_file.split(".")[0]]
	    if split != split_ind:
	        split_path = os.path.join(
		        OUTPUT_DIR, 'fold' + str(split_ind),  'images'
	        )
	        mask_path = os.path.join(
		        OUTPUT_DIR, 'fold' + str(split_ind), 'masks'
	        )
	        shutil.copy(os.path.join(DATA_DIR, in_file), split_path)
	        shutil.copy(os.path.join(MASK_DIR, in_file), mask_path)

