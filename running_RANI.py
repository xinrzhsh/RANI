'''
' Run segmentation'
'''
import pdb
import sys
sys.path.insert(0, 'lib/')
import tifffile as tiff

from img_functions import get_file_name, get_image_sizes
from exc_functions import run_models_list
from modelsetting import sparse_net_normalization_71 as nuclear_cnn1
from modelsetting import sparse_net_normalization_71 as nuclear_cnn2
from modelsetting import sparse_net_normalization_51 as nuclear_ft


import pdb # QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQq
import os
import numpy as np
import time
import matlab.engine
eng = matlab.engine.start_matlab()


start = time.clock()


data_location='original_Nuclear_images/'

setname=['ls']

avegNum=5


trained_network_nuclear_directory = "trained_RANI_model/"
nuclear_prefix1 ="181219_CNN1_mean_71_1I181219_71"   #"180525_CNN1_nuclei_p50_51_1I_180525_51"
nuclear_prefix2 ="190204_CNN2_nuclei_71_1P-mean_1I-mean_190203_71" #"180530_CNN2_nuclei_51_1P-mean_1I-mean-normalize_180529_51"
nuclear_prefixFTS ="180525_CNN_FTS_0d5_51_1P_180525"
nuclear_prefixFTB ="180531_CNN_FTB2_0d5_51_1P_180531_51"

receptiveField_CCNN1=70
receptiveField_CCNN2=70
receptiveField_FT=50

#CNN1
list_of_nuclear_weights1 = []
for j in xrange(avegNum):
	nuclear_weights = os.path.join(trained_network_nuclear_directory,  nuclear_prefix1 + '_' + str(j) + ".h5")
	list_of_nuclear_weights1 += [nuclear_weights]
#CNN2

list_of_nuclear_weights2 = []
for j in xrange(avegNum):
	nuclear_weights = os.path.join(trained_network_nuclear_directory,  nuclear_prefix2 + '_' + str(j) + ".h5")
	list_of_nuclear_weights2 += [nuclear_weights]

list_of_nuclear_weightsFTB = []
for j in xrange(avegNum):
	nuclear_weights = os.path.join(trained_network_nuclear_directory,  nuclear_prefixFTB + '_' + str(j) + ".h5")
	list_of_nuclear_weightsFTB += [nuclear_weights]

list_of_nuclear_weightsFTS = []
for j in xrange(avegNum):
	nuclear_weights = os.path.join(trained_network_nuclear_directory,  nuclear_prefixFTS + '_' + str(j) + ".h5")
	list_of_nuclear_weightsFTS += [nuclear_weights]

list_of_nuclear_weightsFT=[list_of_nuclear_weightsFTB]+[list_of_nuclear_weightsFTS]

#pdb.set_trace()# QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ

result_location = os.path.join('result_RANI/')
result1_location = os.path.join('result_RANI/1_CNN/')
result2_location = os.path.join('result_RANI/2_CCNN/')
result3_location = os.path.join('result_RANI/3_RANI/')
if os.path.isdir(result_location):
     pass
else: 
     os.mkdir(result_location)
if os.path.isdir(result1_location):
     pass
else: 
     os.mkdir(result1_location)

if os.path.isdir(result2_location):
     pass
else: 
     os.mkdir(result2_location)
if os.path.isdir(result3_location):
     pass
else: 
     os.mkdir(result3_location)





NumReferImg=0;
for nuclear_patch_names in setname: ###################################################
    allimages=get_file_name(data_location, nuclear_patch_names)
    win_nuclear = receptiveField_CCNN1/2
    NumImage=len(allimages)-NumReferImg*2 #NumReferImg*2          #for CNN1
    counter=0
    for j in xrange(NumImage):
         image_names = allimages[j:j+1+NumReferImg*2] #3D CNN1
         print "Processing image " + str(j+ 1) + " of " + str(NumImage)
         image_size_x, image_size_y = get_image_sizes(data_location, image_names)
         image_size_x /= 2
         image_size_y /= 2
         nuclear_predictions = run_models_list(data_location, image_names, output_location=result1_location, model_name = nuclear_cnn1, 
	    list_model_weights = list_of_nuclear_weights1, image_size_x = image_size_x, image_size_y = image_size_y, win_x = win_nuclear, 
            win_y = win_nuclear, process=True, save=True, nuclear_prefix=nuclear_prefix1, NumReferImg=NumReferImg, Nextcascade=True);
  


    allfiles=get_file_name(result1_location,   nuclear_patch_names)
    win_nuclear = receptiveField_CCNN2/2
    allimages = [i for i in allfiles if 'image_'+setname[0] in i]
    allProbs = [i for i in allfiles if 'f1_'+setname[0] in i]
    NumImage=len(allProbs)-NumReferImg*2
    counter=0
    for j in xrange(NumImage):
         #pdb.set_trace()# QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
         image_names = allProbs[j:j+1+NumReferImg*2] + allimages[j:j+1+NumReferImg*2]# CNN2~ allimages[j:j+1+NumReferImg*2]+
         print "Processing image+Probability " + str(j + 1) + " of " + str(NumImage)
         image_size_x, image_size_y = get_image_sizes(result1_location, image_names)
         image_size_x /= 2
         image_size_y /= 2
         nuclear_predictions = run_models_list(result1_location, image_names, output_location=result2_location, model_name = nuclear_cnn2, 
	    list_model_weights = list_of_nuclear_weights2, image_size_x = image_size_x, image_size_y = image_size_y, win_x = win_nuclear, 
            win_y = win_nuclear, process=True, save=True, nuclear_prefix=nuclear_prefix2, NumReferImg=NumReferImg);
    
    # fine-tuning
    #pdb.set_trace()# QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
    allfiles=get_file_name(result2_location, nuclear_patch_names)
    win_nuclear = receptiveField_FT/2
    allseg = [i for i in allfiles if 'f1_f1_'+setname[0] in i]
    NumImage=len(allseg)-NumReferImg*2
    counter=0
    for j in xrange(NumImage):
         image_names = allseg[j:j+1+NumReferImg*2]
         print "Fine-Tuning  " + str(j + 1) + " of " + str(NumImage)
         image_size_x, image_size_y = get_image_sizes(result2_location, image_names)
         image_size_x /= 2
         image_size_y /= 2
         eng.addpath(r'lib',nargout=0)
         image_sparate=eng.sparate_tou_sig_nuclei(result2_location,image_names)
         n_images=len(image_names)
         images = np.zeros((1,1, n_images, len(image_sparate[0]),len(image_sparate[0][0])), dtype = 'float32')
         
         feature_overlay=np.zeros((1,2,len(image_sparate[0]),len(image_sparate[0][0])),dtype = 'float32')
         for j in xrange(len(image_sparate)):
	        images[0,0,n_images-1,:,:]=image_sparate[j]
                nuclear_predictions= run_models_list(result2_location, image_names,output_location=result3_location, model_name = nuclear_ft, 
	            list_model_weights = list_of_nuclear_weightsFT[j], image_size_x = image_size_x, image_size_y = image_size_y, win_x = win_nuclear, 
                    win_y = win_nuclear, process=True, save=False, NumReferImg=NumReferImg,images=images);   
                
                if j==1:
                   for j1 in xrange(2):

                     images[0,0,n_images-1,:,:]=nuclear_predictions[0,1,0:image_size_x*2,0:image_size_y*2]
                     nuclear_predictions= run_models_list(result2_location, image_names,output_location=result3_location, model_name = nuclear_ft, 
	                list_model_weights = list_of_nuclear_weightsFT[j-1], image_size_x = image_size_x, image_size_y = image_size_y, win_x = win_nuclear, 
                        win_y = win_nuclear, process=True, save=False, NumReferImg=NumReferImg,images=images);   
                     

                for jj in xrange(2): #feature edge, mask, background
                    feature=nuclear_predictions[0,jj,0:image_size_x*2,0:image_size_y*2]
                    feature[feature<=0.3]=0
		    feature_overlay[0,jj,:,:] += feature
               
         feature_overlay[feature_overlay>1]=1

         seg_name = os.path.join(result3_location, 'f' + str(0) + '_' + image_names[0][0:11] +'_FCCNN'+ r'.png')
	 tiff.imsave(seg_name,feature_overlay[0,0,:,:])               
         seg_name = os.path.join(result3_location, 'f' + str(1) + '_' + image_names[0][0:11] +'_FCCNN'+ r'.png')
	 tiff.imsave(seg_name,feature_overlay[0,1,:,:])

end = time.clock()
print('total time of segmentation is: %s Seconds'%(end-start))

