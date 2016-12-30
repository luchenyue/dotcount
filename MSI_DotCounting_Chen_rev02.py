# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:05:19 2016

@author: lindanieman
"""
#%% ----- LOAD MODULES -----
import os
import fnmatch 
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import io, filters, exposure, measure
from skimage.morphology import disk, watershed, remove_small_objects, binary_dilation, binary_erosion, binary_closing
from skimage.segmentation import clear_border
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.color import label2rgb


#%% 
def get_path():  
    """ 
    INPUT:  Ask user for location of image folder to be processed
    OUTPUT: input_path
    
    """  
    print('Please enter full path of image folder')
    
    input_path = raw_input('>> ')
    
    if not os.path.isdir(input_path):
        print('This is not a directory')
        sys.exit(0)
    return input_path
    
#%% 
def get_No_dots():  
    """ 
    INPUT:  Ask user for the number of types of dots they are counting
    OUTPUT: number of types of dots they are counting
    
    """  
    print('Please enter number of types of dots they are counting')
    
    input_No_dots = raw_input('>> ')
    
    if input_No_dots == '1':
        print('Ok we will count only one type')
    elif input_No_dots == '2':
            print('Ok we will count two types')
    else:
        print ('We can only count 1 or 2 types of dots')
        sys.exit(0)
    return input_No_dots

#%%
def get_files(input_folder):
    """
    Create list of _component_data.tif files within folder
    Folder should only contain image files to be processed 
    Also check if Processed folder already exists to prevent overwriting
    
    INPUT:  file path of input folder
    OUTPUT: list of files within input folder 
    
    """
    filelist = os.listdir(input_folder)       
    if len(filelist) == 0:
        print('There are no files in this folder!')
        sys.exit(0)
    if os.path.isdir(input_folder + '/Processed'):
        print('the Processed folder already exists.')
        print('Are you sure you want to overwrite it?')
        print('Please enter "y" or "n"')
        overwrite = raw_input('>> ')
        if overwrite == 'y':
            print ('will overwrite the Processed folder')
        elif overwrite == 'n':
            sys.exit(0)
        else:
            print('That is not a valid answer')
            sys.exit(0)
    filelist = fnmatch.filter(filelist, '*_component_data.tif')        
    return filelist
    
    #%%
def get_objectTables(input_folder):
    """
    Create list of _ObjectTable.csv files within folder
    Folder should only contain image files to be processed 
    
    INPUT:  file path of input folder
    OUTPUT: objective tables of files within input folder 
    
    """
    filelist = os.listdir(input_folder + '/Processed')   
    if len(filelist) == 0:
        print('There are no files in this folder!')
        sys.exit(0)
    
    filelist = fnmatch.filter(filelist, '*_ObjectTable.csv')        
    return filelist
    
#%% 
def get_minThreshold():  
    """ 
    INPUT:  Ask user for minimum intensity threshold to be applied to dots
    OUTPUT: min_thresh
    
    """  
    print('Please enter minimum intensity value for dots')
    print('default for raw counts is 1000')
    print('default for normalized counts is 1050')
    
    min_thresh = raw_input('>> ')
    
    try:
        val = int(min_thresh)
    except ValueError:
        print('That is not an integer!')
        sys.exit(0)
    return  int(min_thresh)
    
    #%% 
def get_minSize():  
    """ 
    INPUT:  Ask user for minimum size threshold to be applied to dots
    OUTPUT: min_blob_size
    
    """  
    print('Please enter minimum size for dots (default is 5)')
    
    min_size = raw_input('>> ')
    
    try:
        val = int(min_size)
    except ValueError:
        print('That is not an integer!')
        sys.exit(0)
    return  int(min_size)
    
#%% 
def get_units():  
    """ 
    INPUT:  Ask user if MSI images are raw counts or normalized by exposure
    OUTPUT: img_units
    
    """  
    print('Are these MSI images values raw counts (y)')
    print('or normalized counts (n)?')
    
    img_units = raw_input('>> ')
    return img_units
    
#%% 
def get_nDotChannels():  
    """ 
    INPUT:  Ask user how many dots channels are present
    OUTPUT: ndot_channels
    
    """  
    print('How many dot channels are present (1 or 2)?')
    
    ndot_channels = raw_input('>> ')
    
    try:
        val = int(ndot_channels)
    except ValueError:
        print('That is not an integer!')
        sys.exit(0)
    return  int(ndot_channels)
    
    #%% 
def get_SpectralComponents():  
    """ 
    INPUT:  Ask user how many unmixed spectral components are present
    OUTPUT: nComponents
    
    """  
    print('How many spectral components are present?')
    print('(Current default is 6)')
    
    nComponents = raw_input('>> ')
    
    try:
        val = int(nComponents)
    except ValueError:
        print('That is not an integer!')
        sys.exit(0)
    return  int(nComponents)
    
    
#%%
def subtract_background(img, nComponents):
    """
    1. Blur dots image using mean filter
    2. Subtract blurred image from original
    
    INPUT:  Original noisy image of dots
    OUTPUT: Clean image that has background subtracted
    
    """
    #1. Blur image
    #--> to use mean filter, must convert image to type uint16 (0-65535)
    #--> first need to rescale image intensity so don't lose decimal place values
    n = nComponents #6 # number of MSI components
    max_int_12bit = 4096/nComponents # 12 bit unmixed components
    f = 65535/max_int_12bit # multiplication factor
    img_rescaled = img*f # rescale intensity fo dots image to fill 16bit range
    img_uint16 = img_rescaled.round().astype(np.uint16)# convert to uint16
    
    # mean filter with disk of xx pixels
    img_blur = filters.rank.mean(img_uint16,disk(2))
    
    # 2. subtract blurred image from original
    # --> change datatype of images so can accomodate negative numbers after subtraction
    img_clean = img_uint16.astype(np.int16)-img_blur.astype(np.int16)
    # --> replace negative numbers by zero
    img_clean = np.clip(img_clean,0,np.max(img_clean))
        
    return img_clean
    
#%%
def subtract_background2(img):
    """
    1. Blur dots image using mean filter
    2. Subtract blurred image from original
    
    INPUT:  Original noisy image of dots
    OUTPUT: Clean image that has background subtracted
    
    """
    #1. Blur image
    blurred = ndi.uniform_filter(img, size=2)
    
    # 2. subtract blurred image from original
    img_clean = img - blurred
    
    # --> replace negative numbers by zero
    img_clean = np.clip(img_clean,0,np.max(img_clean))        
    return img_clean
    
#%% 
def make_dapi_mask(dapi):
    """
    Create dapi mask for downstream processing
    
    INPUT:  Dapi image
    OUTPUT: Dapi mask
    
    """
    # ----- DAPI MASK ----- 
    # only want to count dots that are co-located with nucleus
    # create dapi mask 
    threshold  = filters.threshold_li(dapi) 
    mask = dapi > threshold
    
    # erode nuclear area just a tad to remove outlines
    mask = binary_erosion(mask,disk(2))
    
    return mask
    
    
#%% 
def process_images(input_path, files, min_thresh, min_blob_size, img_units, nComponents):
    """
    For 20x MSI images of gamma radiation DNA breaks
    1. Subtract background
    2. Apply mask to dapi areas
    3. Apply mask to dots areas
    4. Watershed
    5. Enumerate and measure
    
    INPUT:  file path of folder containing MSI images
    OUTPUT: 1. Segmented image with count overlay
            2. Text file with measurement and enumeration info 
            3. Text file with object table information for all images in folder
    """
        
    for file in files:
        df = pd.DataFrame()
        
        # initialize dataframe
        cols = ['Image_ID','Cell_ID','Blob_ID','Area','MeanIntensity',
                'Diameter','MajorAxis','MinorAxis','Eccentricity',
                'Nuclear_Intensity', 'Cytoplasm_Intensity', 'Nuclear_Area']               
        df = pd.DataFrame(columns=cols)


        # initialize variables
        count=0
        image_ID = []
        cell_ID = []
        blob_ID = []
        area = []
        meanInt = []
        diameter = []
        majorAxis = []
        minorAxis = []
        eccen = []
        
        if file.endswith('.tif'):

            img = io.imread(input_path + '/' + file)    
            print('...processing:  ' + file)
            
            # check indexing of image is (channel,x,y). If not, transpose img
            img_shape = img.shape
            
            if img_shape[0] > img_shape[2]:
                img = np.transpose(img,(2,0,1))
            
            # check whether it is raw counts or normalized counts
            scaling_factor = 1       
            if img_units.lower() == 'n':
                scaling_factor = 5000
                print('applying scaling factor = 5000')
                
            dapi = img[0,:,:] * scaling_factor# nuclear marker
            ctc  = img[1,:,:] * scaling_factor# ctc marker.  typically CK and/or EpCAM           
            
            # ----- GET DAPI MASK -----
            mask = make_dapi_mask(dapi)
            # ----- exclude small thresholded objects that may be in nuclear mask ------
            # identify number of cells by the dapi mask 
            nuc_objects, nuc_nlabels = ndi.label(mask)
            # get the sizes of different objects, including background 
            sizes = np.bincount(nuc_objects.ravel())
            # Here we set the dapi mask size limit to be 50
            mask_sizes = sizes > 50
            # remove largest background object (Chen's note: here we are assuming background is the largest object?)
            mask_sizes[0] = 0 
            # apply the revised dapi mask
            mask_nuc = mask_sizes[nuc_objects]

 
            # label each nucleus in image 
            nuc_label_img = measure.label(mask_nuc)  
            
           
            # remove objects touching edge of image
            nuc_label_img = clear_border(nuc_label_img, buffer_size=3)
            
# 
            # get size of each nucleus
            nuc_labels, nuc_sizes = np.unique(nuc_label_img, return_counts=True)
                      
            # nuc_labels returns labels, including the background
            # object, which has a value of zero           
            nuc_labels = nuc_labels.tolist()
            # remove background as an object 
            nuc_labels.remove(0) 

            
            # ------ GET CYTOPLASM AND DOT INFO FOR EACH CELL IN IMAGE -----  

            # check if there are any cells in image
            if len(nuc_labels) < 1 :
                print('----- No cells found in image -----')
                continue
            
            ## ----- FOR KATE -----
            if No_dots == '1':
                dots = img[2,:,:] * scaling_factor# dot marker
            
                 # ----- CLEAN UP DOTS -----
                dots_clean = subtract_background(dots, nComponents)
            
             
    #            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15,6))
    #            ax.imshow(dapi)
    #            plt.close(fig)

            
#            
                
                # loop over number of cells in image
                for i in range(len(nuc_labels)):   
                    cell_count = i + 1
                    print 'Eligible Cell ', cell_count      
    
                    label_mask = nuc_label_img.copy()
                    cell_label = list(nuc_labels)
                    dots_clean_cell = dots_clean.copy()
                    
                                 
                    # if there are more than one cell in image,
                    # define nuclear mask for each cell
                    if len(nuc_labels) > 1 :
                        # remove object of interest from label list  (Chen's note: I don't understand why)
                        del cell_label[i]
                           
                        # set all object labels (except object of interest) to zero (Chen's note: I don't understand why)
                        for j in range(len(cell_label)):
                            label_mask[label_mask==cell_label[j]] = 0
                    
                    # set object of interest label to 1
                    label_mask[label_mask==nuc_labels[i]] = 1
                    # get the size(s) of the nucleus/nuclei
                    nuc_size = nuc_sizes[cell_count]
       
                    
                    # ----- DEFINE CYTOPLASM REGION FOR EACH NUCLEUS -----    (Chen's note: Does this make sense? can try subtracting nucleus area from cell area)
                    mask_nuc_big = binary_dilation(label_mask,disk(20))
                    mask_cyto = mask_nuc_big * ~mask_nuc
           
                    
                    # get mean background cytoplasm counts
                    tmp = dots_clean_cell * mask_cyto
                    cyto_counts = np.mean(tmp).astype('float32')
                    #cyto_counts = np.mean(tmp.nonzero()).astype('float32')
                    print('cyto_count = ' + str(cyto_counts))
                    
                    if np.isnan(cyto_counts):
                        print('cyto_counts = nan')
                        continue
              
                    # ----- REQUIRE DOTS BE CO-LOCATED WITH NUCLEUS -----
                    # limit dots image to nuclear area
                    dots_clean_cell = dots_clean_cell * label_mask
                    
    
                    # determine average dot signal intensity within nuclear region
                    nuc_counts = np.mean(dots_clean_cell).astype('float32')
                    print('nuc_counts = ' + str(nuc_counts))
                
                
                    #%% ----- DOTS MASK ----- (Chen's note: not entirely understanding this part) 
                    # mask_nuc_big = binary_dilation(label_mask,disk(20))   (Chen's note: this part is redundant, as we have the same code earlier)
                    # require that dot intensity value is greater than 
                    # 2x mean cytoplasm intensity  
                    if cyto_counts > 0 :
                        cyto_thresh_mask = dots_clean_cell > cyto_counts * 2
                        dots_clean_cell = dots_clean_cell * cyto_thresh_mask
                    
                    
                    # create mask for dots only
                    threshold_dots = filters.threshold_otsu(dots_clean_cell)
                    print('threshold_dots: ' + str(threshold_dots))
                    mask_dots = dots_clean_cell > threshold_dots
                    
                    # apply mask to dots_clean_cell image
                    dots_clean_cell = dots_clean_cell * mask_dots  
                    
                    
                    #%% ----- SAVE DOTS_CLEAN CELL IMAGE -----
                    root, ext = os.path.splitext(file)
                    if not os.path.exists(input_path + '/Processed'):
                        os.mkdir(input_path + '/Processed')
                        
                    # adjust contrast on dots_clean cell before saving
                    dots_clean_save = exposure.rescale_intensity(dots_clean_cell.astype('float32'))     
                    
                    filename = '_dots_clean_cell_1.png'
                    if cell_count >1:
                        filename = '_dots_clean_cell_' + str(cell_count) + '.png'
                        
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                    ax.imshow(dots_clean_save)
                    plt.savefig(input_path + '/Processed/' + root + filename, bbox_inches='tight')
                    plt.close(fig)
                    
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                    ax.imshow(nuc_label_img)
                    plt.savefig(input_path + '/Processed/' + root + '_nuc_seg.png', bbox_inches='tight')
                    plt.close(fig)
                
                    #%% ----- SIZE THRESHOLD -----
                    # require dots be greater than pre-defined minimum size 
                    #min_blob_size = 5
                    #print('min_blob_size = ' + str(min_blob_size) )
                    blobs, blob_nlabels = ndi.label(dots_clean_cell)
                    #blob_sizes = np.bincount(blobs.ravel())
                    blob_label, blob_sizes = np.unique(blobs.ravel(), return_counts=True)
                    #note: ravel doesn't make a copy so is faster than flatten
                    
                    mask_blob_sizes = blob_sizes > min_blob_size
                    mask_blob_sizes[0] = 0 # remove largest background object
                    mask_blob = mask_blob_sizes[blobs]
        
                    dots_clean_cell = dots_clean_cell * mask_blob
                     
                    # move to next image in queue if no blobs meet above criteria
                    # for intensity and size
                    blobs, nblobs = ndi.label(dots_clean_cell)
                    if nblobs < 2:
                        print('No dots found!')
                        continue
                   
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                    ax.imshow(blobs)#, cmap=plt.get_cmap('bone'))
                
                    
                    #%% ----- ENUMERATE AND MEAUSURE DOTS ------                      
                    
                    # loop over number of blobs in each image
                    for region in measure.regionprops(blobs, dots_clean_cell):
                        print 'hi we are in this foor loop'
                        if region.area < min_blob_size + 1:
                            print 'area = ', region.area
                            print('No dots fitting size critiera found!')
                            continue
                        
                        if region.mean_intensity < min_thresh:
                            print 'mean_intensity = ', region.mean_intensity
                            print('No dots above intensity threshold found!')
                            continue
                        
                        count = count + 1
                        
                        # create lists of blob properties
                        image_ID.append(file)
                        cell_ID.append(cell_count)
                        blob_ID.append(count)
                        area.append(region.area)
                        meanInt.append(region.mean_intensity)
                        diameter.append(region.equivalent_diameter)
                        majorAxis.append(region.major_axis_length)
                        minorAxis.append(region.minor_axis_length)
                        eccen.append(region.eccentricity)
                                         
                                       
                        # annotate blobs in segmentation image
                        # --> find coordinates of blob:
                        ctr_x = np.max(region.coords[:,0]) - np.min(region.coords[:,0])
                        ctr_y = np.max(region.coords[:,1]) - np.min(region.coords[:,1])
                        
                        ctr_x = round(ctr_x / 2) + np.min(region.coords[:,0])
                        ctr_y = round(ctr_y / 2) + np.min(region.coords[:,1])
                        
                        ctr_x = int(ctr_x)
                        ctr_y = int(ctr_y)
        
                        # --> add annotatation to segementation image
                        ab = plt.annotate(count, xy=(ctr_y, ctr_x),
                                     ha='center', va='center',fontsize=16,
                                     color='yellow')
                        
                        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                        ax.add_artist(ab)
                        ax.axis('off') # plt.axis('off')
                              
                              
                    # save watershed segmentation with labels
                    filename = '_seg_w_labels_cell_1.png'
                    if cell_count >1:
                        filename = '_seg_w_labels_cell_' + str(cell_count) + '.png'  
                                                 
                    
                    plt.savefig(input_path + '/Processed/' + root + filename ,bbox_inches='tight')
                    plt.close(fig)
                            
                # populate dataframe 
                df.Image_ID = image_ID
                df.Cell_ID = cell_ID
                df.Blob_ID = blob_ID  
                df.Area = area   
                df.MeanIntensity = meanInt
                df.Diameter = diameter
                df.MajorAxis = majorAxis
                df.MinorAxis = minorAxis
                df.Eccentricity = eccen                
                df.Nuclear_Intensity = nuc_counts
                df.Cytoplasm_Intensity = cyto_counts
                df.Nuclear_Area = nuc_size
                    
                    
                # save dataframe
                fname = '_ObjectTable.csv'
                df.to_csv(input_path + '/Processed/' + root + fname, index=False)   
            
            
            
            ## ----- FOR JOSEPH -----
            if No_dots == '2':
                dots_1 = img[2,:,:] # dot marker CY3
                dots_2 = img[3,:,:] # dots marker Cy5
                    
                 # ----- CLEAN UP DOTS -----
                dots_clean_1 = subtract_background(dots_1, nComponents)
                dots_clean_2 = subtract_background(dots_2, nComponents)     #(Chen's note: may need to change subtract_background function for dots_2)
                
             
                # loop over number of cells in image
                for i in range(len(nuc_labels)):   
                    cell_count = i + 1
                    print 'Eligible Cell ', cell_count      
    
                    label_mask = nuc_label_img.copy()
                    cell_label = list(nuc_labels)
                    dots_clean_cell_1 = dots_clean_1.copy()
                    dots_clean_cell_2 = dots_clean_2.copy()
                    
                                 
                    # if there are more than one cell in image,
                    # define nuclear mask for each cell
                    if len(nuc_labels) > 1 :
                        # remove object of interest from label list  (Chen's note: I don't understand why)
                        del cell_label[i]
                           
                        # set all object labels (except object of interest) to zero (Chen's note: I don't understand why)
                        for j in range(len(cell_label)):
                            label_mask[label_mask==cell_label[j]] = 0
                    
                    # set object of interest label to 1
                    label_mask[label_mask==nuc_labels[i]] = 1
                    # get the size(s) of the nucleus/nuclei
                    nuc_size = nuc_sizes[cell_count]
       
                    
                    # ----- DEFINE CYTOPLASM REGION FOR EACH NUCLEUS -----    (Chen's note: Does this make sense? can try subtracting nucleus area from cell area)
                    mask_nuc_big = binary_dilation(label_mask,disk(20))
                    mask_cyto = mask_nuc_big * ~mask_nuc
           
                    
                    # get mean background cytoplasm counts
                    tmp_1 = dots_clean_cell_1 * mask_cyto
                    cyto_counts_1 = np.mean(tmp_1).astype('float32')
                    tmp_2 = dots_clean_cell_2 * mask_cyto
                    cyto_counts_2 = np.mean(tmp_2).astype('float32')
                    #cyto_counts = np.mean(tmp.nonzero()).astype('float32')
                    print('cyto_count_1 = ' + str(cyto_counts_1))
                    print('cyto_count_2 = ' + str(cyto_counts_2))
                    
                    if np.isnan(cyto_counts_1):
                        print('cyto_counts_1 = nan')
                        continue
                    if np.isnan(cyto_counts_2):
                        print('cyto_counts_2 = nan')
                        continue
                    # ----- REQUIRE DOTS BE CO-LOCATED WITH NUCLEUS -----
                    # limit dots image to nuclear area
                    dots_clean_cell_1 = dots_clean_cell_1 * label_mask
                    dots_clean_cell_2 = dots_clean_cell_2 * label_mask
    
                    # determine average dot signal intensity within nuclear region
                    nuc_counts_1 = np.mean(dots_clean_cell_1).astype('float32')
                    nuc_counts_2 = np.mean(dots_clean_cell_2).astype('float32')
                    
                    print('nuc_counts_1 = ' + str(nuc_counts_1))
                    print('nuc_counts_2 = ' + str(nuc_counts_2))
                
                    #%% ----- DOTS MASK ----- (Chen's note: not entirely understanding this part) 
                    # mask_nuc_big = binary_dilation(label_mask,disk(20))   (Chen's note: this part is redundant, as we have the same code earlier)
                    # require that dot intensity value is greater than 
                    # 2x mean cytoplasm intensity  
                    if cyto_counts_1 > 0 :
                        cyto_thresh_mask_1 = dots_clean_cell_1 > cyto_counts_1 * 2
                        dots_clean_cell_1 = dots_clean_cell_1 * cyto_thresh_mask_1
                    if cyto_counts_2 > 0 :
                        cyto_thresh_mask_2 = dots_clean_cell_2 > cyto_counts_2 * 2
                        dots_clean_cell_2 = dots_clean_cell_2 * cyto_thresh_mask_2
                        
                        
                    # create mask for dots only
                    threshold_dots_1 = filters.threshold_otsu(dots_clean_cell_1)
                    threshold_dots_2 = 100 # (Chen's note: may change filter)
                    
                    print('threshold_dots_1: ' + str(threshold_dots_1))
                    print('threshold_dots_2: ' + str(threshold_dots_2))
                    mask_dots_1 = dots_clean_cell_1 > threshold_dots_1
                    mask_dots_2 = dots_clean_cell_2 > threshold_dots_2                  
                    # apply mask to dots_clean_cell image
                    dots_clean_cell_1 = dots_clean_cell_1 * mask_dots_1
                    dots_clean_cell_2 = dots_clean_cell_2 * mask_dots_2
                    
                    plt.imshow(dots_clean_cell_2)
                    
                    #%% ----- SAVE DOTS_CLEAN CELL IMAGE -----
                    root, ext = os.path.splitext(file)
                    if not os.path.exists(input_path + '/Processed'):
                        os.mkdir(input_path + '/Processed')
                        os.mkdir(input_path + '/Processed/dots_1')
                        os.mkdir(input_path + '/Processed/dots_2')
                        
                    # adjust contrast on dots_clean cell before saving
                    dots_clean_save_1 = exposure.rescale_intensity(dots_clean_cell_1.astype('float32'))     
                    dots_clean_save_2 = exposure.rescale_intensity(dots_clean_cell_2.astype('float32'))     
                    
                    filename_1 = '_dots_clean_cell_dots1.png'
                    if cell_count >1:
                        filename_1 = '_dots_clean_cell_dots1_' + str(cell_count) + '.png'
                    filename_2 = '_dots_clean_cell_dots1.png'
                    if cell_count >1:
                        filename_2 = '_dots_clean_cell_dots2_' + str(cell_count) + '.png'
                        
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                    ax.imshow(dots_clean_save_1)
                    plt.savefig(input_path + '/Processed/dots_1/' + root + filename_1, bbox_inches='tight')
                    plt.close(fig)
                    
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                    ax.imshow(nuc_label_img)
                    plt.savefig(input_path + '/Processed/dots_1/' + root + '_nuc_seg.png', bbox_inches='tight')
                    plt.close(fig)
                    
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                    ax.imshow(dots_clean_save_2)
                    plt.savefig(input_path + '/Processed/dots_2/' + root + filename_2, bbox_inches='tight')
                    plt.close(fig)
                    
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                    ax.imshow(nuc_label_img)
                    plt.savefig(input_path + '/Processed/dots_2/' + root + '_nuc_seg.png', bbox_inches='tight')
                    plt.close(fig)
                
                    #%% ----- SIZE THRESHOLD -----
                    # require dots be greater than pre-defined minimum size 
                    #min_blob_size = 5
                    #print('min_blob_size = ' + str(min_blob_size) )
                    blobs_1, blob_nlabels_1 = ndi.label(dots_clean_cell_1)
                    #blob_sizes = np.bincount(blobs.ravel())
                    blob_label_1, blob_sizes_1 = np.unique(blobs_1.ravel(), return_counts=True)
                    #note: ravel doesn't make a copy so is faster than flatten
              
                    mask_blob_sizes_1 = blob_sizes_1 > min_blob_size    #may add function to get different min_blob_size for different dots
                    mask_blob_sizes_1[0] = 0 # remove largest background object
                    mask_blob_1 = mask_blob_sizes_1[blobs_1]
        
                    dots_clean_cell_1 = dots_clean_cell_1 * mask_blob_1
                     
                    # move to next image in queue if no blobs meet above criteria
                    # for intensity and size
                    blobs_1, nblobs_1 = ndi.label(dots_clean_cell_1)
                    if nblobs_1 < 2:
                        print('No dots_1 found!')
                        continue
                   
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                    ax.imshow(blobs_1)#, cmap=plt.get_cmap('bone'))
                
                            
                    # now we do the same thing with dots_2 
                    blobs_2, blob_nlabels_2 = ndi.label(dots_clean_cell_2)
                    blob_label_2, blob_sizes_2 = np.unique(blobs_2.ravel(), return_counts=True)
                    mask_blob_sizes_2 = blob_sizes_2 > min_blob_size    #may add function to get different min_blob_size for different dots
                    mask_blob_sizes_2[0] = 0 # remove largest background object
                    mask_blob_2 = mask_blob_sizes_2[blobs_2]
                    dots_clean_cell_2 = dots_clean_cell_2 * mask_blob_2
                    blobs_2, nblobs_2 = ndi.label(dots_clean_cell_2)
                    if nblobs_2 < 2:
                        print('No dots_2 found!')
                        continue
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                    ax.imshow(blobs_2)#, cmap=plt.get_cmap('bone'))
                            
                    #%% ----- ENUMERATE AND MEAUSURE DOTS ------                      
                    
                    # loop over number of blobs in each image
                    for region in measure.regionprops(blobs_1, dots_clean_cell_1):
                        if region.area < min_blob_size + 1:
                            print 'area = ', region.area
                            print('No dots fitting size critiera found!')
                            continue
                        
                        if region.mean_intensity < min_thresh:
                            #print 'mean_intensity = ', region.mean_intensity
                            #print('No dots above intensity threshold found!')
                            continue
                        
                        count = count + 1
                        
                        # create lists of blob properties
                        image_ID.append(file)
                        cell_ID.append(cell_count)
                        blob_ID.append(count)
                        area.append(region.area)
                        meanInt.append(region.mean_intensity)
                        diameter.append(region.equivalent_diameter)
                        majorAxis.append(region.major_axis_length)
                        minorAxis.append(region.minor_axis_length)
                        eccen.append(region.eccentricity)
                                         
                                       
                        # annotate blobs in segmentation image
                        # --> find coordinates of blob:
                        ctr_x = np.max(region.coords[:,0]) - np.min(region.coords[:,0])
                        ctr_y = np.max(region.coords[:,1]) - np.min(region.coords[:,1])
                        
                        ctr_x = round(ctr_x / 2) + np.min(region.coords[:,0])
                        ctr_y = round(ctr_y / 2) + np.min(region.coords[:,1])
                        
                        ctr_x = int(ctr_x)
                        ctr_y = int(ctr_y)
        
                        # --> add annotatation to segementation image
                        ab = plt.annotate(count, xy=(ctr_y, ctr_x),
                                     ha='center', va='center',fontsize=16,
                                     color='yellow')
                        
                        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                        ax.add_artist(ab)
                        ax.axis('off') # plt.axis('off')
                              
                              
                    # save watershed segmentation with labels
                    filename_1 = '_seg_w_labels_cell_dots1_1.png'
                    if cell_count >1:
                        filename_1 = '_seg_w_labels_cell_dots1_' + str(cell_count) + '.png'  
                                                 
                    
                    plt.savefig(input_path + '/Processed/dots_1/' + root + filename_1 ,bbox_inches='tight')
                    plt.close(fig)
                            
                    # populate dataframe 
                    df.Image_ID = image_ID
                    df.Cell_ID = cell_ID
                    df.Blob_ID = blob_ID  
                    df.Area = area   
                    df.MeanIntensity = meanInt
                    df.Diameter = diameter
                    df.MajorAxis = majorAxis
                    df.MinorAxis = minorAxis
                    df.Eccentricity = eccen                
                    df.Nuclear_Intensity = nuc_counts_1
                    df.Cytoplasm_Intensity = cyto_counts_1
                    df.Nuclear_Area = nuc_size
                        
                        
                    # save dataframe
                    fname = '_ObjectTable_dots_1.csv'
                    df.to_csv(input_path + '/Processed/dots_1/' + root + fname, index=False)   
                
                
                    for region in measure.regionprops(blobs_2, dots_clean_cell_2):
                        if region.area < min_blob_size + 1:
                            print 'area = ', region.area
                            print('No dots fitting size critiera found!')
                            continue
                        
                        if region.mean_intensity < min_thresh:
                            #print 'mean_intensity = ', region.mean_intensity
                            #print('No dots above intensity threshold found!')
                            continue
                        
                        count = count + 1
                        
                        # create lists of blob properties
                        image_ID.append(file)
                        cell_ID.append(cell_count)
                        blob_ID.append(count)
                        area.append(region.area)
                        meanInt.append(region.mean_intensity)
                        diameter.append(region.equivalent_diameter)
                        majorAxis.append(region.major_axis_length)
                        minorAxis.append(region.minor_axis_length)
                        eccen.append(region.eccentricity)
                                         
                                       
                        # annotate blobs in segmentation image
                        # --> find coordinates of blob:
                        ctr_x = np.max(region.coords[:,0]) - np.min(region.coords[:,0])
                        ctr_y = np.max(region.coords[:,1]) - np.min(region.coords[:,1])
                        
                        ctr_x = round(ctr_x / 2) + np.min(region.coords[:,0])
                        ctr_y = round(ctr_y / 2) + np.min(region.coords[:,1])
                        
                        ctr_x = int(ctr_x)
                        ctr_y = int(ctr_y)
        
                        # --> add annotatation to segementation image
                        ab = plt.annotate(count, xy=(ctr_y, ctr_x),
                                     ha='center', va='center',fontsize=16,
                                     color='yellow')
                        
                        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30,12))
                        ax.add_artist(ab)
                        ax.axis('off') # plt.axis('off')
                              
                              
                    # save watershed segmentation with labels
                    filename_2 = '_seg_w_labels_cell_dots2_1.png'
                    if cell_count >1:
                        filename_2= '_seg_w_labels_cell_dots2_' + str(cell_count) + '.png'  
                                                 
                    
                    plt.savefig(input_path + '/Processed/dots_2/' + root + filename_2 ,bbox_inches='tight')
                    plt.close(fig)
                                
                    # populate dataframe 
                    df.Image_ID = image_ID
                    df.Cell_ID = cell_ID
                    df.Blob_ID = blob_ID  
                    df.Area = area   
                    df.MeanIntensity = meanInt
                    df.Diameter = diameter
                    df.MajorAxis = majorAxis
                    df.MinorAxis = minorAxis
                    df.Eccentricity = eccen                
                    df.Nuclear_Intensity = nuc_counts_2
                    df.Cytoplasm_Intensity = cyto_counts_2
                    df.Nuclear_Area = nuc_size
                        
                    
                    # save dataframe
                    fname = '_ObjectTable_dots_2.csv'
                    df.to_csv(input_path + '/Processed/dots_2/' + root + fname, index=False)   
    
    
                
#%% COMBINE ALL DATA TABLES
def combine_ObjectTables(input_path):
    print 'combining object tables...'
    df = pd.DataFrame()
    df_all = pd.DataFrame()
    files = get_objectTables(input_path)
    
    for file in files:   
        if file.endswith('_ObjectTable.csv'):     
            df = pd.read_csv(input_path + '/Processed/' + file) 
            df_all = df_all.append(df, ignore_index=True)
    df_all.to_csv(input_path + '/Processed/' + 'All_ObjectTable.csv', index=False)
    print 'All_ObjectTable.csv created!'
    
    
#%% ----- MAIN ----- 
input_path = get_path()
files = get_files(input_path)
No_dots = get_No_dots()
min_thresh = get_minThreshold()
min_blob_size = get_minSize()
img_units = get_units()
ndot_channels = get_nDotChannels()
nComponents = get_SpectralComponents()
process_images(input_path, files, min_thresh, min_blob_size, img_units, nComponents)
combine_ObjectTables(input_path)
print('Done!')


