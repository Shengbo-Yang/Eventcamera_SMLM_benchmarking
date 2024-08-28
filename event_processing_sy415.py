#Self defined python functions for event-data processing 
#For Prophesee packages, please download and install metavision


#//---- import section ----//

#//-- Basic packages --//
import numpy as np 
import skimage.feature
import skimage.transform
import tifffile
import skimage 
import matplotlib.pyplot as plt 
import time
from tqdm import tqdm 
import scipy

#//-- Prophesee packages --//
import metavision_sdk_base
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.py_reader import EventDatReader
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm 
import metavision_sdk_cv
import metavision_sdk_core



#//---- self defined functions ----//

def convert_to_photon(raw_image_path,camera_gain_path,camera_offset_path,camera_readnoise_path,QE,save_file=False):
    """
    Converting intensity to photon number

    #Parameters:
    //
    raw_image_path : Filepath of the original image to be processed
    camera_gain_path : Filepath of the sCMOS camera gain calibration image (needs to be cropped to the same ROI as the raw image)
    camera_offset_path : Filepath of the sCMOS camera offset calibration image (needs to be cropped to the same ROI as the raw image)
    camera_readnoise_path : Filepath of the sCMOS camera read noise calibration image (needs to be cropped to the same ROI as the raw image)
    QE : Quantum efficiency of the sCMOS camera at corresponding wavelength. 
    save_file : If set as True, the converted 'photon number images' will be saved as .tif
    //

    """ 
    #Read in image files
    bead_recording = tifffile.imread(raw_image_path)
    gain = tifffile.imread(camera_gain_path)
    offset = tifffile.imread(camera_offset_path)
    read_noise = tifffile.imread(camera_readnoise_path)

    #Converting to photons:  (raw-offset)/gain*quantum efficiency
    temp_stack1 = bead_recording-offset-read_noise
    temp_stack2 = temp_stack1/gain
    result_stack = temp_stack2*QE

    if save_file==True:
        result_path = raw_image_path[:-8]+'_photon_counts.tif'
        tifffile.imsave(result_path,result_stack)

def photon_difference_bead(base_image,bead_ROI=[0,575,0,610],exposure_frame=8):
    """
    Calculate the integrated photon count difference of the specified bead ROI

    #Parameters:
    //
    base_image: base image stack as numpy array.  Unit should be converted as photon count
    bead_ROI: ROI of the bead area. Format in [xmin,xmax,ymin,ymax]
    exposure_frame: average number of the beads being in bright or dim state
    //
    """

    bead = base_image[:,bead_ROI[2]:bead_ROI[3],bead_ROI[0]:bead_ROI[1]]  # Images are read in as (stack,y,x),  ROI specified as [xmin,xmax,ymin,ymax].  Slicing are used to crop image
    photon_sum = np.sum(bead,axis=(1,2)) # Take the sum of each frame as the integrated area photon count
    threshold = (np.max(photon_sum)+np.min(photon_sum))/2
    bright_state = photon_sum[(photon_sum > threshold)]
    bright_state_avg = np.mean(bright_state)
    dim_state = photon_sum[(photon_sum < threshold)]
    dim_state_avg = np.mean(dim_state)
    photon_diff_avg = bright_state_avg-dim_state_avg
    photon_diff = photon_diff_avg*exposure_frame

    return photon_diff


def read_raw_to_npy(event_recording_path,read_time=20000,max_read_time=1e9,save_file=False):
    """
    Read .raw event recording and convert to numpy array. Save array as .npy file if required

    Returns raw data (as numpy array), height and width of the raw record dimension

    #Parameters:
    //
    event_recording_path: File path for the .raw recording file
    read_time: The reading time stample duration for each iteration. Unit in mircoseconds (us)
    max_read_time: Maximum time stample of the events to be read in
    save_file: If set as True, the converted numpy file will be saved as .npy file
    //
    """
    event_iterator = EventsIterator(event_recording_path,delta_t=read_time,max_duration=max_read_time) #create event iterator object
    height,width = event_iterator.get_size() # get camera geometry
    
    raw_event_np = np.zeros(1,dtype=metavision_sdk_base.EventCD) #initialize array for storing events

    print('Start reading events from:')
    print(event_recording_path)
    print('')

    for evs in tqdm(event_iterator):
        raw_event_np = np.append(raw_event_np,evs)
    raw_event_np = np.delete(raw_event_np,0)

    print('File conversion finished')
    print('')

    if save_file==True:
        result_path = event_recording_path[:-4]+'_coverted_numpy.npy'
        np.save(result_path,raw_event_np)

    return raw_event_np,height,width

def load_filter_event(filepath,read_time=100000,max_read_time=1e9,filter_threshold_us=20000):
    """
    Load raw event data from specified files
    Filter out noise using algorithm

    Return filtered events and raw events as numpy arrays, together with camera geometry (height, width)

    #Parameters:
    filepath : path of the event recording file as .raw file
    read_time : read_time for each event load iteration. unit in mircosecond(us).
    max_read_time : the maxium timestamp of the entire event load process. unit in mircosecond(us).
    filter_threshold_us : the time threshold of the noise filtering algorithm. events supported by another event with the same polarity within this threshold will be accepted. 

    """
    event_iterator = EventsIterator(filepath,delta_t=read_time,max_duration=max_read_time) #create event iterator object
    height,width = event_iterator.get_size() # get camera geometry

    noise_filter_buffer = ActivityNoiseFilterAlgorithm.get_empty_output_buffer() #empty buffer for noise filtering algorithm
    raw_event_np = np.zeros(1,dtype=metavision_sdk_base.EventCD)
    filtered_event_np = np.zeros(1,dtype=metavision_sdk_base.EventCD)

    print('Raw recording from the following file has been read in and filtered')
    print(filepath)
    print("Filtering Start!")
    start_time = time.time() #log start time
    
    for evs in tqdm(event_iterator):
        raw_event_np = np.append(raw_event_np,evs)
        ActivityNoiseFilterAlgorithm(width,height,filter_threshold_us).process_events(evs,noise_filter_buffer)
        filtered_event_np = np.append(filtered_event_np,noise_filter_buffer.numpy())

    raw_event_np = np.delete(raw_event_np,0)
    filtered_event_np = np.delete(filtered_event_np,0)

    end_time = time.time() #log end time

    
    print("Filtering Done!")
    print(f'Total number of events before filtering is {raw_event_np.size}')
    print(f'Total number of events after filtering is {filtered_event_np.size}')
    print(f'Filter ratio is {(100*filtered_event_np.size/raw_event_np.size)}%')
    print(f'Filtering time = {(end_time-start_time)} seconds ')

    return filtered_event_np,raw_event_np,height,width

def event_frame_gen(events,width=1280,height=720,accumulation_time_us=10000,bit_depth=8,sign='positive',upcon_ratio=15):
    """
    Self-defined pseudo-frame generation function.  Bit depth can be specified 
    File size is calculated as well 

    Return pseudo-frame array (gray scale)


    #Parameters:
    events: Events fro pseudo-frame generation. Numpy array, format as ['x','y','p','t']
    width: width of the pseudo-frame. 
    height: height of the pseudo-frame.
    accumulation_time_us: Accumulation time for events to be used for a single frame. Unit in mircosecond (us)
    bit_depth: bit depth of the pseudo-frame. #Currently only 8bit and 16bit are supported
    upcon_ratio: Up converting ratio. Events are multiplied by this ratio to improve the visibility of the pseudo-frames. 

    """

    timestamp_range = np.max(events['t'])-np.min(events['t'])
    frame_counts = int(np.ceil(timestamp_range/accumulation_time_us))
    if bit_depth==8: 
        frames_gray = np.zeros((height,width,frame_counts),dtype=np.uint8) #(y,x,n)
    elif bit_depth==16:
        frames_gray = np.zeros((height,width,frame_counts),dtype=np.uint16)
    
    #file_size_in_MB = np.ceil(width*height*bit_depth*frame_counts/8/1024/1024)
    """
    W * H * bit * counts = file size in bits
    /8 to convert to Bytes
    /1024 to kiloBytes (KB)
    /1024 to MegaBytes (MB)
    """

    

    print('Frame generation start!')

    for i in tqdm(range(frame_counts)):
        timestamp_msk = (events['t']>=np.floor(i*accumulation_time_us+events['t'][0]))*(events['t']<np.floor((i+1)*accumulation_time_us+events['t'][0]))
        temp_event = events[timestamp_msk]
        for k in range(len(temp_event)):
            if sign == 'positive':
                frames_gray[temp_event['y'][k],temp_event['x'][k],i] += upcon_ratio*temp_event['p'][k]
            elif sign == 'negative':
                frames_gray[temp_event['y'][k],temp_event['x'][k],i] += np.abs(upcon_ratio*(temp_event['p'][k]-1))
            elif sign == 'both':
                frames_gray[temp_event['y'][k],temp_event['x'][k],i] += np.abs(upcon_ratio*(temp_event['p'][k]*2-1))

    return frames_gray


def event_FPS_frame_gen(events,width=1280,height=720,accumulation_time_us=10000,FPS=30,bit_depth=8,sign='positive',upcon_ratio=15):
    """
    Self-defined pseudo-frame generation function.  Bit depth can be specified as 8bit or 16bit 
    File size is calculated as well

    Return pseudo-frame array (gray scale) 


    #Parameters:
    events: Events fro pseudo-frame generation. Numpy array, format as ['x','y','p','t']
    width: width of the pseudo-frame. 
    height: height of the pseudo-frame.
    accumulation_time_us: Accumulation time for events to be used for a single frame. Unit in mircosecond (us)
    FPS: Frames per second
    bit_depth: bit depth of the pseudo-frame. #Currently only 8bit and 16bit are supported
    upcon_ratio: Up converting ratio. Events are multiplied by this ratio to improve the visibility of the pseudo-frames. 
    """

    timestamp_range = np.max(events['t'])-np.min(events['t'])
    final_timepoint = np.max(events['t'])
    frame_counts = int(np.ceil(timestamp_range*FPS/1e6))
    if bit_depth==8: 
        frames_gray = np.zeros((height,width,frame_counts),dtype=np.uint8) #(y,x,n)
    elif bit_depth==16:
        frames_gray = np.zeros((height,width,frame_counts),dtype=np.int16)
    
    #file_size_in_MB = np.ceil(width*height*bit_depth*frame_counts/8/1024/1024)
    """
    W * H * bit * counts = file size in bits
    /8 to convert to Bytes
    /1024 to kiloBytes (KB)
    /1024 to MegaBytes (MB)
    """

    frame_startingpoints = np.zeros((frame_counts))
    frame_startingpoints[0] = np.min(events['t'])
    for fm in range(frame_counts-1):
        frame_startingpoints[fm+1]=frame_startingpoints[fm]+np.floor(1e6/FPS)

    

    print('Frame generation start!')

    for i in tqdm(range(frame_counts)):
        if i < (frame_counts-1):
            timestamp_msk = (events['t']>=frame_startingpoints[i])*(events['t']<frame_startingpoints[i]+accumulation_time_us)
            temp_event = events[timestamp_msk]
            for k in range(len(temp_event)):
                if sign == 'positive':
                    frames_gray[temp_event['y'][k],temp_event['x'][k],i] += upcon_ratio*temp_event['p'][k]
                elif sign == 'negative':
                    frames_gray[temp_event['y'][k],temp_event['x'][k],i] += np.abs(upcon_ratio*(temp_event['p'][k]-1))
                elif sign == 'both':
                    frames_gray[temp_event['y'][k],temp_event['x'][k],i] += upcon_ratio*(temp_event['p'][k]*2-1)
        
        elif i == frame_counts-1:
            timestamp_msk = (events['t']>=frame_startingpoints[i])*(events['t']<final_timepoint)
            temp_event = events[timestamp_msk]
            for k in range(len(temp_event)):
                if sign == 'positive':
                    frames_gray[temp_event['y'][k],temp_event['x'][k],i] += upcon_ratio*temp_event['p'][k]
                elif sign == 'negative':
                    frames_gray[temp_event['y'][k],temp_event['x'][k],i] += np.abs(upcon_ratio*(temp_event['p'][k]-1))
                elif sign == 'both':
                    frames_gray[temp_event['y'][k],temp_event['x'][k],i] += upcon_ratio*(temp_event['p'][k]*2-1)            

    return frames_gray


def ROI_selection(events,xmin,xmax,ymin,ymax):
    """
    Extract the events specified by the ROI

    Return events within the ROI

    #Parameters:
    events: Events as numpy array. 
    xmin,xmax,ymin.ymax:  As the name indicates :-)
    """
    roi = (events['x']>xmin)*(events['x']<xmax)*(events['y']>ymin)*(events['y']<ymax) 
    selected_events = events[roi]
    return selected_events



def PSF_event_count(roi_events,plot_hist=False,max_period_number=100,period_diff_threshold=10000,count_threshold=100):
    """
    Calculate the average number of positive and negative events for each PSF/ROI area

    Returns: The average number of positive events and negative events. The standard deviation of postive and negative events 

    #Parameters:
    roi_events : Event array cropped by ROI. 
    plot_hist : If set as True, the histogram of event timestamps will be plotted for reference/debug purpose
    max_period_number : Maximum number of periods used for event number calculation. 
    period_diff_threshold : The timestamp difference threshould to cluster events. 
    count_threshold: Only event count exceed this threshold will be considered as signal events. 
    """

    timestamp_positive = roi_events[(roi_events['p']==1)]['t']
    timestamp_negative = roi_events[(roi_events['p']==0)]['t']

    if plot_hist == True:
        plt.hist([timestamp_positive,timestamp_negative],bins=5000,label=['Positive events','Negative events'],alpha=0.8)
        #plt.title()
        plt.xlabel('Timestamps/us')
        plt.legend()
        plt.show()
    
    positive_periods = np.zeros((max_period_number))
    negative_periods = np.zeros((max_period_number))
    count_postive_index = 0
    count_negative_index = 0

    for ts_p in tqdm(range(timestamp_positive.size-1)):
        if (timestamp_positive[ts_p+1]-timestamp_positive[ts_p]) < period_diff_threshold :
            positive_periods[count_postive_index] += 1 # cluster into current period
        
        else :
            count_postive_index += 1 # move to next period
            positive_periods[count_postive_index] += 1 # cluster into N+1 period
    
    final_positive_periods = positive_periods[(positive_periods>count_threshold)]

    if len(final_positive_periods)!=0:
        final_positive_periods = np.delete(final_positive_periods,[np.argmax(final_positive_periods),np.argmin(final_positive_periods)])
        #print('Numbers of postive events of each period are:')    
        #print(final_positive_periods)
        print('Positive events calculation success :)') #debug to cheer up 
    elif len(final_positive_periods)==0:
        final_positive_periods = 0
        print('Not enough positive events found')

    for ts_n in tqdm(range(timestamp_negative.size-1)):
        if (timestamp_negative[ts_n+1]-timestamp_negative[ts_n]) < period_diff_threshold :
            negative_periods[count_negative_index] += 1 # cluster into current period
        
        else :
            count_negative_index += 1 # move to next period
            negative_periods[count_negative_index] += 1 # cluster into N+1 period
    
    final_negative_periods = negative_periods[(negative_periods>count_threshold)]

    if len(final_negative_periods)!=0 :
        final_negative_periods = np.delete(final_negative_periods,[np.argmax(final_negative_periods),np.argmin(final_negative_periods)])
        #print('Numbers of negative events of each period are:')
        #print(final_negative_periods)
        print('Negative events calculation success :)') #debug to cheer up 
    elif len(final_negative_periods)==0 :
        final_negative_periods = 0
        print('Not enough negative events found')

    mean_postive_events = np.mean(final_positive_periods)
    std_positive_events = np.std(final_positive_periods)
    mean_negative_events = np.mean(final_negative_periods)
    std_negative_events = np.std(final_negative_periods)

    return mean_postive_events,mean_negative_events,std_positive_events,std_negative_events

    
def bead_mask_gen(frame_path,min_sig=4,max_sig=20,num_sig=10,b_threshold=5,b_overlap=0.2):
    """
    Customized function to generate a mask of bead ROIs from frame-based data.  
    ROIs format [x,y,r,photon differences]

    Return the ROIs

    #Parameters:
    frame_path: File path for the frame-based data.
    min_sig: min sigma value.  (used for bead detection)
    max_sig: max sigma value.  (used for bead detection)
    num_sig: number of sigma values. (used for bead detection)
    b_threshold: intensity threshold to be considered as beads
    b_overlap: ratio threshold of overlapping area to be consider as same/different beads. 


    """

    base_image = skimage.io.imread(frame_path,plugin='tifffile')
    image_width = np.shape(base_image)[2] # get image dimension for edge check 
    image_height = np.shape(base_image)[1]
    base_image_flip = np.flip(base_image,axis=1) # Filp vertically to match event-based data.  Only required for Prime95B data -- Event camera data on Salix 

    
    # Use the max intensity projection as maker image to get bead ROIs
    max_I_projection = np.max(base_image_flip,axis=0)
    beads_log = skimage.feature.blob_log(max_I_projection,min_sigma=min_sig,max_sigma=max_sig,num_sigma=num_sig,threshold=b_threshold,overlap=b_overlap)
    beads_num = beads_log.size/3
    print(f"Number of detected beads/ROIs is: {beads_num}") #debug info

    resized_frame = skimage.transform.rescale(base_image_flip,scale=1.12)
    resized_maxp = np.max(resized_frame,axis=0)
    beads_log_resized = skimage.feature.blob_log(resized_maxp,min_sigma=min_sig,max_sigma=max_sig,num_sigma=num_sig,threshold=b_threshold,overlap=b_overlap)
    resized_num = beads_log_resized.size/3
    print(f"Number of detected beads/ROIs of resized image is: {resized_num}") #debug info

    #initialize ROI array
    photon_diff = np.zeros((int(beads_num)))
    photon_diff_idx = 0 
    mask_x = np.zeros((int(beads_num)),dtype=np.int64)
    mask_x[:] = -1
    mask_y = np.zeros((int(beads_num)),dtype=np.int64)
    mask_y[:] = -1
    mask_r = np.zeros((int(beads_num)),dtype=np.int64)
    mask_r[:] = -1

    photon_diff_re = np.zeros((int(resized_num)))
    photon_diff_idx_re = 0 

    mask_x_re = np.zeros((int(resized_num)),dtype=np.int64)
    mask_x_re[:] = -1
    mask_y_re = np.zeros((int(resized_num)),dtype=np.int64)
    mask_y_re[:] = -1
    mask_r_re = np.zeros((int(resized_num)),dtype=np.int64)
    mask_r_re[:] = -1


    #debug plotting verification 
    fig1,ax1=plt.subplots()
    #ax1.imshow(max_I_projection,cmap='gray',vmin=0,vmax=127)
    for b in beads_log:
        y,x,r = b
        c = plt.Circle((x,y),r,color='red',linewidth=2,fill=False,alpha=0.8)
        #ax1.add_artist(c)

        #assign coord values
        mask_x[photon_diff_idx] = x
        mask_y[photon_diff_idx] = y
        mask_r[photon_diff_idx] = r

        #getting photon difference
        #checking edge to avoid Nan values
        if ((x+r)<(image_width-1))and((y+r)<(image_height-1))and((x-r)>0)and((y-r)>0):
            photon_diff[photon_diff_idx] = photon_difference_bead(base_image=base_image_flip,bead_ROI=[int(x-r),int(x+r),int(y-r),int(y+r)],exposure_frame=9)
            photon_diff_idx += 1
        else:
            print('Point at edge, disgarded, photon value assigned as 0')
            photon_diff[photon_diff_idx] = 0
            photon_diff_idx += 1

    #plt.show()
    #time.sleep(2)
    #plt.close()

    for br in beads_log_resized:
        y,x,r = br

        #assign value for check 
        mask_x_re[photon_diff_idx_re] = x
        mask_y_re[photon_diff_idx_re] = y
        mask_r_re[photon_diff_idx_re] = r
    

        #getting photon difference
        #checking edge to avoid Nan values
        if ((x+r)<(image_width*1.12-1))and((y+r)<(image_height*1.12-1))and((x-r)>0)and((y-r)>0):
            photon_diff_re[photon_diff_idx_re] = photon_difference_bead(base_image=resized_frame,bead_ROI=[int(x-r),int(x+r),int(y-r),int(y+r)],exposure_frame=6)
            photon_diff_idx_re += 1
        else:
            print('Point at edge, disgarded')
            photon_diff_re[photon_diff_idx_re] = 0
            photon_diff_idx_re += 1

    #print(photon_diff) #debug
    #print(photon_diff_re) #debug

    mask_x = np.floor(mask_x*1.12)+197 #0325 data
    mask_y = np.floor(mask_y*1.12)+45
    mask_r = np.floor(mask_r*1.12)

    #mask_x = np.floor(mask_x*1.12)+133 #0412 data
    #mask_y = np.floor(mask_y*1.12)-15
    #mask_r = np.floor(mask_r*1.12)

    ROI_mask = np.zeros((int(beads_num),4))
    ROI_mask[:,0] = mask_x
    ROI_mask[:,1] = mask_y
    ROI_mask[:,2] = mask_r
    ROI_mask[:,3] = photon_diff

    return ROI_mask


def plot_spot(base_frame,spot_rois,vmin=0,vmax=127):
    """
    Plotting base_frame and spot detection resutls for visual check
    """

    fig,ax = plt.subplots()
    ax.imshow(base_frame,cmap='gray',vmin=vmin,vmax=vmax)
    for spot in spot_rois:
        y,x,r = spot
        c = plt.Circle((x,y),r,color='yellow',linewidth=1,fill=False,alpha=0.8)
        ax.add_artist(c)
    plt.show()


def meas_freq_prophe(events,width=1280,height=720,center_freq=0.,freq_margin=0.2,savefile=False):
    """
    Extracting the frequency information using frequency detection algorithm from prophesee
    """
    max_freq = center_freq+center_freq*freq_margin
    min_freq = center_freq-center_freq*freq_margin

    freq_file_path='C:/Users/Shengbo Yang/Desktop/Event Cam/Code_output/'+str(time.time())+'.npy'


    #creating the algorithm and corresponding buffer 
    freq_filter = metavision_sdk_cv.FrequencyAlgorithm(width=width,height=height,min_freq=min_freq,max_freq=max_freq,filter_length=10)
    freq_cluster_filter = metavision_sdk_cv.FrequencyClusteringAlgorithm(width=width,height=height,min_cluster_size=4,max_time_diff=10000)

    freq_buff = freq_filter.get_empty_output_buffer()
    cluster_buff = freq_cluster_filter.get_empty_output_buffer()

    freq_filter.process_events(events,freq_buff)
    freq_cluster_filter.process_events(freq_buff,cluster_buff)

    print('Datatype of the frequency clustering results')
    print(cluster_buff.numpy().dtype)
    freq_avg = np.mean(cluster_buff.numpy()['frequency'])
    freq_std = np.std(cluster_buff.numpy()['frequency'])
    print(f'Average frequency is {freq_avg} Hz')
    print(f'Standard deviation is {freq_std} Hz')
    print('')
    
    #ploting the histogram of the frequency clusters to check the distribution
    #plt.hist(cluster_buff.numpy()['frequency'],bins=20)
    #plt.xlabel('Frequency/Hz')
    #plt.show()

    if savefile:
        np.save(freq_file_path,cluster_buff.numpy())


    return freq_avg,freq_std


def calculate_event_per_cycle(event,mod_frequency=2.):
    """
    Calculate average number of postive and negative events per unit area per cycle
    """

    postive_events = event[(event['p']==1)]
    negative_events = event[(event['p']==0)]

    total_pos_events = len(postive_events)
    total_neg_events = len(negative_events)

    recording_time = np.max(event['t']) - np.min(event['t'])
    cycles = recording_time * mod_frequency / 1e6 

    avg_pos_events = total_pos_events/cycles
    avg_neg_events = total_neg_events/cycles

    return avg_pos_events,avg_neg_events


def generate_heat_map(event,unit_photon_difference,mod_frequency=2.,ROI=[50,1230,50,670]):
    """
    generate a big heatmap for event camera response
    """

    width = ROI[1]-ROI[0]
    height = ROI[3]-ROI[2]

    heat_map_pos = np.zeros((width,height))
    heat_map_neg = np.zeros((width,height))

    #Scan through the array 

    recording_time = np.max(event['t']) - np.min(event['t'])
    cycles = recording_time * mod_frequency / 1e6 

    photon_value = unit_photon_difference*4.86*4.86

    for X in tqdm(np.arange(ROI[0],ROI[1])):
        for Y in tqdm(np.arange(ROI[2],ROI[3])):
            temp_events = event[(event['x']==X)*(event['y']==Y)]
            pos_events = temp_events[(temp_events['p']==1)]
            neg_events = temp_events[(temp_events['p']==0)]

            total_pos_events = np.size(pos_events)
            total_neg_events = np.size(neg_events)

            avg_pos_events = total_pos_events/cycles
            avg_neg_events = total_neg_events/cycles

            if avg_pos_events != 0 : pos_event_heat = photon_value/avg_pos_events
            else : pos_event_heat = 0

            if avg_neg_events != 0 : neg_event_heat = photon_value/avg_neg_events
            else : neg_event_heat = 0 

            heat_map_pos[X-width,Y-height] = pos_event_heat
            heat_map_neg[X-width,Y-height] = neg_event_heat

    return heat_map_pos,heat_map_neg





