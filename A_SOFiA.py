'''                        SWOT Oceanagrophy Functions ingvild Analysis of Surface-variability                     '''
'''                                     FUNCTIONS TO USE FOR SWOT NETCDF FILES :                                   '''
'''                                         BY : INGVILD OLDEN BJERKELUND                                          '''

'''                                      GENERAL INFORMATION ABOUT THE FUNCTIONS                                   '''
# The functions in this script are made to be used for SWOT NetCDF files. The functions are made to be used for
# the following purposes :
# 1) Add corrections to the data, such as geoid, xover, all or none
# 2) Mask the data, such as good, suspect, degraded or bad
# 3) Get the unraveled latitude, longitude and ssha-values from the corrected data
# 4) Calculate central difference, geostrophic velocity, speed, EKE, relative vorticity, divergence,
#    convergence, strain and Okubo-Weiss parameter
# 5) Smooth the data with 9-point smoothing
# 6) Get all files for one pass, get the number of cycles per pass, get file and cycle names and add
#    corrections to all cycles in a pass
# 7) Get all cycles corrected

'''                                         IMPORTING PACKAGES AND MODULES                                         '''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy as cy
from   matplotlib.dates import DateFormatter
import cartopy.crs as ccrs 
import cmocean
import netCDF4 as nc
import glob as glob
import pandas as pd
from scipy.stats import binned_statistic_2d

''' FUNCTIONS TO MOVE 8 POINT :                                                                                    '''
# RIGHT : 
def move_right(data):
    data_right = np.zeros((data.shape[0], data.shape[1]+1))
    data_right[:, 1:] = data
    data_right[:, 0]  = np.nan
    data_right = data_right[:, :-1]
    return data_right
# LEFT : 
def move_left(data):
    data_left = np.zeros((data.shape[0], data.shape[1]+1))
    data_left[:, :-1] = data
    data_left[:, -1]  = np.nan
    data_left = data_left[:, 1:]
    return data_left
# DOWN : 
def move_down(data):
    data_down = np.zeros((data.shape[0]+1, data.shape[1]))
    data_down[1:, :] = data
    data_down[0, :]  = np.nan
    data_down = data_down[:-1, :]
    return data_down
# UP :
def move_up(data):
    move_up = np.zeros((data.shape[0]+1, data.shape[1]))
    move_up[:-1, :] = data
    move_up[-1, :]  = np.nan
    move_up = move_up[1:, :]
    return move_up
# UP LEFT : 
def move_up_left(data):
    data_up      = move_up(data)
    data_up_left = move_left(data_up)
    return data_up_left
# UP RIGHT :
def move_up_right(data):
    data_up       = move_up(data)
    data_up_right = move_right(data_up)
    return data_up_right
# DOWN LEFT : 
def move_down_left(data):
    data_down      = move_down(data)
    data_down_left = move_left(data_down)
    return data_down_left
# DOWN RIGHT : 
def move_down_right(data):
    data_down       = move_down(data)
    data_down_right = move_right(data_down)
    return data_down_right
''' 9 POINT SMOOTHING FUNCTION :                                                                                   '''
def smooth_9_point(data):
    data_right      = move_right(data)
    data_left       = move_left(data)
    data_up         = move_up(data)
    data_down       = move_down(data)
    data_up_right   = move_up_right(data)
    data_up_left    = move_up_left(data)
    data_down_right = move_down_right(data)
    data_down_left  = move_down_left(data)
    # np.array([data, data_right, data_left, data_up, data_down, data_up_right, 
    # data_up_left, data_down_right, data_down_left])
    data_all = ([data, data_right, data_left, data_up, data_down, data_up_right, 
                 data_up_left, data_down_right, data_down_left])
    data_all_mean = np.nanmean(data_all, axis=0)
    return data_all_mean
''' SMOOTH MULTIPLE TIMES :                                                                                        '''
def smooth_multiple_times(data, times):
    for i in range(times):
        data = smooth_9_point(data)
    return data
''' CLEANER FUNCTION :                                                                                             '''
# CELAN EDGES AND MIDDEL :
def clean_smoothing(data, orig_data):
    # Make mask for nan in orig-data :
    #nan_df         = np.isnan(orig_data)#.isna()
    orig_data      = pd.DataFrame(orig_data)
    nan_df         = orig_data.isna()
    clean_data_msk = np.ma.masked_where(nan_df, data)
    clean_data     = np.ma.filled(clean_data_msk, np.nan)
    # CLEAN EDGES :
    #clean_data = clean_data[1:-1,1:-1]
    return clean_data
''' TEST THE FUNCTIONS ABOVE WITH A TEST FUNCTION :                                                                '''
def test_smooth_9_point():
    # TEST DATA :
    test_data = np.array([[1, 2, np.nan, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    # TEST SMOOTHING :
    data_right      = move_right(test_data)
    data_left       = move_left(test_data)
    data_up         = move_up(test_data)
    data_down       = move_down(test_data)
    data_up_right   = move_up_right(test_data)
    data_up_left    = move_up_left(test_data)
    data_down_right = move_down_right(test_data)
    data_down_left  = move_down_left(test_data)
    # PRINT RESULTS :
    print('test data')
    print(test_data)
    print('right')
    print(data_right)
    print('left')
    print(data_left)
    print('up')
    print(data_up)
    print('down')
    print(data_down)
    print('up_right')
    print(data_up_right)
    print('up_left')
    print(data_up_left)
    print('down_right')
    print(data_down_right)
    print('down_left')
    print(data_down_left)
    # SMOOTHING :
    smoothed_data = smooth_9_point(test_data)
    print('smoothed data')
    print(smoothed_data)
    # CLEAN DATA :
    cleaned_data = clean_smoothing(smoothed_data, test_data)
    print('cleaned data')
    print(cleaned_data)
''' FUNCTION TO GET THE YEAR, MONTH AND DAY OF THE CYCLE : '''
def finding_year_month_day_of_cycle(dataset_merged, cycle_number, int_or_str):
    ''' FINDING THE YEAR, MONTH AND DAY OF THE CYCLE : '''
    name_of_all_cycles = dataset_merged.concat_dim.data
    name_of_cycle      = name_of_all_cycles[cycle_number]
    # Find the year, month and day of the cycle :
    year_str  = name_of_cycle[13:17]
    month_str = name_of_cycle[17:19]
    day_str   = name_of_cycle[19:21]
    # Convert the strings to integers :
    year_int  = int(year_str)
    month_int = int(month_str)
    day_int   = int(day_str)
    ''' RETURN THE YEAR, MONTH AND DAY OF THE CYCLE : '''
    if int_or_str == 'str':
        return year_str, month_str, day_str
    elif int_or_str == 'int':
        return year_int, month_int, day_int
''' FUNCTION TO FIND THE SEASON OF THE CYCLE : '''
def find_season_of_cycle(dataset_merged):
    ''' FIRST FIND THE LENGTH OF THE DATASET (NUMBER OF CYCLES) : '''
    num_cycles = len(dataset_merged.concat_dim.data)
    ''' FINDING THE MONTH OF THE CYCLE : '''
    month_list = []
    for i in range(num_cycles):
        year, month, day = finding_year_month_day_of_cycle(dataset_merged, i, 'int')
        month_list.append(month)
    ''' FINDING THE SEASON OF THE CYCLE : '''
    DJF = []
    MAM = []
    JJA = []
    SON = []
    for j,k in zip(month_list, range(len(month_list))):
        if j == 12 or j == 1 or j == 2:
            DJF.append(k)
        elif j == 3 or j == 4 or j == 5:
            MAM.append(k)
        elif j == 6 or j == 7 or j == 8:
            JJA.append(k)
        elif j == 9 or j == 10 or j == 11:
            SON.append(k)
    Winter = []
    Summer = []
    for l,m in zip(month_list, range(len(month_list))):
        if l ==11 or l == 12 or l == 1 or l == 2:
            Winter.append(m)
        elif l == 5 or l == 6 or l == 7 or l == 8:
            Summer.append(m)
    return month_list, DJF, MAM, JJA, SON, Winter, Summer
def find_season_of_cycle_new(dataset_merged):
    ''' FIRST FIND THE LENGTH OF THE DATASET (NUMBER OF CYCLES) : '''
    num_cycles = len(dataset_merged.concat_dim.data)
    ''' FINDING THE MONTH OF THE CYCLE : '''
    month_list = []
    year_list  = []
    for i in range(num_cycles):
        year, month, day = finding_year_month_day_of_cycle(dataset_merged, i, 'int')
        month_list.append(month)
        year_list.append(year)
    ''' FINDING THE SEASON OF THE CYCLE : '''
    # Winter is Dec, Jan, Feb, Mar [12, 1, 2, 3]
    # Summer is Jun, Jul, Aug, Sep [ 6, 7, 8, 9]
    Winter_24 = []
    Winter_25 = []
    Summer_24 = []
    for j,k, y in zip(month_list, range(len(month_list)), year_list):
        if (j == 12 and y == 2023) or (j == 1 and y == 2024) or (j == 2 and y == 2024) or (j == 3 and y == 2024):
            Winter_24.append(k)
        elif (j == 12 and y == 2024) or (j == 1 and y == 2025) or (j == 2 and y == 2025) or (j == 3 and y == 2025):
            Winter_25.append(k)
        elif (j == 6 and y == 2024) or (j == 7 and y == 2024) or (j == 8 and y == 2024) or (j == 9 and y == 2024):
            Summer_24.append(k)
    return month_list, year_list, Winter_24, Winter_25, Summer_24
''' FUNCTION TO CONCATENATE THE SMOOTHED FILES : '''
# NB! ASSUMES MASKED AND CORRECTED BEFORE !
def concat_files_smoothed(merged_dataset, merged_masked_dataset, times_smooth):
    # Define lines and pixel number : 
    num_lines  = len(merged_dataset['num_lines'])
    num_pixels = len(merged_dataset['num_pixels'])
    print(f'num_lines:{num_lines} , num_pixels:{num_pixels}')
    # Initialize the new dimension name :
    filenamedim = []
    files = {}
    # The number of files to concatenate :
    len_files = len(merged_dataset['concat_dim'])
    print(f'nr of cycles: {len_files}')
    for i in range(len_files):
        # Keep track of the pass and day :
        name_now = merged_dataset['concat_dim'][i].data
        print('Working on ' + name_now)
        cycle_ds = merged_masked_dataset[i]
        # Smooth each cycle separate
        smoothed_one_cycle = smooth_multiple_times(cycle_ds, times_smooth)
        cleaned_one_cycle  = clean_smoothing(smoothed_one_cycle, cycle_ds)
        # Store ass DataArray
        files[i] = xr.DataArray(cleaned_one_cycle, dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
        filenamedim.append(name_now) 
    # Concatenating the files : (meanings that the files are concatenated along the new dimension)
    concated_files = xr.concat([(files[a]) for a in range(len_files)], dim =filenamedim) 
    return concated_files
''' FUNCTION TO GET CORRECTED AND VARIABLES, LAT, LON, SEASON-INFORMATION : '''
# Per now, can only smooth one cycle at time, need to fix or dont use smooth ?
def get_corrected_and_masked_variables(merged_dataset, var_type, correction_type, mask_type, smoothing):
    # Initialize variables
    corrected = None
    ''' CORRECT THE VARIABLES : '''
    if correction_type == 'ssh':
       corrected = merged_dataset[var_type]  + merged_dataset['height_cor_xover'] - merged_dataset['geoid']
    elif correction_type == 'ssha':
        corrected = merged_dataset[var_type] + merged_dataset['height_cor_xover']
    elif correction_type == 'ssh_tide':
        corrected = merged_dataset[var_type] + merged_dataset['height_cor_xover'] - merged_dataset['geoid'] - merged_dataset['solid_earth_tide'] - merged_dataset['ocean_tide_fes'] - merged_dataset['internal_tide_hret'] - merged_dataset['pole_tide'] - merged_dataset['dac']
    ''' MASK THE VARIABLES : '''
    if mask_type == 'qual_flag':
        good_data = merged_dataset[var_type+'_qual'] > 0
        # Apply the mask :
        masked_1 = np.ma.masked_where(good_data, corrected)
        masked   = np.ma.filled(masked_1, np.nan)
    elif mask_type == 'height_and_qual_flag':
        good_data1  = merged_dataset[var_type+'_qual'] > 0
        good_data2  = merged_dataset['height_cor_xover_qual'] > 0
        # Apply the mask :
        masked_1 = np.ma.masked_where(good_data1, corrected)
        masked_2 = np.ma.masked_where(good_data2, masked_1)
        masked   = np.ma.filled(masked_2, np.nan)
    elif mask_type == 'all':
        good_data1 = merged_dataset[var_type+'_qual'] > 0
        good_data2 = merged_dataset['height_cor_xover_qual'] > 0
        good_data3 = merged_dataset['ancillary_surface_classification_flag'] > 1
        # Apply the mask :
        masked_1 = np.ma.masked_where(good_data1, corrected)
        masked_2 = np.ma.masked_where(good_data2, masked_1)
        masked_3 = np.ma.masked_where(good_data3, masked_2)
        masked   = np.ma.filled(masked_3, np.nan)
    ''' SMOOTH IF WANTED : '''
    if smoothing == None:
        masked_d = masked
        #month_list, DJF, MAM, JJA, SON, Winter, Summer = find_season_of_cycle(merged_dataset)
    else:
        concated_files = concat_files_smoothed(merged_dataset, masked, times_smooth=smoothing)
        masked_d = concated_files
        #month_list, DJF, MAM, JJA, SON, Winter, Summer = find_season_of_cycle(merged_dataset)
    ''' GET THE LATITUDE AND LONGITUDE : '''
    lat = corrected.latitude.data
    lon = corrected.longitude.data
    #lat = merged_dataset.latitude.data
    #lon = merged_dataset.longitude.data
    ''' GET THE SEASON INFORMATION : '''
    month_list, DJF, MAM, JJA, SON, Winter, Summer = find_season_of_cycle(merged_dataset)
    ''' RETURN THE MASKED VARIABLE : '''
    return masked_d, lat, lon, month_list, DJF, MAM, JJA, SON, Winter, Summer
def get_corrected_and_masked_variables2(merged_dataset, var_type, correction_type, mask_type, smoothing):
    # Initialize variables
    corrected = None
    ''' CORRECT THE VARIABLES : '''
    if correction_type == 'ssh':
       corrected = merged_dataset[var_type]  + merged_dataset['height_cor_xover'] - merged_dataset['geoid']
    elif correction_type == 'ssha':
        corrected = merged_dataset[var_type] + merged_dataset['height_cor_xover']
    elif correction_type == 'ssh_tide':
        corrected = merged_dataset[var_type] + merged_dataset['height_cor_xover'] - merged_dataset['geoid'] - merged_dataset['solid_earth_tide'] - merged_dataset['ocean_tide_fes'] - merged_dataset['ocean_tide_non_eq'] - merged_dataset['internal_tide_hret'] - merged_dataset['pole_tide'] - merged_dataset['dac']
    ''' MASK THE VARIABLES : '''
    if mask_type == 'qual_flag':
        good_data = merged_dataset[var_type+'_qual'] > 0
        # Apply the mask :
        masked_1 = np.ma.masked_where(good_data, corrected)
        masked   = np.ma.filled(masked_1, np.nan)
    elif mask_type == 'height_and_qual_flag':
        good_data1  = merged_dataset[var_type+'_qual'] > 0
        good_data2  = merged_dataset['height_cor_xover_qual'] > 0
        # Apply the mask :
        masked_1 = np.ma.masked_where(good_data1, corrected)
        masked_2 = np.ma.masked_where(good_data2, masked_1)
        masked   = np.ma.filled(masked_2, np.nan)
    elif mask_type == 'all':
        good_data1 = merged_dataset[var_type+'_qual'] > 0
        good_data2 = merged_dataset['height_cor_xover_qual'] > 0
        #good_data3 = merged_dataset['ancillary_surface_classification_flag'] > 1
        good_data3 = merged_dataset['ancillary_surface_classification_flag'] > 0
        good_data4 = merged_dataset['dynamic_ice_flag'] > 0
        # Apply the mask :
        masked_1 = np.ma.masked_where(good_data1, corrected)
        masked_2 = np.ma.masked_where(good_data2, masked_1)
        masked_3 = np.ma.masked_where(good_data3, masked_2)
        masked_4 = np.ma.masked_where(good_data4, masked_3)
        masked   = np.ma.filled(masked_4, np.nan)
    ''' SMOOTH IF WANTED : '''
    if smoothing == None:
        masked_d = masked
        #month_list, DJF, MAM, JJA, SON, Winter, Summer = find_season_of_cycle(merged_dataset)
    else:
        concated_files = concat_files_smoothed(merged_dataset, masked, times_smooth=smoothing)
        masked_d = concated_files
        #month_list, DJF, MAM, JJA, SON, Winter, Summer = find_season_of_cycle(merged_dataset)
    ''' GET THE LATITUDE AND LONGITUDE : '''
    lat = corrected.latitude.data
    lon = corrected.longitude.data
    #lat = merged_dataset.latitude.data
    #lon = merged_dataset.longitude.data
    ''' GET THE SEASON INFORMATION : '''
    month_list, year_list, Winter_24, Winter_25, Summer_24 = find_season_of_cycle_new(merged_dataset)
    ''' RETURN THE MASKED VARIABLE : '''
    return masked_d, lat, lon, month_list, year_list, Winter_24, Winter_25, Summer_24
''' FUNCTION TO CALCULATE ALL STATISTICS FOR ALL CYCLES : '''
def calculate_statistics_all(masked_dataset):
    ''' NUMBER OF GOOD DATA POINTS : '''
    count = np.sum(~np.isnan(masked_dataset), axis=0)
    ''' MEAN OF VARIABLE : '''
    mean = np.nanmean(masked_dataset, axis=0)
    '''MEAN OF VARIABLE^2 : '''
    mean_of_square = np.nanmean(masked_dataset**2, axis=0)
    '''MEAN OF (VARIABLE^2 - MEAN OF VARIABLE^2) : '''
    mean_of_squared_difference = np.nanmean(masked_dataset**2 - mean_of_square, axis=0)
    ''' MEAN OF (VARIABLE - MEAN OF VARIABLE)^2 : '''
    mean_of_difference_squared = np.nanmean((masked_dataset - mean)**2, axis=0)
    ''' STANDARD DEVIATION OF VARIABLE : '''
    std = np.nanstd(masked_dataset, axis=0)
    ''' MEDIAN OF VARIABLE : '''
    median = np.nanmedian(masked_dataset, axis=0)
    ''' MINIMUM OF VARIABLE : '''
    min = np.nanmin(masked_dataset, axis=0)
    ''' MAXIMUM OF VARIABLE : '''
    max = np.nanmax(masked_dataset, axis=0)
    ''' VARIANCES OF VARIABLE : '''
    variance = np.nansum((masked_dataset - mean)**2, axis=0) / count
    ''' RETURN THE STATISTICS : '''
    return count, mean, mean_of_square, mean_of_squared_difference, mean_of_difference_squared, std, median, min, max, variance
''' FUNCTION TO CALCULATE THE BASIC STATISTICS FOR ALL CYCLES OR SEASONAL : '''
def calculate_statistics_basics(masked_dataset):
    ''' MEAN OF VARIABLE : '''
    mean = np.nanmean(masked_dataset, axis=0)
    '''MEAN OF VARIABLE^2 : '''
    mean_of_square = np.nanmean(masked_dataset**2, axis=0)
    '''MEAN OF (VARIABLE^2 - MEAN OF VARIABLE^2) : '''
    mean_of_squared_difference = np.nanmean(masked_dataset**2 - mean_of_square, axis=0)
    ''' MEAN OF (VARIABLE - MEAN OF VARIABLE)^2 : '''
    mean_of_difference_squared = np.nanmean((masked_dataset - mean)**2, axis=0)
    ''' STANDARD DEVIATION OF VARIABLE : '''
    std = np.nanstd(masked_dataset, axis=0)
    ''' RETURN THE STATISTICS : '''
    return mean, mean_of_squared_difference, mean_of_difference_squared, std
''' FUNCTION TO CALCULATE UG AND VG (KEEPING THE CYCLES DIMENSION) : '''
def calculate_ug_vg(masked_dataset, lat, lon):
    ''' GET SLA_DX AND SLA_DY : '''
    sla_diff_x = np.gradient(masked_dataset, axis=2)
    sla_diff_y = np.gradient(masked_dataset, axis=1)
    ''' dx, dy : '''
    dx = 2000
    dy = 2000
    ''' Tetha, g, f : '''
    tetha = np.deg2rad(lat)
    g     = 9.81
    f     = 2 * 7.2921e-5 * np.sin(tetha)
    ''' ug, vg : '''
    ug  = - (g / f) * (sla_diff_y / dy)
    vg  =   (g / f) * (sla_diff_x / dx)
    ''' RETURN UG AND VG : '''
    return ug, vg
''' FUNCTION TO ROTATE UG AND VG : '''
def rotate_dataset_original_grid_all_cycles(lat, lon, u, v):
    TETHA = lon
    PHI   = lat
    num_lines  = PHI.shape[0]
    num_pixels = PHI.shape[1]
    TETHA      = np.where(TETHA > 180, TETHA - 360, TETHA)
    # because num_lines is y and num_pixels is x
    delta_PHI        = np.diff(PHI,   axis=1)
    delta_TETHA      = np.diff(TETHA, axis=1)
    delta_PHI        = np.append(delta_PHI, np.zeros((num_lines, 1)), axis=1)
    delta_TETHA      = np.append(delta_TETHA, np.zeros((num_lines, 1)), axis=1)
    tetha_rad        = np.deg2rad(TETHA)
    phi_rad          = np.deg2rad(PHI)
    delta_tetha_rad  = np.deg2rad(delta_TETHA)
    delta_phi_rad    = np.deg2rad(delta_PHI)
    cos_PHI          = np.cos(phi_rad)
    curly_TETHA      = np.arctan2(delta_phi_rad,cos_PHI*delta_tetha_rad)
    u_rotated        = u*np.cos(curly_TETHA) - v*np.sin(curly_TETHA)
    v_rotated        = u*np.sin(curly_TETHA) + v*np.cos(curly_TETHA)
    u_rotated_by_phi = u_rotated/cos_PHI
    return u_rotated, v_rotated, u_rotated_by_phi
''' FUNCTION TO GET THE VARIANCE ELLIPSE VARIABLES : '''
def variance_ellipse(u,v):
    """
    Compute parameters of the variance ellipse.
    Args: 
        u: 1-D array of eastward velocities (or real part of complex variable)
        v: 1-D array of northward velocities (or imaginary part of complex variable)
    Returns:
        a: semi-major axis 
        b: semi-minor axis 
        theta: orientation angle counterclockwise from x axis, in radians
    """
    #compute terms in the covariance matrix
    cuu=np.nanmean(np.multiply(u-np.nanmean(u),u-np.nanmean(u)))
    cvv=np.nanmean(np.multiply(v-np.nanmean(v),v-np.nanmean(v)))
    cuv=np.nanmean(np.multiply(u-np.nanmean(u),v-np.nanmean(v)))
    detc=np.real(cuu*cvv-cuv**2) #determinant of covariance matrix
    trc=cuu+cvv #trace of covariance matrix
    a=np.sqrt(trc/2+np.sqrt(trc**2-4*detc)/2)#semi-major axis
    b=np.sqrt(trc/2-np.sqrt(trc**2-4*detc)/2)#semi-minor axis
    theta=np.arctan2(2*cuv,cuu-cvv)/2#orientation angle
    return a,b,theta
''' FUNCTION TO GET CUU, CVV AND CUV, ROTATED : '''
def calculate_cuu_cvv_cuv(masked_dataset, lat, lon):
    ''' CALCULATE UG AND VG : '''
    ug, vg = calculate_ug_vg(masked_dataset, lat, lon)
    ''' CALCULATE UG AND VG ROTATED : '''
    u_rotated, v_rotated, u_rotated_by_phi = rotate_dataset_original_grid_all_cycles(lat, lon, ug, vg)
    ''' MEAN OF ug AND vg : '''
    ug_mean = np.nanmean(u_rotated, axis=0)
    vg_mean = np.nanmean(v_rotated, axis=0)
    ''' CALCULATE CUU, CVV, CUV : '''
    cuu = np.nanmean(np.multiply(u_rotated - ug_mean, u_rotated - ug_mean), axis=0)
    cvv = np.nanmean(np.multiply(v_rotated - vg_mean, v_rotated - vg_mean), axis=0)
    cuv = np.nanmean(np.multiply(u_rotated - ug_mean, v_rotated - vg_mean), axis=0)
    ''' CALCULATE DETC, TRC : '''
    detc = np.real(cuu*cvv-cuv**2)   # determinant of covariance matrix
    trc = cuu + cvv                  # trace of covariance matrix
    ''' CALCULATE A, B, THETA : '''
    a     = np.sqrt(trc/2+np.sqrt(trc**2-4*detc)/2)  # semi-major axis
    b     = np.sqrt(trc/2-np.sqrt(trc**2-4*detc)/2)  # semi-minor axis
    theta = np.arctan2(2*cuv,cuu-cvv)/2              # orientation angle
    ''' RETURN CUU, CVV, CUV, DETC, TRC, A, B, THETA : '''
    return cuu, cvv, cuv, detc, trc, a, b, theta
def calculate_cuu_cvv_cuv_mean(masked_dataset, lat, lon):
    ''' CALCULATE UG AND VG : '''
    ug, vg = calculate_ug_vg(masked_dataset, lat, lon)
    ''' CALCULATE UG AND VG ROTATED : '''
    u_rotated, v_rotated, u_rotated_by_phi = rotate_dataset_original_grid_all_cycles(lat, lon, ug, vg)
    ''' MEAN OF ug AND vg : '''
    #ug_mean = np.nanmean(u_rotated, axis=0)
    #vg_mean = np.nanmean(v_rotated, axis=0)
    ''' CALCULATE CUU, CVV, CUV : '''
    cuu = np.nanmean(np.multiply(u_rotated, u_rotated), axis=0)
    cvv = np.nanmean(np.multiply(v_rotated, v_rotated), axis=0)
    cuv = np.nanmean(np.multiply(u_rotated, v_rotated), axis=0)
    ''' CALCULATE DETC, TRC : '''
    detc = np.real(cuu*cvv-cuv**2)   # determinant of covariance matrix
    trc = cuu + cvv                  # trace of covariance matrix
    ''' CALCULATE A, B, THETA : '''
    a     = np.sqrt(trc/2+np.sqrt(trc**2-4*detc)/2)  # semi-major axis
    b     = np.sqrt(trc/2-np.sqrt(trc**2-4*detc)/2)  # semi-minor axis
    theta = np.arctan2(2*cuv,cuu-cvv)/2              # orientation angle
    ''' RETURN CUU, CVV, CUV, DETC, TRC, A, B, THETA : '''
    return cuu, cvv, cuv, detc, trc, a, b, theta
''' FUNCTION TO CALCULATE EKE, UG_MEAN, VG_MEAN : '''
def calculate_eke_ug_mean_vg_mean(masked_dataset, lat, lon):
    ''' CALCULATE UG AND VG : '''
    ug, vg = calculate_ug_vg(masked_dataset, lat, lon)
    ''' MEAN OF ug AND vg : '''
    ug_mean = np.nanmean(ug, axis=0)
    vg_mean = np.nanmean(vg, axis=0)
    ''' EKE : '''
    ''' METHOD 1 : REMOVE MEAN BEFORE SQUARING : '''
    # EKE Jonathan : EKE = 1/2*[<(u-<u>)^2>+<(v-<v>)^2>] 
    ug_diff_mean = ug - ug_mean
    vg_diff_mean = vg - vg_mean
    ug_diff_sq = ug_diff_mean**2
    vg_diff_sq = vg_diff_mean**2
    eke_jon = 0.5 * (np.nanmean(ug_diff_sq, axis=0) + np.nanmean(vg_diff_sq, axis=0))
    ''' METHOD 2 : WITHOUT REMOVING THE MEAN : '''
    # EKE Jonathan Without removing the mean
    eke_not_removed   = 0.5 * (np.nanmean(ug**2, axis=0) + np.nanmean(vg**2, axis=0))
    ''' METHOD 3 : MEAN BEFORE SQUARING : '''
    # Mean before the square root
    eke_wrong     = 0.5 * (ug_mean**2 + vg_mean**2)
    ''' RETURN ug, vg AND EKE : '''
    return ug_mean, vg_mean, eke_jon, eke_wrong, eke_not_removed
''' FUNCTION TO CALCULATE EKE, UG_MEAN, VG_MEAN SEASONAL : '''
def calculate_eke_ug_mean_vg_mean_seasonal(masked_dataset, lat, lon, season):
    ''' CALCULATE UG AND VG : '''
    ug, vg = calculate_ug_vg(masked_dataset, lat, lon)
    ''' GET THE SEASONAL INDICES : '''
    ug_seasonal = ug[season]
    vg_seasonal = vg[season]
    ''' MEAN OF ug AND vg : '''
    ug_mean_o = np.nanmean(ug, axis=0)
    vg_mean_o = np.nanmean(vg, axis=0)
    ''' MEAN OF ug AND vg  SEASONAL : '''
    ug_mean = np.nanmean(ug_seasonal, axis=0)
    vg_mean = np.nanmean(vg_seasonal, axis=0)
    ''' EKE : '''
    ''' METHOD 1 : REMOVE MEAN BEFORE SQUARING : '''
    # EKE Jonathan : EKE = 1/2*[<(u-<u>)^2>+<(v-<v>)^2>] 
    ug_diff_mean = ug_seasonal - ug_mean
    vg_diff_mean = vg_seasonal - vg_mean
    ug_diff_mean_o = ug_seasonal - ug_mean_o
    vg_diff_mean_o = vg_seasonal - vg_mean_o
    ug_diff_sq = ug_diff_mean**2
    vg_diff_sq = vg_diff_mean**2
    ug_diff_sq_o = ug_diff_mean_o**2
    vg_diff_sq_o = vg_diff_mean_o**2
    eke_jon = 0.5 * (np.nanmean(ug_diff_sq, axis=0) + np.nanmean(vg_diff_sq, axis=0))
    eke_jon_o = 0.5 * (np.nanmean(ug_diff_sq_o, axis=0) + np.nanmean(vg_diff_sq_o, axis=0))
    ''' METHOD 2 : WITHOUT REMOVING THE MEAN : '''
    # EKE Jonathan Without removing the mean
    eke_not_removed   = 0.5 * (np.nanmean(ug_seasonal**2, axis=0) + np.nanmean(vg_seasonal**2, axis=0))
    ''' METHOD 3 : MEAN BEFORE SQUARING : '''
    # Mean before the square root
    eke_wrong     = 0.5 * (ug_mean**2 + vg_mean**2)
    ''' RETURN ug, vg AND EKE : '''
    return ug_mean, vg_mean, eke_jon, eke_jon_o, eke_wrong, eke_not_removed
''' FUNCTION TO CREATE THE DATASET FOR STATISTICS : '''
def store_all_statistics_all_cycles(merged_dataset, var_type, correction_type, mask_type, smoothing, smoothing_after):
    ''' GET THE MASKED VARIABLE, LAT, LON, AND INFORMATION ABOUT CYCLES : ''' 
    masked_dataset, lat, lon, month_list, DJF, MAM, JJA, SON, Winter, Summer = get_corrected_and_masked_variables(merged_dataset, var_type, correction_type, mask_type, smoothing)
    ''' CALCULATE ALL STATISTICS : '''
    count, mean, mean_of_square, mean_of_squared_difference, mean_of_difference_squared, std, median, min, max, variance = calculate_statistics_all(masked_dataset)
    ''' Num_lines, Num_pixels '''
    num_lines  = lat.shape[0]
    num_pixels = lat.shape[1]
    ''' Fix the Longitude : '''
    new_lon = np.where(lon > 180, lon - 360, lon)
    ''' SMOOTHING : '''
    if smoothing_after == None:
        countcd        = count
        meancd         = mean
        mean_sqcd      = mean_of_square
        mean_sq_diffcd = mean_of_squared_difference
        mean_diff_sqcd = mean_of_difference_squared
        stdcd          = std
        mediancd       = median
        mincd          = min
        maxcd          = max
        variancecd     = variance
        latcd          = lat
        new_loncd      = new_lon
    elif smoothing_after != None:
        countd        = smooth_multiple_times(count,                      smoothing)
        meand         = smooth_multiple_times(mean,                       smoothing)
        mean_sqd      = smooth_multiple_times(mean_of_square,             smoothing)
        mean_sq_diffd = smooth_multiple_times(mean_of_squared_difference, smoothing)
        mean_diff_sqd = smooth_multiple_times(mean_of_difference_squared, smoothing)
        stdd          = smooth_multiple_times(std,                        smoothing)
        mediand       = smooth_multiple_times(median,                     smoothing)
        mind          = smooth_multiple_times(min,                        smoothing)
        maxd          = smooth_multiple_times(max,                        smoothing)
        varianced     = smooth_multiple_times(variance,                   smoothing)
        # CLEAN : ?!
        countcd        = clean_smoothing(countd,        count)
        meancd         = clean_smoothing(meand,         mean)
        mean_sqcd      = clean_smoothing(mean_sqd,      mean_of_square)
        mean_sq_diffcd = clean_smoothing(mean_sq_diffd, mean_of_squared_difference)
        mean_diff_sqcd = clean_smoothing(mean_diff_sqd, mean_of_difference_squared)
        stdcd          = clean_smoothing(stdd,          std)
        mediancd       = clean_smoothing(mediand,       median)
        mincd          = clean_smoothing(mind,          min)
        maxcd          = clean_smoothing(maxd,          max)
        variancecd     = clean_smoothing(varianced,     variance)
        # SAME : 
        latcd          = lat
        new_loncd      = new_lon
    ''' CREATE DataArrays '''
    countc        = xr.DataArray(countcd,        dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    meanc         = xr.DataArray(meancd,         dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    mean_sqc      = xr.DataArray(mean_sqcd,      dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    mean_sq_diffc = xr.DataArray(mean_sq_diffcd, dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    mean_diff_sqc = xr.DataArray(mean_diff_sqcd, dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    stdc          = xr.DataArray(stdcd,          dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    medianc       = xr.DataArray(mediancd,       dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    minc          = xr.DataArray(mincd,          dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    maxc          = xr.DataArray(maxcd,          dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    variancec     = xr.DataArray(variancecd,     dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    latc          = xr.DataArray(latcd,          dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    new_lonc      = xr.DataArray(new_loncd,      dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    ''' CREATE DataSet '''
    dataset  = xr.Dataset({'no_nan_count'                  : countc, 
                           'mean_np_nanmean'               : meanc, 
                           'mean_of_squared'               : mean_sqc, 
                           'mean_of_difference_of_squared' : mean_sq_diffc, 
                           'mean_of_squared_difference'    : mean_diff_sqc, 
                           'np_std'                        : stdc, 
                           'median'                        : medianc, 
                           'min'                           : minc, 
                           'max'                           : maxc,
                           'variance'                      : variancec, 
                           'latitude'                      : latc, 
                           'longitude'                     : new_lonc})
    ''' RETURN THE DataSet '''
    return dataset
''' FUNCTION TO CREATE THE DATASET FOR EKE, UG, VG : '''
def store_eke_mean_ug_vg_all_cycles(merged_dataset, var_type, correction_type, mask_type, smoothing, smoothing_after):
    ''' GET THE MASKED VARIABLE, LAT, LON, AND INFORMATION ABOUT CYCLES : '''
    masked_dataset, lat, lon, month_list, DJF, MAM, JJA, SON, Winter, Summer = get_corrected_and_masked_variables(merged_dataset, var_type, correction_type, mask_type, smoothing)
    ''' CALCULATE EKE, UG_MEAN, VG_MEAN : '''
    ug_mean, vg_mean, eke_jon, eke_wrong, eke_not_removed = calculate_eke_ug_mean_vg_mean(masked_dataset, lat, lon)
    ''' Num_lines, Num_pixels '''
    num_lines  = lat.shape[0]
    num_pixels = lat.shape[1]
    ''' Fix the Longitude : '''
    new_lon = np.where(lon > 180, lon - 360, lon)
    ''' SMOOTHING : '''
    if smoothing_after == None:
        ug_meancd         = ug_mean
        vg_meancd         = vg_mean
        eke_joncd         = eke_jon
        eke_wrongcd       = eke_wrong
        eke_not_removedcd = eke_not_removed
        latcd             = lat
        new_loncd         = new_lon
    elif smoothing_after != None:
        ug_meand         = smooth_multiple_times(ug_mean,         smoothing)
        vg_meand         = smooth_multiple_times(vg_mean,         smoothing)
        eke_jond         = smooth_multiple_times(eke_jon,         smoothing)
        eke_wrongd       = smooth_multiple_times(eke_wrong,       smoothing)
        eke_not_removedd = smooth_multiple_times(eke_not_removed, smoothing)
        # CLEAN : ?!
        ug_meancd         = clean_smoothing(ug_meand,         ug_mean)
        vg_meancd         = clean_smoothing(vg_meand,         vg_mean)
        eke_joncd         = clean_smoothing(eke_jond,         eke_jon)
        eke_wrongcd       = clean_smoothing(eke_wrongd,       eke_wrong)
        eke_not_removedcd = clean_smoothing(eke_not_removedd, eke_not_removed)
        # SAME : 
        latcd          = lat
        new_loncd      = new_lon
    ''' CREATE DataArrays '''
    ug_mean          = xr.DataArray(ug_meancd,         dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    vg_mean          = xr.DataArray(vg_meancd,         dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    eke_jon          = xr.DataArray(eke_joncd,         dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    eke_wrong        = xr.DataArray(eke_wrongcd,       dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    eke_not_removed  = xr.DataArray(eke_not_removedcd, dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    lat              = xr.DataArray(latcd,             dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    new_lon          = xr.DataArray(new_loncd,         dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    ''' CREATE DataSet '''
    dataset  = xr.Dataset({'ug_mean'   : ug_mean, 
                           'vg_mean'   : vg_mean, 
                           'eke_m1'    : eke_jon, 
                           'eke_m2'    : eke_wrong, 
                           'eke_m3'    : eke_not_removed,
                           'latitude'  : lat, 
                           'longitude' : new_lon})
    ''' RETURN THE DataSet '''
    return dataset
''' FUNCTION TO CREATE THE DATASET FOR SEASONAL VARIABLES : '''
def store_eke_mean_ug_vg_seasonal(merged_dataset, var_type, correction_type, mask_type, smoothing, smoothing_after):
    ''' GET THE MASKED VARIABLE, LAT, LON, AND INFORMATION ABOUT CYCLES : '''
    masked_dataset, lat, lon, month_list, DJF, MAM, JJA, SON, Winter, Summer = get_corrected_and_masked_variables(merged_dataset, var_type, correction_type, mask_type, smoothing)
    ''' MAKE SEASONAL VARIABLES : '''
    # For now, we only use winter and summer (not DJF, MAM, JJA, SON) :
    ''' SUMMER : '''
    masked_dataset_summer = masked_dataset[Summer]
    ''' WINTER : '''
    masked_dataset_winter = masked_dataset[Winter]
    ''' CALCULATE BASIC STATISTICS SEASONAL : '''
    mean_s, mean_of_squared_difference_s, mean_of_difference_squared_s, std_s = calculate_statistics_basics(masked_dataset_summer)
    mean_w, mean_of_squared_difference_w, mean_of_difference_squared_w, std_w = calculate_statistics_basics(masked_dataset_winter)
    ''' CALCULATE EKE, UG_MEAN, VG_MEAN SEASONAL : '''
    ug_mean_s, vg_mean_s, eke_jon_s, eke_jon_o_s, eke_wrong_s, eke_not_removed_s = calculate_eke_ug_mean_vg_mean_seasonal(masked_dataset, lat, lon, Summer)
    ug_mean_w, vg_mean_w, eke_jon_w, eke_jon_o_w, eke_wrong_w, eke_not_removed_w = calculate_eke_ug_mean_vg_mean_seasonal(masked_dataset, lat, lon, Winter)
    ''' Num_lines, Num_pixels '''
    num_lines  = lat.shape[0]
    num_pixels = lat.shape[1]
    ''' Fix the Longitude : '''
    new_lon = np.where(lon > 180, lon - 360, lon)
    ''' SMOOTHING : '''
    if smoothing_after == None:
        mean_scd                          = mean_s
        mean_wcd                          = mean_w
        std_scd                           = std_s
        std_wcd                           = std_w
        mean_of_squared_difference_scd    = mean_of_squared_difference_s
        mean_of_squared_difference_wcd    = mean_of_squared_difference_w
        mean_of_difference_squared_scd    = mean_of_difference_squared_s
        mean_of_difference_squared_wcd    = mean_of_difference_squared_w
        ug_mean_scd                       = ug_mean_s
        vg_mean_scd                       = vg_mean_s
        eke_jon_scd                       = eke_jon_s
        eke_jon_o_scd                     = eke_jon_o_s
        eke_wrong_scd                     = eke_wrong_s
        eke_not_removed_scd               = eke_not_removed_s
        ug_mean_wcd                       = ug_mean_w
        vg_mean_wcd                       = vg_mean_w
        eke_jon_wcd                       = eke_jon_w
        eke_jon_o_wcd                     = eke_jon_o_w
        eke_wrong_wcd                     = eke_wrong_w
        eke_not_removed_wcd               = eke_not_removed_w
        latcd                             = lat
        new_loncd                         = new_lon
    elif smoothing_after != None:
        mean_sd                          = smooth_multiple_times(mean_s,                         smoothing)
        mean_wd                          = smooth_multiple_times(mean_w,                         smoothing)
        std_sd                           = smooth_multiple_times(std_s,                          smoothing)
        std_wd                           = smooth_multiple_times(std_w,                          smoothing)
        mean_of_squared_difference_sd    = smooth_multiple_times(mean_of_squared_difference_s,   smoothing)
        mean_of_squared_difference_wd    = smooth_multiple_times(mean_of_squared_difference_w,   smoothing)
        mean_of_difference_squared_sd    = smooth_multiple_times(mean_of_difference_squared_s,   smoothing)
        mean_of_difference_squared_wd    = smooth_multiple_times(mean_of_difference_squared_w,   smoothing)
        ug_mean_sd                       = smooth_multiple_times(ug_mean_s,                      smoothing)
        vg_mean_sd                       = smooth_multiple_times(vg_mean_s,                      smoothing)
        eke_jon_sd                       = smooth_multiple_times(eke_jon_s,                      smoothing)
        eke_jon_o_sd                     = smooth_multiple_times(eke_jon_o_s,                    smoothing)
        eke_wrong_sd                     = smooth_multiple_times(eke_wrong_s,                    smoothing)
        eke_not_removed_sd               = smooth_multiple_times(eke_not_removed_s,              smoothing)
        ug_mean_wd                       = smooth_multiple_times(ug_mean_w,                      smoothing)
        vg_mean_wd                       = smooth_multiple_times(vg_mean_w,                      smoothing)
        eke_jon_wd                       = smooth_multiple_times(eke_jon_w,                      smoothing)
        eke_jon_o_wd                     = smooth_multiple_times(eke_jon_o_w,                    smoothing)
        eke_wrong_wd                     = smooth_multiple_times(eke_wrong_w,                    smoothing)
        eke_not_removed_wd               = smooth_multiple_times(eke_not_removed_w,              smoothing)
        # CLEAN : ?!
        mean_scd                          = clean_smoothing(mean_sd,                          mean_s)
        mean_wcd                          = clean_smoothing(mean_wd,                          mean_w)
        std_scd                           = clean_smoothing(std_sd,                           std_s)
        std_wcd                           = clean_smoothing(std_wd,                           std_w)
        mean_of_squared_difference_scd    = clean_smoothing(mean_of_squared_difference_sd,    mean_of_squared_difference_s)
        mean_of_squared_difference_wcd    = clean_smoothing(mean_of_squared_difference_wd,    mean_of_squared_difference_w)
        mean_of_difference_squared_scd    = clean_smoothing(mean_of_difference_squared_sd,    mean_of_difference_squared_s)
        mean_of_difference_squared_wcd    = clean_smoothing(mean_of_difference_squared_wd,    mean_of_difference_squared_w)
        ug_mean_scd                       = clean_smoothing(ug_mean_sd,                       ug_mean_s)
        vg_mean_scd                       = clean_smoothing(vg_mean_sd,                       vg_mean_s)
        eke_jon_scd                       = clean_smoothing(eke_jon_sd,                       eke_jon_s)
        eke_jon_o_scd                     = clean_smoothing(eke_jon_o_sd,                     eke_jon_o_s)
        eke_wrong_scd                     = clean_smoothing(eke_wrong_sd,                     eke_wrong_s)
        eke_not_removed_scd               = clean_smoothing(eke_not_removed_sd,               eke_not_removed_s)
        ug_mean_wcd                       = clean_smoothing(ug_mean_wd,                       ug_mean_w)
        vg_mean_wcd                       = clean_smoothing(vg_mean_wd,                       vg_mean_w)
        eke_jon_wcd                       = clean_smoothing(eke_jon_wd,                       eke_jon_w)
        eke_jon_o_wcd                     = clean_smoothing(eke_jon_o_wd,                     eke_jon_o_w)
        eke_wrong_wcd                     = clean_smoothing(eke_wrong_wd,                     eke_wrong_w)
        eke_not_removed_wcd               = clean_smoothing(eke_not_removed_wd,               eke_not_removed_w)
        # SAME :
        latcd                             = lat
        new_loncd                         = new_lon
    ''' CREATE DataArrays '''
    mean_s                       = xr.DataArray(mean_scd,                       dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    mean_w                       = xr.DataArray(mean_wcd,                       dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    std_s                        = xr.DataArray(std_scd,                        dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    std_w                        = xr.DataArray(std_wcd,                        dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    mean_of_squared_difference_s = xr.DataArray(mean_of_squared_difference_scd, dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    mean_of_squared_difference_w = xr.DataArray(mean_of_squared_difference_wcd, dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    mean_of_difference_squared_s = xr.DataArray(mean_of_difference_squared_scd, dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    mean_of_difference_squared_w = xr.DataArray(mean_of_difference_squared_wcd, dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    ug_mean_s                    = xr.DataArray(ug_mean_scd,                    dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    vg_mean_s                    = xr.DataArray(vg_mean_scd,                    dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    eke_jon_s                    = xr.DataArray(eke_jon_scd,                    dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    eke_jon_o_s                  = xr.DataArray(eke_jon_o_scd,                  dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    eke_wrong_s                  = xr.DataArray(eke_wrong_scd,                  dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    eke_not_removed_s            = xr.DataArray(eke_not_removed_scd,            dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    ug_mean_w                    = xr.DataArray(ug_mean_wcd,                    dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    vg_mean_w                    = xr.DataArray(vg_mean_wcd,                    dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    eke_jon_w                    = xr.DataArray(eke_jon_wcd,                    dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    eke_jon_o_w                  = xr.DataArray(eke_jon_o_wcd,                  dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    eke_wrong_w                  = xr.DataArray(eke_wrong_wcd,                  dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    eke_not_removed_w            = xr.DataArray(eke_not_removed_wcd,            dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    lat                          = xr.DataArray(latcd,                          dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    new_lon                      = xr.DataArray(new_loncd,                      dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    ''' CREATE DataSet '''
    dataset  = xr.Dataset({ 'mean_s'                       : mean_s, 
                            'mean_w'                       : mean_w,
                            'std_s'                        : std_s,
                            'std_w'                        : std_w,
                            'mean_of_squared_difference_s' : mean_of_squared_difference_s,
                            'mean_of_squared_difference_w' : mean_of_squared_difference_w,
                            'mean_of_difference_squared_s' : mean_of_difference_squared_s,
                            'mean_of_difference_squared_w' : mean_of_difference_squared_w,
                            'ug_mean_s'                    : ug_mean_s,
                            'vg_mean_s'                    : vg_mean_s,
                            'eke_m1_s'                     : eke_jon_s,
                            'eke_m1_o_s'                   : eke_jon_o_s,
                            'eke_m2_s'                     : eke_wrong_s,
                            'eke_m3_s'                     : eke_not_removed_s,
                            'ug_mean_w'                    : ug_mean_w,
                            'vg_mean_w'                    : vg_mean_w,
                            'eke_m1_w'                     : eke_jon_w,
                            'eke_m1_o_w'                   : eke_jon_o_w,
                            'eke_m2_w'                     : eke_wrong_w,
                            'eke_m3_w'                     : eke_not_removed_w,
                            'latitude'                     : lat, 
                            'longitude'                    : new_lon})
    ''' RETURN THE DataSet '''
    return dataset
''' FUNCTION TO CREATE THE DATASET FOR VARIANCE ELLIPSE : '''
def store_cuu_cvv_cuv_a_b_tehta(merged_dataset, var_type, correction_type, mask_type, smoothing, smoothing_after):
    ''' GET THE MASKED VARIABLE, LAT, LON, AND INFORMATION ABOUT CYCLES : '''
    masked_dataset, lat, lon, month_list, DJF, MAM, JJA, SON, Winter, Summer = get_corrected_and_masked_variables(merged_dataset, var_type, correction_type, mask_type, smoothing)
    ''' MAKE VARIABLES : '''
    cuu, cvv, cuv, detc, trc, a, b, theta = calculate_cuu_cvv_cuv(masked_dataset, lat, lon)
    ''' Fix the Longitude : '''
    new_lon = np.where(lon > 180, lon - 360, lon)
    ''' Num_lines, Num_pixels '''
    num_lines  = lat.shape[0]
    num_pixels = lat.shape[1]
    ''' SMOOTHING : '''
    if smoothing_after == None:
        cuucd     = cuu
        cvvcd     = cvv
        cuvcd     = cuv
        detccd    = detc
        trccd     = trc
        acd       = a
        bcd       = b
        thetacd   = theta
        latcd     = lat
        new_loncd = new_lon
    elif smoothing_after != None:
        cuud    = smooth_multiple_times(cuu,   smoothing)
        cvvd    = smooth_multiple_times(cvv,   smoothing)
        cuvd    = smooth_multiple_times(cuv,   smoothing)
        detcd   = smooth_multiple_times(detc,  smoothing)
        trcd    = smooth_multiple_times(trc,   smoothing)
        ad      = smooth_multiple_times(a,     smoothing)
        bd      = smooth_multiple_times(b,     smoothing)
        thetad  = smooth_multiple_times(theta, smoothing)
        # CLEAN : ?!
        cuucd   = clean_smoothing(cuud,   cuu)
        cvvcd   = clean_smoothing(cvvd,   cvv)
        cuvcd   = clean_smoothing(cuvd,   cuv)
        detccd  = clean_smoothing(detcd,  detc)
        trccd   = clean_smoothing(trcd,   trc)
        acd     = clean_smoothing(ad,     a)
        bcd     = clean_smoothing(bd,     b)
        thetacd = clean_smoothing(thetad, theta)
        # SAME : 
        latcd          = lat
        new_loncd      = new_lon
    ''' CREATE DataArrays '''
    cuu     = xr.DataArray(cuucd,     dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    cvv     = xr.DataArray(cvvcd,     dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    cuv     = xr.DataArray(cuvcd,     dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    detc    = xr.DataArray(detccd,    dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    trc     = xr.DataArray(trccd,     dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    a       = xr.DataArray(acd,       dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    b       = xr.DataArray(bcd,       dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    theta   = xr.DataArray(thetacd,   dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    lat     = xr.DataArray(latcd,     dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    new_lon = xr.DataArray(new_loncd, dims=('num_lines', 'num_pixels'), coords={'num_lines': np.arange(num_lines), 'num_pixels': np.arange(num_pixels)})
    ''' CREATE DataSet '''
    dataset  = xr.Dataset({ 'cuu'         : cuu, 
                            'cvv'         : cvv,
                            'cuv'         : cuv,
                            'detc'        : detc,
                            'trc'         : trc,
                            'a'           : a,
                            'b'           : b,
                            'theta'       : theta,
                            'latitude'    : lat,
                            'longitude'   : new_lon})
    ''' RETURN THE DataSet '''
    return dataset
''' FUNCTION TO CREATE AND STORE THE FILES : '''
def create_and_store_files(path_files_merged, path_save_files_stat, path_save_files_eke, path_save_files_seas, pass_number, var_type, correction_type, mask_type, smoothing, smoothing_after):
    ''' Open the merged file : '''
    merged_dataset = xr.open_dataset(path_files_merged + '/' + pass_number + '.nc')
    ''' Create the DataSet : '''
    dataset_stat     = store_all_statistics_all_cycles(merged_dataset, var_type, correction_type, mask_type, smoothing, smoothing_after)
    dataset_eke      = store_eke_mean_ug_vg_all_cycles(merged_dataset, var_type, correction_type, mask_type, smoothing, smoothing_after)
    dataset_seasonal = store_eke_mean_ug_vg_seasonal(merged_dataset, var_type,   correction_type, mask_type, smoothing, smoothing_after)
    ''' Save the DataSet : '''
    if smoothing != None:
        sm_name = str(smoothing)
    else:
        sm_name = '0'
    dataset_stat.to_netcdf(    path_save_files_stat + '/smoothed_{}/'.format(sm_name) + 'statistics_' + var_type + '_' + correction_type + '_' + mask_type + '_smoothed_' + sm_name + '_' +  pass_number + '.nc')
    dataset_eke.to_netcdf(     path_save_files_eke  + '/smoothed_{}/'.format(sm_name) + 'eke_'        + var_type + '_' + correction_type + '_' + mask_type + '_smoothed_' + sm_name + '_' +  pass_number + '.nc')
    dataset_seasonal.to_netcdf(path_save_files_seas + '/smoothed_{}/'.format(sm_name) + 'seasonal_'   + var_type + '_' + correction_type + '_' + mask_type + '_smoothed_' + sm_name + '_' +  pass_number + '.nc')
    ''' Print the file name : '''
    print('statistics_' + var_type + '_' + correction_type + '_' + mask_type + '_smoothed_' + sm_name + '_' +  pass_number + '.nc')
    print('eke_'        + var_type + '_' + correction_type + '_' + mask_type + '_smoothed_' + sm_name + '_' +  pass_number + '.nc')
    print('seasonal_'   + var_type + '_' + correction_type + '_' + mask_type + '_smoothed_' + sm_name + '_' +  pass_number + '.nc')
''' FUNCTION TO CREATE AND STORE THE FILES : '''
def create_and_store_files_var_eli(path_files_merged, path_save_files_eli, pass_number, var_type, correction_type, mask_type, smoothing, smoothing_after):
    ''' Open the merged file : '''
    merged_dataset = xr.open_dataset(path_files_merged + '/' + pass_number + '.nc')
    ''' Create the DataSet : '''
    dataset_var_eli = store_cuu_cvv_cuv_a_b_tehta(merged_dataset, var_type, correction_type, mask_type, smoothing, smoothing_after)
    ''' Save the DataSet : '''
    if smoothing != None:
        sm_name = str(smoothing)
    else:
        sm_name = '0'
    dataset_var_eli.to_netcdf(path_save_files_eli + '/smoothed_{}/'.format(sm_name) + 'var_eli_' + var_type + '_' + correction_type + '_' + mask_type + '_smoothed_' + sm_name + '_' +  pass_number + '.nc')
    ''' Print the file name : '''
    print('var_eli_' + var_type + '_' + correction_type + '_' + mask_type + '_smoothed_' + sm_name + '_' +  pass_number + '.nc')


''' FUNCTION TO MAKE BINNED STATISTICS : '''
def make_binned_statistics_from_unraveled_df_inval(df, variable_name, inval, left_lon, right_lon, down_lat, up_lat, delta_lat_lon=0.2, statistic='mean'):
    # GET LAT, LON AND VARIABLE VALUES :
    lat = df['lat'].values
    lon = df['lon'].values
    var = df[variable_name].values
    # GET RID OF NAN VALUES :
    lat = lat[inval]
    lon = lon[inval]
    var = var[inval]
    # DEFINE BINS :
    dlatlon = delta_lat_lon
    lonbins = np.arange(left_lon, right_lon, dlatlon)
    latbins = np.arange(down_lat, up_lat, dlatlon)
    # COMPUTE STATISTICS :
    mean_var = binned_statistic_2d(lat, lon, var, bins=[latbins, lonbins], statistic='mean')
    # RETURN :
    return mean_var, lonbins, latbins

''' FUNCTION TO GET <MEAN_U> :'''
def calculate_spatial_and_temporal_u_and_v(masked_dataset, lat, lon):
    ''' CALCULATE UG AND VG : '''
    ug, vg = calculate_ug_vg(masked_dataset, lat, lon)
    ''' TEMPORAL MEAN OF ug AND vg : '''
    ug_temp_mean = np.nanmean(ug, axis=0)
    vg_temp_mean = np.nanmean(vg, axis=0)
    ''' SPATIAL MEAN OF ug AND vg : '''
    # UNRAVEL VALUES : 
    lat_unravel = np.ravel(lat)
    lon_unravel = np.ravel(lon)
    ug_temp_mean_unravel = np.ravel(ug_temp_mean)
    vg_temp_mean_unravel = np.ravel(vg_temp_mean)
    # GET RID OF NAN VALUES :
    inval_ug = ~np.isnan(ug_temp_mean_unravel)
    inval_vg = ~np.isnan(vg_temp_mean_unravel)
    lat_ug = lat_unravel[inval_ug]
    lon_ug = lon_unravel[inval_ug]
    var_ug = ug_temp_mean_unravel[inval_ug]
    lat_vg = lat_unravel[inval_vg]
    lon_vg = lon_unravel[inval_vg]
    var_vg = vg_temp_mean_unravel[inval_vg]
    return 


