# Dictionaries for plotting decadal forecasts/hindcasts

# Complete list of models used
models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1"]

# Models used in Marcheggiani et al. (2023)
marcheg_models = ['BCC-CSM2-MR', 'MPI-ESM1-2-HR', 'CanESM5', 'CMCC-CM2-SR5', 'HadGEM3-GC31-MM', 'EC-Earth3', 'MIROC6', 'IPSL-CM6A-LR', 'CESM1-1-CAM5-CMIP5', 'NorCPM1']

# CMIP6 models used in Smith et al. (2020)
smith_cmip6_models = [ 'MPI-ESM1-2-HR', 'HadGEM3-GC31-MM', 'EC-Earth3', 'MIROC6', 'IPSL-CM6A-LR', 'CESM1-1-CAM5-CMIP5', 'NorCPM1']

# JASMIN directory for individual ensemble members
JASMIN_ind_dir = "/home/users/benhutch/multi-model/ind-members"

# File path for the observations from ERA5
# Processed using CDO manipulation
obs = "/home/users/benhutch/ERA5_psl/nao-anomaly/nao-anomaly-ERA5.8yrRM.nc"

# long obs
obs_long = "/home/users/benhutch/multi-model/multi-model-jasmin/NAO_index_8yrRM_long.nc"

# Directory for plots
plots_dir = "/home/users/benhutch/multi-model/plots/"