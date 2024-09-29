using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 

using Optim 
using LineSearches  
using GaussianProcesses 
using DataFrames 


## ============================================ ## 

csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/50hz_noise_0/rollout_3.csv" 

df_best_sindy, df_best_gpsindy, fig_csv = cross_validate_csv( csv_path_file, 1 ) 
fig_csv 

## ============================================ ## 


# df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.01 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.04 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.05 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.06 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.07 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.08 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.09 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.1 ) 


## ============================================ ## 
## ============================================ ## 
# save results 


# go through all files in result folder 
results_path    = "test/results/jake_car_csvs_ctrlshift_no_trans/" 

somi_df = create_somi_df( results_path ) 

# save data frame 
CSV.write( string( results_path, "somi_df.csv" ), somi_df ) 






