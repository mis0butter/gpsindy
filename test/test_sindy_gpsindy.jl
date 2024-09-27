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


## ============================================ ## 
## ============================================ ## 
# save results 


# go through all files in result folder 
results_path    = "test/results/jake_car_csvs_ctrlshift_no_trans/" 
results_folders = readdir( results_path ) 

header = [ "rmse", "hz", "noise", "rollout", "method", "lambda" ] 
somi_df = DataFrame( fill( [], length(header) ), header ) 

for i in eachindex( results_folders )

    folder = results_folders[i] 
    folder_string = split( folder, "_" ) 
    
    # get freq and hz strings 
    freq_hz = replace( folder_string[1], "hz" => "" ) 
    noise   = folder_string[3] 

    println( "folder = ", folder ) 

    folder_path = string( results_path, folder)  
    somi_df = push_somi_df( somi_df, folder_path, freq_hz, noise, "sindy" ) 
    somi_df = push_somi_df( somi_df, folder_path, freq_hz, noise, "gpsindy" ) 

end 

# save data frame 
CSV.write( string( results_path, "somi_df.csv" ), somi_df ) 


## ============================================ ##


function push_somi_df( somi_df, folder_path, freq_hz, noise, method ) 
    
    sindy_df = CSV.read( string( folder_path, "/dfs/df_min_err_csvs_", method, ".csv" ), DataFrame ) 
    
    # rollout 
    rollouts = sindy_df.csv_file 
    for j in eachindex( rollouts ) 
        rollouts[j] = replace( rollouts[j], ".csv" => "" ) 
        rollouts[j] = replace( rollouts[j], "rollout_" => "" ) 
    end 
    N = length(rollouts)  

    rmse_vec   = sindy_df.test_err                      # rmse    
    freq_vec   = fill( freq_hz, N )   # hz 
    noise_vec  = fill( noise, N )     # noise 
    method_vec = fill( method, N )    # method 
    lambda_vec = sindy_df.λ_min                         # lambda 

    # create dataframe 
    data = [ rmse_vec freq_vec noise_vec rollouts method_vec lambda_vec ]

    # add data to somi_df 
    for i in 1:size(data, 1) 
        push!( somi_df, data[i, :] ) 
    end 

    return somi_df 
end 


