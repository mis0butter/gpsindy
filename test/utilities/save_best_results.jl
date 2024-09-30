using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 
using CSV, DataFrames 

## ============================================ ##

# for freq = 5 hz 

freq_hz = 50  
noise   = 0.02     
σn      = 0.2   
opt_σn  = false 

interpolate_gp = false  

save_best_results( freq_hz, noise, σn, opt_σn, interpolate_gp ) 


## ============================================ ##

# go through all files in result folder 
final_results_path = string( "test/results/jake_car_csvs_ctrlshift_no_trans/" )  

results_folders = readdir( final_results_path ) 
header = [ "rmse", "hz", "noise", "rollout", "method", "lambda" ] 
somi_df = DataFrame( fill( [], length(header) ), header ) 

for i in eachindex( results_folders )

    folder = results_folders[i] 
    folder_string = split( folder, "_" ) 
    
    # get freq and hz strings 
    freq_hz = replace( folder_string[1], "hz" => "" )
    noise   = folder_string[3] 

    println( "folder = ", folder ) 

    somi_df = push_somi_df( somi_df, final_results_path, folder, freq_hz, noise, "sindy" ) 
    somi_df = push_somi_df( somi_df, final_results_path, folder, freq_hz, noise, "gpsindy" ) 

end 

# save data frame 
CSV.write( "somi_df.csv", somi_df ) 


## ============================================ ##

function push_somi_df( somi_df, final_results_path, folder, freq_hz, noise, method ) 
    
    sindy_df = CSV.read( string( final_results_path, folder, "/df_min_err_csvs_", method, ".csv" ), DataFrame ) 
    
    # rollout 
    rollout_vec = sindy_df.csv_file 
    for j in eachindex( rollout_vec ) 
        rollout_vec[j] = replace( rollout_vec[j], ".csv" => "" ) 
        rollout_vec[j] = replace( rollout_vec[j], "rollout_" => "" ) 
    end 

    # rmse 
    rmse_vec = sindy_df.test_err 

    # hz 
    freq_vec = fill( freq_hz, length(rollout_vec) ) 

    # noise 
    noise_vec = fill( noise, length(rollout_vec) ) 

    # method 
    method_vec = fill( method, length(rollout_vec) ) 

    # lambda 
    lambda_vec = sindy_df.λ_min  

    # create dataframe 
    data = [ rmse_vec freq_vec noise_vec rollout_vec method_vec lambda_vec ]

    # add data to somi_df 
    for i in 1:size(data, 1) 
        push!( somi_df, data[i, :] ) 
    end 

    return somi_df 
end 


function save_best_results( freq_hz, noise, σn, opt_σn, interpolate_gp ) 

    if interpolate_gp == true 
        csv_path      = "test/results/jake_car_csvs_ctrlshift_no_trans/interpolate_gp/" 
    else 
        csv_path      = "test/results/jake_car_csvs_ctrlshift_no_trans/no_intp/" 
    end 
    csv_folder    = string( freq_hz, "hz_noise_", noise, "_σn_", σn, "_opt_", opt_σn )  
    
    # move the data into final_results folder 
    final_results_path = string( "test/results/jake_car_csvs_ctrlshift_no_trans/final_results/", csv_folder )  
    if !isdir( final_results_path ) 
        mkdir( final_results_path ) 
    end 
    
    # read the csv file 
    csv_path_file = string( csv_path, csv_folder, "/dfs/df_min_err_csvs_gpsindy.csv" ) 
    df = CSV.read( csv_path_file, DataFrame ) 
    CSV.write( string( final_results_path, "/df_min_err_csvs_gpsindy.csv" ), df ) 
    
    # save the data 
    csv_path_file = string( csv_path, csv_folder, "/dfs/df_min_err_csvs_sindy.csv" ) 
    df = CSV.read( csv_path_file, DataFrame ) 
    CSV.write( string( final_results_path, "/df_min_err_csvs_sindy.csv" ), df ) 

end 

