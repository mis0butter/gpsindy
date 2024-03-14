using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 
using CSV, DataFrames 


## ============================================ ##
## ============================================ ##
# let's look at 50 hz noise = 0.02 rollout_8.csv 

freq_hz = 50 
noise   = 0 

# σn      = 0.1 

for σn = [ 0.01, 0.02, 0.1, 0.2 ]
    
    opt_σn  = false  
    df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, df_mean_err = cross_validate_csv_path( freq_hz, noise, σn, opt_σn ) 

    opt_σn  = true   
    df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, df_mean_err = cross_validate_csv_path( freq_hz, noise, σn, opt_σn ) 

end 


## ============================================ ## 

freq_hz = 50 

for noise = 0.01 : 0.01 : 0.02 

    for σn = [ 0.01, 0.02, 0.1, 0.2 ] 
        
        opt_σn  = false  
        df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, df_mean_err = cross_validate_csv_path( freq_hz, noise, σn, opt_σn ) 
    
        opt_σn  = true   
        df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, df_mean_err = cross_validate_csv_path( freq_hz, noise, σn, opt_σn ) 
    
    end 

end 


## ============================================ ##
## ============================================ ##
# let's compile all the stats of the files that I just made 

csv_path_up = string( "test/results/jake_car_csvs_ctrlshift_no_trans/" ) 
csv_path_vec = readdir( csv_path_up ) 

header = [ "freq_hz", "noise", "σn", "opt_σn", "mean_err_sindy_train", "mean_err_gpsindy_train", "mean_err_sindy_test", "mean_err_gpsindy_test" ]
df_mean_err_all = DataFrame( fill( [], 8 ), header ) 
for i = eachindex(csv_path_vec) 
# i = 1 

    csv_path = string( csv_path_up, csv_path_vec[i] ) 
    csv_path_df = string( csv_path, "/df_mean_err.csv" )

    # read in the csv 
    if isfile( csv_path_df )
        df_mean_err = CSV.read( csv_path_df, DataFrame ) 
        push!( df_mean_err_all, df_mean_err[1,:] )             
    end 

end 

# save the dataframe 
CSV.write( "df_mean_err_all.csv", df_mean_err_all) 



