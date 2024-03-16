using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 
using CSV, DataFrames 


## ============================================ ##
## ============================================ ##
# let's look at 50 hz noise = 0.02 rollout_8.csv 

freq_hz = 5 
noise   = 0 
σn      = 0.02 
opt_σn  = false 
    

    df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, df_mean_err = cross_validate_csv_path( freq_hz, noise, σn, opt_σn ) 


## ============================================ ## 

for freq_hz = 5 

    noise_vec = [] 
    push!( noise_vec, 0 ) 
    push!( noise_vec, 0.01 ) 
    push!( noise_vec, 0.02 ) 

    for noise = noise_vec 

        for σn = [ 0.001, 0.002, 0.003, 0.01, 0.02, 0.03, 0.1, 0.2, 0.3 ] 
            
            opt_σn  = false  
            df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, df_mean_err = cross_validate_csv_path( freq_hz, noise, σn, opt_σn ) 
        
            opt_σn  = true   
            df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, df_mean_err = cross_validate_csv_path( freq_hz, noise, σn, opt_σn ) 
        
        end 

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

















## ============================================ ##
 
# let's break it out 

freq_hz = 5 
noise   = 0  
σn      = 0.02 
opt_σn  = false  

csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

csv_files_vec, save_path, save_path_fig, save_path_dfs = mkdir_save_path_σn( csv_path, σn, opt_σn ) 

i_csv = 1 

csv_path_file = csv_files_vec[i_csv] 

# extract data 
data_train, data_test = car_data_struct( csv_path_file ) 

# x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test, σn, opt_σn ) 
t_train_double, u_train_GP, x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_double_test( data_train, data_test, σn, opt_σn ) 

# cross-validate gpsindy 
λ_vec      = λ_vec_fn() 

i_λ = 1 
λ = λ_vec[i_λ] 

# sindy!!! 
x_sindy_train, x_sindy_test = sindy_lasso_int( data_train.x_noise, data_train.dx_noise, λ, data_train, data_test ) 

# gpsindy!!! 
# x_gpsindy_train, x_gpsindy_test = sindy_lasso_int( x_train_GP, dx_train_GP, λ, data_train, data_test ) 
x_gpsindy_train, x_gpsindy_test = gpsindy_lasso_int( x_train_GP, dx_train_GP, u_train_GP, λ, data_train, data_test )  

