using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 
using DataFrames 


## ============================================ ##
# test all the files  

freq_hz  = 50 

df_λ_vec_hist     = [] 
for noise = 0.01 : 0.01 : 0.02
    
    csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

    df_min_err_hist = cross_validate_csv_path( csv_path, freq_hz, true ) 
    push!( df_λ_vec_hist, df_min_err_hist ) 
end 



## ============================================ ##
# single car test 

freq_hz = 50 
noise   = 0.02 

csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

df_min_err_hist = cross_validate_csv_path( csv_path, freq_hz, true ) 



## ============================================ ##
# single car test 

freq_hz = 50 
csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz/" )

df_min_err_hist = cross_validate_csv_path( csv_path, freq_hz, true )





