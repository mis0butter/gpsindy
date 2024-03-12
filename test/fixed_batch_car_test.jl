using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 


## ============================================ ##
# test all the files  

freq_hz  = 50 

df_λ_vec_hist     = [] 
mean_5hz_err_hist = x_train_test_err_struct( [], [], [], [] ) 
for noise = 0.01 : 0.01 : 0.1 

    csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

    x_min_err_hist, df_λ_vec = cross_validate_csv_path( csv_path, freq_hz ) 
    
    sindy_train_err_mean   = mean( x_min_err_hist.sindy_train   ) 
    sindy_test_err_mean    = mean( x_min_err_hist.sindy_test    ) 
    gpsindy_train_err_mean = mean( x_min_err_hist.gpsindy_train ) 
    gpsindy_test_err_mean  = mean( x_min_err_hist.gpsindy_test  )     

    push!( df_λ_vec_hist, df_λ_vec ) 
    push!( mean_5hz_err_hist.sindy_train,   sindy_train_err_mean    ) 
    push!( mean_5hz_err_hist.sindy_test,    sindy_test_err_mean     ) 
    push!( mean_5hz_err_hist.gpsindy_train, gpsindy_train_err_mean  ) 
    push!( mean_5hz_err_hist.gpsindy_test,  gpsindy_test_err_mean   ) 

end 


## ============================================ ##

freq_hz  = 10 

mean_10hz_err_hist = x_train_test_err_struct( [], [], [], [] ) 
for noise = 0.01 : 0.01 : 0.1 

    csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

    x_min_err_hist = cross_validate_csv_path( csv_path ) 
    
    sindy_train_err_mean   = mean( x_min_err_hist.sindy_train   ) 
    sindy_test_err_mean    = mean( x_min_err_hist.sindy_test    ) 
    gpsindy_train_err_mean = mean( x_min_err_hist.gpsindy_train ) 
    gpsindy_test_err_mean  = mean( x_min_err_hist.gpsindy_test  )     

    push!( mean_10hz_err_hist.sindy_train,   sindy_train_err_mean   ) 
    push!( mean_10hz_err_hist.sindy_test,    sindy_test_err_mean    ) 
    push!( mean_10hz_err_hist.gpsindy_train, gpsindy_train_err_mean ) 
    push!( mean_10hz_err_hist.gpsindy_test,  gpsindy_test_err_mean  ) 

end 









## ============================================ ##
# single car test 

freq_hz = 25 
csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz/" )

x_min_err_hist_25hz_σn = cross_validate_csv_path( csv_path )





