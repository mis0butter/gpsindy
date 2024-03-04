using GaussianSINDy 
using LinearAlgebra 
using Statistics 


## ============================================ ##
# controls adjust sparse 5 hz 

csv_path = "test/data/jake_car_csvs_control_adjust_50hz_noise_0.1/" 
img_path = "test/images/50hz_noise_0.1/" 

csv_files_vec = readdir( csv_path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( csv_path, csv_files_vec[i] ) 
end 

# check if save_path exists 
if !isdir( img_path ) 
    mkdir( img_path ) 
end 

x_err_hist_5hz  = x_err_struct( [], [], [], [] ) 
for i = eachindex( csv_files_vec ) 
# for i = [ 4 ]
    # i = 42 
    csv_file = csv_files_vec[i] 
    t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy, fig_train, fig_test = cross_validate_sindy_gpsindy( csv_file, 1 ) 

    # save fig_train 
    fig_train_save = replace( csv_file, csv_path => img_path )  
    fig_train_save = replace( fig_train_save, ".csv" => "_train.png" ) 
    save( fig_train_save, fig_train ) 

    # save fig_test 
    fig_test_save = replace( csv_file, csv_path => img_path )  
    fig_test_save = replace( fig_test_save, ".csv" => "_test.png" ) 
    save( fig_test_save, fig_test ) 
    
    push!( x_err_hist_5hz.sindy_lasso, norm( x_test_noise - x_test_sindy )  ) 
    push!( x_err_hist_5hz.gpsindy,     norm( x_test_noise - x_test_gpsindy )  ) 

end 

# find index that is equal to maximum 
findall( x_err_hist_5hz.sindy_lasso .== maximum( x_err_hist_5hz.sindy_lasso ) ) 

# reject 3-sigma outliers 
sindy_5hz_3sigma   = reject_outliers( x_err_hist_5hz.sindy_lasso ) 
gpsindy_5hz_3sigma = reject_outliers( x_err_hist_5hz.gpsindy ) 


## ============================================ ##
# controls adjust sparse 5 hz 

csv_path = "test/data/jake_car_csvs_control_adjust_10hz/" 
img_path = "test/images/10hz/" 

csv_files_vec = readdir( csv_path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( csv_path, csv_files_vec[i] ) 
end 

# check if save_path exists 
if !isdir( img_path ) 
    mkdir( img_path ) 
end 

x_err_hist_5hz  = x_err_struct( [], [], [], [] ) 
for i = eachindex( csv_files_vec ) 
# for i = [ 4 ]
    # i = 42 
    csv_file = csv_files_vec[i] 
    t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy, fig_train, fig_test = cross_validate_sindy_gpsindy( csv_file, 1 ) 

    # save fig_train 
    fig_train_save = replace( csv_file, csv_path => img_path )  
    fig_train_save = replace( fig_train_save, ".csv" => "_train.png" ) 
    save( fig_train_save, fig_train ) 

    # save fig_test 
    fig_test_save = replace( csv_file, csv_path => img_path )  
    fig_test_save = replace( fig_test_save, ".csv" => "_test.png" ) 
    save( fig_test_save, fig_test ) 
    
    push!( x_err_hist_5hz.sindy_lasso, norm( x_test_noise - x_test_sindy )  ) 
    push!( x_err_hist_5hz.gpsindy,     norm( x_test_noise - x_test_gpsindy )  ) 

end 

# find index that is equal to maximum 
findall( x_err_hist_5hz.sindy_lasso .== maximum( x_err_hist_5hz.sindy_lasso ) ) 

# reject 3-sigma outliers 
sindy_5hz_3sigma   = reject_outliers( x_err_hist_5hz.sindy_lasso ) 
gpsindy_5hz_3sigma = reject_outliers( x_err_hist_5hz.gpsindy ) 






## ============================================ ##
# test cross_validate_sindy_gpsindy  

data_train, data_test = car_data_struct( csv_file ) 

# smooth with GPs 
x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 

# get x0 from smoothed data 
x0_train_GP = x_train_GP[1,:] 
x0_test_GP  = x_test_GP[1,:] 

# cross-validate SINDy!!! 
λ_vec = λ_vec_fn() 
# Ξ_sindy_vec, err_x_sindy, err_dx_sindy = cross_validate_λ( data_train.t, data_train.x_noise, data_train.dx_noise, data_train.u, λ_vec ) 



## ============================================ ##
# test cross_validate_λ

# massage inputs 
t_train  = data_train.t 
u_train  = data_train.u 
x_train  = data_train.x_noise 
dx_train = data_train.dx_noise 

fig = plot_car_x_dx_noise_GP( t_train, x_train, dx_train, x_train_GP, dx_train_GP )  








