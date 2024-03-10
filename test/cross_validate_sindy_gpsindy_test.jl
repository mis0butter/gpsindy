using GaussianSINDy 
using LinearAlgebra 
using Statistics 


## ============================================ ##
# cross validation function 

csv_path  = "test/data/jake_car_csvs_ctrlshift/5hz/" 
save_path = "test/results/5hz dbl u_linear/" 

sigma_3sigma_mean, gpsindy_3sigma_mean = cross_validate_all_csvs( csv_path, save_path ) 


## ============================================ ##

freq_hz   = 10 
for noise = 0.01 : 0.01 : 0.04 

    csv_path  = string( "test/data/jake_car_csvs_ctrlshift/", freq_hz, "hz_noise_", noise, "/" )  
    save_path = string( "test/results/", freq_hz, "hz_noise_", noise, "_dbl_u_linear/" ) 

    println(csv_path) 
    println(save_path) 

    sigma_3sigma_mean, gpsindy_3sigma_mean = cross_validate_all_csvs( csv_path, save_path ) 

end 

# freq_hz   = 50  
# for noise = 0.01 : 0.01 : 0.01 

#     csv_path  = string( "test/data/jake_car_csvs_ctrlshift/", freq_hz, "hz_noise_", noise, "/" )  
#     save_path = string( "test/results/", freq_hz, "hz_noise_", noise, "/" ) 

#     println(csv_path) 
#     println(save_path) 

#     sigma_3sigma_mean, gpsindy_3sigma_mean = cross_validate_all_csvs( csv_path, save_path ) 

# end 





























## ============================================ ##
# run cross validation 

csv_path = "test/data/jake_car_csvs_control_adjust_5hz/" 
img_path = "test/images/5hz/" 

csv_files_vec = readdir( csv_path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( csv_path, csv_files_vec[i] ) 
end 

# check if save_path exists 
if !isdir( img_path ) 
    mkdir( img_path ) 
end 

x_err_hist  = x_err_struct( [], [], [], [] ) 
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
    
    push!( x_err_hist.sindy_lasso, norm( x_test_noise - x_test_sindy )  ) 
    push!( x_err_hist.gpsindy,     norm( x_test_noise - x_test_gpsindy )  ) 

end 

# find index that is equal to maximum 
findall( x_err_hist.sindy_lasso .== maximum( x_err_hist.sindy_lasso ) ) 

# reject 3-sigma outliers 
sindy_3sigma_mean   = mean( reject_outliers( x_err_hist.sindy_lasso ) ) 
gpsindy_3sigma_mean = mean( reject_outliers( x_err_hist.gpsindy ) ) 












## ============================================ ##
# test cross_validate_sindy_gpsindy  

csv_file = "test/data/jake_car_csvs_control_adjust_5hz/rollout_shift_5hz_1.csv" 

data_train, data_test = car_data_struct( csv_file ) 

# let's double the points 
t_train = data_train.t 
dt = t_train[2] - t_train[1] 

t_train_double = [  ] 
for i in eachindex(t_train) 

    println(i) 
    push!( t_train_double, t_train[i] ) 
    push!( t_train_double, t_train[i] + dt/2 ) 

end 

x_col, x_row = size( data_train.x_noise ) 

# first - smooth measurements with Gaussian processes 
x_train_GP  = gp_post( t_train_double, zeros(2 * x_col, x_row), data_train.t, 0*data_train.x_noise, data_train.x_noise ) 

x_test = t_train_double 
μ_prior = zeros(2 * x_col, x_row) 
x_train = data_train.t 
μ_train = 0 * data_train.x_noise 
y_train = data_train.x_noise 

f2 = Figure() 
    Axis( f2[1,1] )
        lines!( data_train.x_noise[:,1], data_train.x_noise[:,2] )
        lines!( x_train_GP[:,1], x_train_GP[:,2] )   



## ============================================ ##

dx_train_GP = gp_post( x_train_GP, zeros(x_col, x_row), x_train_GP, 0*data_train.dx_noise, data_train.dx_noise ) 
x_test_GP   = gp_post( data_test.t, 0*data_test.x_noise, data_test.t, 0*data_test.x_noise, data_test.x_noise ) 
dx_test_GP  = gp_post( x_test_GP, 0*data_test.dx_noise, x_test_GP, 0*data_test.dx_noise, data_test.dx_noise ) 



## ============================================ ##


# smooth with GPs 
# x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 
# ----------------------- #
# function gp_train_test 
# first - smooth measurements with Gaussian processes 
x_train_GP  = gp_post( data_train.t, 0*data_train.x_noise, data_train.t, 0*data_train.x_noise, data_train.x_noise ) 
dx_train_GP = gp_post( x_train_GP, 0*data_train.dx_noise, x_train_GP, 0*data_train.dx_noise, data_train.dx_noise ) 
x_test_GP   = gp_post( data_test.t, 0*data_test.x_noise, data_test.t, 0*data_test.x_noise, data_test.x_noise ) 
dx_test_GP  = gp_post( x_test_GP, 0*data_test.dx_noise, x_test_GP, 0*data_test.dx_noise, data_test.dx_noise ) 

# get x0 from smoothed data 
x0_train_GP = x_train_GP[1,:] 
x0_test_GP  = x_test_GP[1,:] 

# cross-validate SINDy!!! 
λ_vec = λ_vec_fn() 
# Ξ_sindy_vec, err_x_sindy, err_dx_sindy = cross_validate_λ( data_train.t, data_train.x_noise, data_train.dx_noise, data_train.u, λ_vec ) 



## ============================================ ##




## ============================================ ##
# test cross_validate_λ

# massage inputs 
t_train  = data_train.t 
u_train  = data_train.u 
x_train  = data_train.x_noise 
dx_train = data_train.dx_noise 

fig = plot_car_x_dx_noise_GP( t_train, x_train, dx_train, x_train_GP, dx_train_GP )  








