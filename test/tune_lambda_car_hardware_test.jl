using GaussianSINDy 


## ============================================ ##
# setup 

# load data 
csv_file = "test/data/jake_robot_data.csv" 

data_train, data_test = car_data_struct( csv_file ) 

## ============================================ ##
# smooth with GPs 

x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 

## ============================================ ##

# get x0 from smoothed data 
x0_train_GP = x_train_GP[1,:] 
x0_test_GP  = x_test_GP[1,:] 

# SINDy (STLS) 
λ = 0.1 
Ξ_sindy_stls  = sindy_stls( data_train.x_noise, data_train.dx_noise, λ, false, data_train.u ) 

## ============================================ ##

# build dx_fn from Ξ and integrate 
x_train_sindy, x_test_sindy = dx_Ξ_integrate( data_train, data_test, Ξ_sindy_stls, x0_train_GP, x0_test_GP )

## ============================================ ##

λ_vec = λ_vec_fn() 

# cross-validate !!!  
Ξ_gpsindy_vec, err_x_vec, err_dx_vec = cross_validate_λ( data_train.t, x_train_GP, dx_train_GP, data_train.u, λ_vec ) 


## ============================================ ## 

# save ξ with smallest x error  
Ξ_gpsindy_minerr = Ξ_minerr( Ξ_gpsindy_vec, err_x_vec ) 

# build dx_fn from Ξ and integrate 
x_train_gpsindy, x_test_gpsindy = dx_Ξ_integrate( data_train, data_test, Ξ_gpsindy_minerr, x0_train_GP, x0_test_GP )



## ============================================ ##
# PLOT 

# plot training 
plot_noise_sindy_gpsindy( data_train.t, data_train.x_noise, x_train_GP, x_train_sindy, x_train_gpsindy, "training data" ) 

# plot testing 
plot_noise_sindy_gpsindy( data_test.t, data_test.x_noise, x_test_GP, x_test_sindy, x_test_gpsindy, "testing data" ) 

