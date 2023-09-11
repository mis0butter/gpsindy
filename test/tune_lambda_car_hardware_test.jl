using GaussianSINDy 


## ============================================ ##
# setup 

# load data 
t, x, u = extract_car_data() 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x, u ) 
x, dx_fd = unroll( t, x ) 

# split into training and test data 
test_fraction = 0.2 
portion       = 5 
u_train,  u_test  = split_train_test( u, test_fraction, portion ) 
t_train,  t_test  = split_train_test( t, test_fraction, portion ) 
x_train_noise,  x_test_noise  = split_train_test( x, test_fraction, portion ) 
dx_train_noise, dx_test_noise = split_train_test( dx_fd, test_fraction, portion ) 


## ============================================ ##
# smooth with GPs 

# first - smooth measurements with Gaussian processes 
x_train_GP  = gp_post( t_train, 0*x_train_noise, t_train, 0*x_train_noise, x_train_noise ) 
dx_train_GP = gp_post( x_train_GP, 0*dx_train_noise, x_train_GP, 0*dx_train_noise, dx_train_noise ) 
x_test_GP   = gp_post( t_test, 0*x_test_noise, t_test, 0*x_test_noise, x_test_noise ) 
dx_test_GP  = gp_post( x_test_GP, 0*dx_test_noise, x_test_GP, 0*dx_test_noise, dx_test_noise ) 

# get x0 from smoothed data 
x0_train_GP = x_train_GP[1,:] 
x0_test_GP  = x_test_GP[1,:] 

# SINDy (STLS) 
λ = 0.1 
Ξ_sindy_stls  = sindy_stls( x_train_noise, dx_train_noise, λ, poly_order, u_train ) 

# build dx_fn from Ξ and integrate 
dx_fn_sindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_stls ) 
x_train_sindy = integrate_euler( dx_fn_sindy, x0_train_GP, t_train, u_train ) 
x_test_sindy  = integrate_euler( dx_fn_sindy, x0_test_GP, t_test, u_test ) 


## ============================================ ##

λ_vec = [ 1e-6 ] 
while λ_vec[end] < 1e-1 
    push!( λ_vec, 10 * λ_vec[end] ) 
end 
while λ_vec[end] < 1.0  
    push!( λ_vec, 0.1 + λ_vec[end] ) 
end 
while λ_vec[end] < 10 
    push!( λ_vec, 1.0 + λ_vec[end] ) 
end 
while λ_vec[end] < 100 
    push!( λ_vec, 10.0 + λ_vec[end] ) 
end

# cross-validate !!!  
Ξ_gpsindy_vec, err_x_vec, err_dx_vec = cross_validate_λ( t_train, x_train_GP, dx_train_GP, u_train ) 


## ============================================ ## 

# save ξ with smallest x error  
Ξ_gpsindy_minerr = Ξ_minerr( Ξ_gpsindy_vec, err_x_vec ) 

# build dx_fn from Ξ and integrate 
dx_fn_gpsindy_minerr = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy_minerr ) 
x_train_gpsindy = integrate_euler( dx_fn_gpsindy_minerr, x0_train_GP, t_train, u_train ) 
x_test_gpsindy  = integrate_euler( dx_fn_gpsindy_minerr, x0_test_GP, t_test, u_test )


## ============================================ ##
# PLOT 

# plot training 
plot_noise_sindy_gpsindy( t_train, x_train_noise, x_train_GP, x_train_sindy, x_train_gpsindy, "training data" ) 

# plot testing 
plot_noise_sindy_gpsindy( t_test, x_test_noise, x_test_GP, x_test_sindy, x_test_gpsindy, "testing data" ) 

