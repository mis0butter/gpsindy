# this is a script for tuning lambda for each dx based on the error from the TRAINING DATA 

using GaussianSINDy 
using LinearAlgebra 
using Plots 

## ============================================ ##

fn = unicycle 

λ = 0.1 
data_train, data_test = ode_train_test( fn )

x_train_noise  = data_train.x_noise 
dx_train_noise = data_train.dx_noise 
u_train        = data_train.u 
t_train        = data_train.t 
x_train_true   = data_train.x_true 
dx_train_true  = data_train.dx_true 

x_test_noise   = data_test.x_noise 
dx_test_noise  = data_test.dx_noise 
u_test         = data_test.u
t_test         = data_test.t    
x_test_true    = data_test.x_true 
dx_test_true   = data_test.dx_true 

# get sizes 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x_train_noise, u_train ) 
poly_order = 4 

Ξ_true       = sindy_stls( x_train_true, dx_train_true, λ, poly_order, u_train ) 
Ξ_true_terms = pretty_coeffs( Ξ_true, x_train_true, u_train ) 


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
while λ_vec[end] < 10 
    push!( λ_vec, 10 * λ_vec[end] ) 
end 
while λ_vec[end] < 500 
    push!( λ_vec, 10 + λ_vec[end] ) 
end 

# cross-validate !!!  
Ξ_gpsindy_vec, err_x_vec, err_dx_vec = cross_validate_λ( t_train, x_train_GP, dx_train_GP, u_train ) 

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
