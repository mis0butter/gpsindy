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
