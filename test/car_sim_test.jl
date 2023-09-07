using GaussianSINDy 


## ============================================ ##
# truth 

fn = unicycle 
data_train, data_test = ode_train_test( fn ) 

λ = 0.1 
Ξ_true = SINDy_test( data_train.x_true, data_train.dx_true, λ, data_train.u ) 
Ξ_true_terms = pretty_coeffs(Ξ_true, data_train.x_true, data_train.u) 

Ξ_sindy = SINDy_test( data_train.x_noise, data_train.dx_noise, λ, data_train.u ) 
Ξ_sindy_terms = pretty_coeffs(Ξ_sindy, data_train.x_noise, data_train.u) 

# GPSINDy 
x_GP  = gp_post( data_train.t, 0*data_train.x_noise, data_train.t, 0*data_train.x_noise, data_train.x_noise ) 
dx_GP = gp_post( x_GP, 0*data_train.dx_noise, x_GP, 0*data_train.dx_noise, data_train.dx_noise ) 
Ξ_gpsindy = SINDy_test( x_GP, dx_GP, λ, data_train.u ) 
Ξ_gpsindy_terms = pretty_coeffs(Ξ_gpsindy, x_GP, data_train.u) 


## ============================================ ##
# validate 

x_vars = size( data_train.x_true, 2 ) 
u_vars = size( data_train.u, 2 ) 
poly_order = x_vars 

dx_fn_true       = build_dx_fn( poly_order, x_vars, u_vars, Ξ_true ) 
dx_fn_sindy      = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy ) 
dx_fn_gpsindy    = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 

xu0 = data_train.x_true[1,:] 
push!( xu0, data_train.u[1,1] ) 
push!( xu0, data_train.u[1,2] ) 
dx0_test = dx_fn_true( xu0, 0, 0 ) 

x_true_test    = integrate_euler( dx_fn_true, data_test.x_true[1,:], data_test.t, data_test.u ) 
x_sindy_test   = integrate_euler( dx_fn_sindy, data_test.x_true[1,:], data_test.t, data_test.u ) 
x_gpsindy_test = integrate_euler( dx_fn_gpsindy, data_test.x_true[1,:], data_test.t, data_test.u ) 
t_test = data_test.t 

using Plots 

i = 1 
plot( t_test, data_test.x_true[:,1], label = "true", legend = true ) 
plot!( t_test, x_sindy_test[:,1], ls = :dash, label = "sindy" ) 
plot!( t_test, x_gpsindy_test[:,1], ls = :dashdot, label = "gpsindy" ) 


# dt = data_train.t[2] - data_train.t[1] 
# t_sindy_val,   x_sindy_val   = validate_data( data_test.t, [ data_test.x_noise data_test.u ], dx_sindy_fn, dt ) 
# t_gpsindy_val, x_gpsindy_val = validate_data( data_test.t, x_GP_train, dx_gpsindy_fn, dt ) 








