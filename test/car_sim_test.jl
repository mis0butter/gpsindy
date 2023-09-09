using GaussianSINDy 


## ============================================ ##
# truth 

fn = unicycle 
data_train, data_test = ode_train_test( fn ) 


λ = 0.1 
Ξ_true = SINDy_test( data_train.x_true, data_train.dx_true, λ, data_train.u ) 
Ξ_true_terms = pretty_coeffs(Ξ_true, data_train.x_noise, data_train.u) 
Ξ_sindy, Ξ_gpsindy, Ξ_gpsindy_x2, Ξ_sindy_terms, Ξ_gpsindy_terms, Ξ_gpsindy_x2_terms = gpsindy_Ξ_fn( data_train.t, data_train.x_noise, data_train.dx_noise, λ, data_train.u ) 


## ============================================ ##
# validate 

x_vars = size( data_train.x_true, 2 ) 
u_vars = size( data_train.u, 2 ) 
poly_order = x_vars 

dx_fn_true       = build_dx_fn( poly_order, x_vars, u_vars, Ξ_true ) 
dx_fn_sindy      = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy ) 
dx_fn_gpsindy    = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
dx_fn_gpsindy_x2 = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy_x2 ) 

xu0 = data_train.x_true[1,:] 
push!( xu0, data_train.u[1,1] ) 
push!( xu0, data_train.u[1,2] ) 
dx0_test = dx_fn_sindy( xu0, 0, 0 ) 

x0 = data_test.x_true[1,:] 
x_unicycle_test   = integrate_euler_unicycle(fn, x0, data_test.t, data_test.u) 
x_sindy_test      = integrate_euler( dx_fn_sindy, x0, data_test.t, data_test.u) 
x_gpsindy_test    = integrate_euler( dx_fn_gpsindy, x0, data_test.t, data_test.u) 
x_gpsindy_x2_test = integrate_euler( dx_fn_gpsindy_x2, x0, data_test.t, data_test.u) 

t_test = data_test.t 

# ----------------------- #

# plot smoothed data and validation test data 
plot_validation_test( t_test, data_test.x_true, x_unicycle_test, x_sindy_test, x_gpsindy_test) 


