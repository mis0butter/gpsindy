using GaussianSINDy 

# generate data 
fn = predator_prey 
data_train, data_test, data_train_stand = ode_train_test( fn, 0.01, 1 ) 

## ============================================ ##
# SINDy vs GPSINDy 

λ = 0.1 

# run SINDy (STLS) 
Ξ_true_stls       = sindy_stls( data_train.x_true, data_train.dx_true, λ, data_train.u ) 
Ξ_true_stls_terms = pretty_coeffs( Ξ_true_stls, data_train.x_true, data_train.u ) 
Ξ_sindy_stls, Ξ_sindy_lasso, Ξ_gpsindy, Ξ_sindy_stls_terms, Ξ_sindy_lasso_terms, Ξ_gpsindy_terms = gpsindy_Ξ_fn( data_train_stand.t, data_train_stand.x_true, data_train_stand.dx_true, λ, data_train_stand.u ) 

Ξ_nn_lasso = nn_Ξ_fn( data_train_stand.dx_noise, data_train_stand.x_noise, λ ) 


## ============================================ ##
# validate 

x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 

dx_fn_nn          = build_dx_fn( poly_order, x_vars, u_vars, Ξ_nn_lasso ) 
dx_fn_true        = build_dx_fn( poly_order, x_vars, u_vars, Ξ_true_stls ) 
dx_fn_sindy_stls  = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_stls ) 
dx_fn_sindy_lasso = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_lasso ) 
dx_fn_gpsindy     = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 

x0 = data_test.x_true[1,:] 
# x_unicycle_test    = integrate_euler_unicycle( fn, x0, data_test.t, data_test.u ) 
# t, x_int_tsit5_true = validate_data( data_test.t, data_test.x_true, dx_fn_true, t[2] - t[1] ) 
x_int_euler_true   = integrate_euler( dx_fn_true, x0, data_test.t, data_test.u ) 
x_sindy_stls_test  = integrate_euler( dx_fn_sindy_stls, x0, data_test.t, data_test.u ) 
x_sindy_lasso_test = integrate_euler( dx_fn_sindy_lasso, x0, data_test.t, data_test.u ) 
x_nn_test          = integrate_euler( dx_fn_nn, x0, data_test.t, data_test.u ) 
x_gpsindy_test     = integrate_euler( dx_fn_gpsindy, x0, data_test.t, data_test.u ) 

t_test = data_test.t 

# ----------------------- # 
# plot smoothed data and validation test data 

plot_validation_test( t_test, data_test.x_true, x_sindy_stls_test, x_sindy_lasso_test, x_nn_test, x_gpsindy_test ) 





