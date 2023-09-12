using GaussianSINDy 

# generate data 
fn = predator_prey 
data_train, data_test = ode_train_test( fn, 0.01, 1 ) 


## ============================================ ##
# SINDy vs GPSINDy 

λ = 0.1 
Ξ_sindy_stls, Ξ_sindy_lasso, Ξ_gpsindy, Ξ_sindy_stls_terms, Ξ_sindy_lasso_terms, Ξ_gpsindy_terms = gpsindy_Ξ_fn( data_train.t, data_train.x_true, data_train.dx_true, λ, data_train.u ) 

## ============================================ ##

t_train, u_train, x_train_true, dx_train_true, x_train, dx_train, t_test, u_test, x_test_true, dx_test_true, x_test_noise, dx_test_noise = datastruct_to_train_test( data_train, data_test ) 

## ============================================ ##

    # get sizes 
    x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x_train, u_train ) 

    # GP smooth data 
    x_GP_train      = gp_post( t_train, 0*x_train, t_train, 0*x_train, x_train ) 
    dx_GP_train     = gp_post( x_GP_train, 0*dx_train, x_GP_train, 0*dx_train, dx_train ) 

    # run SINDy (STLS) 
    Ξ_sindy_stls        = sindy_lasso( x_train, dx_train, λ, u_train ) 
    Ξ_sindy_stls_terms  = pretty_coeffs( Ξ_sindy_stls, x_train, u_train ) 

    # run SINDy (LASSO) 
    Ξ_sindy_lasso       = sindy_lasso( x_train, dx_train, λ, u_train ) 
    Ξ_sindy_lasso_terms = pretty_coeffs( Ξ_sindy_lasso, x_train, u_train ) 

    # run GPSINDy 
    # Θx_gpsindy      = pool_data_test( [ x_GP_train u_train ], n_vars, poly_order ) 
    Ξ_gpsindy       = sindy_lasso( x_GP_train, dx_GP_train, λ, u_train ) 
    Ξ_gpsindy_terms = pretty_coeffs( Ξ_gpsindy, x_GP_train, u_train ) 


## ============================================ ##
# validate 

x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 

dx_fn_true        = build_dx_fn( poly_order, x_vars, u_vars, Ξ_true_lasso ) 
dx_fn_sindy_stls  = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_stls ) 
dx_fn_sindy_lasso = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_lasso ) 
dx_fn_gpsindy     = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 

x0 = data_test.x_true[1,:] 
# x_unicycle_test    = integrate_euler_unicycle( fn, x0, data_test.t, data_test.u ) 
x_int_euler_true   = integrate_euler( dx_fn_true, x0, data_test.t, data_test.u ) 
x_sindy_stls_test  = integrate_euler( dx_fn_sindy_stls, x0, data_test.t, data_test.u ) 
x_sindy_lasso_test = integrate_euler( dx_fn_sindy_lasso, x0, data_test.t, data_test.u ) 
x_gpsindy_test     = integrate_euler( dx_fn_gpsindy, x0, data_test.t, data_test.u ) 

t_test = data_test.t 

# ----------------------- # 
# plot smoothed data and validation test data 

plot_validation_test( t_test, data_test.x_true, x_int_euler_true, x_sindy_stls_test, x_gpsindy_test ) 





