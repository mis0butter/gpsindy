using GaussianSINDy 
using LinearAlgebra

## ============================================ ##

# generate data 
# fn    = predator_prey  
fn    = unicycle 
noise = 0.01 
data_train, data_test, data_train_stand = ode_train_test( fn, noise ) 


## ============================================ ## 

λ = 0.1 

# truth 
Ξ_true_stls       = sindy_stls( data_train.x_true, data_train.dx_true, λ, data_train.u ) 
Ξ_true_stls_terms = pretty_coeffs( Ξ_true_stls, data_train.x_true, data_train.u ) 

# SINDy and GPSINDy 
Ξ_sindy_stls, Ξ_sindy_lasso, Ξ_gpsindy, Ξ_sindy_stls_terms, Ξ_sindy_lasso_terms, Ξ_gpsindy_terms = gpsindy_Ξ_fn( data_train_stand.t, data_train_stand.x_true, data_train_stand.dx_true, λ, data_train_stand.u ) 

# NN 
Ξ_nn_lasso = nn_Ξ_fn( data_train.dx_noise, data_train.x_noise , λ ) 

# ----------------------- # 
# validate 

x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 

dx_fn_nn          = build_dx_fn( poly_order, x_vars, u_vars, Ξ_nn_lasso ) 
dx_fn_true        = build_dx_fn( poly_order, x_vars, u_vars, Ξ_true_stls ) 
dx_fn_sindy_stls  = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_stls ) 
dx_fn_sindy_lasso = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_lasso ) 
dx_fn_gpsindy     = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 

# integrate !! 
x0 = data_test.x_true[1,:] 
x_sindy_stls_test  = integrate_euler( dx_fn_sindy_stls, x0, data_test.t, data_test.u ) 
x_sindy_lasso_test = integrate_euler( dx_fn_sindy_lasso, x0, data_test.t, data_test.u ) 
x_nn_test          = integrate_euler( dx_fn_nn, x0, data_test.t, data_test.u ) 
x_gpsindy_test     = integrate_euler( dx_fn_gpsindy, x0, data_test.t, data_test.u ) 

# ----------------------- #
# plot smoothed data and validation test data 

plot_validation_test( t_test, data_test.x_true, x_sindy_stls_test, x_sindy_lasso_test, x_sindy_lasso_test, x_gpsindy_test ) 









# ## ============================================ ##
    
# function err_Ξ_x( ) 

#     x_sindy_stls_err = [] 
#     for i = 1 : x_vars 
#         push!( x_sindy_stls_err, norm( data_test.x_true[:,i] - x_sindy_stls_test[:,i] )  ) 
#     end 
#     x_sindy_lasso_err = [] 
#     for i = 1 : x_vars 
#         push!( x_sindy_lasso_err, norm( data_test.x_true[:,i] - x_sindy_lasso_test[:,i] )  ) 
#     end
#     x_nn_err = [] 
#     for i = 1 : x_vars 
#         push!( x_nn_err, norm( data_test.x_true[:,i] - x_nn_test[:,i] )  ) 
#     end 
#     x_gpsindy_err = [] 
#     for i = 1 : x_vars 
#         push!( x_gpsindy_err, norm( data_test.x_true[:,i] - x_gpsindy_test[:,i] )  ) 
#     end 
#     x_err = x_err_struct( x_sindy_stls_err, x_sindy_lasso_err, x_nn_err, x_gpsindy_err ) 

#     Ξ_sindy_stls_err = [] 
#     for i = 1 : n_vars 
#         push!( Ξ_sindy_stls_err, norm( Ξ_true_stls[:,i] - Ξ_sindy_stls[:,i] )  ) 
#     end 
#     Ξ_sindy_lasso_err = [] 
#     for i = 1 : n_vars 
#         push!( Ξ_sindy_lasso_err, norm( Ξ_true_stls[:,i] - Ξ_sindy_lasso[:,i] )  ) 
#     end 
#     Ξ_nn_err = [] 
#     for i = 1 : n_vars 
#         push!( Ξ_nn_err, norm( Ξ_true_stls[:,i] - Ξ_nn_lasso[:,i] )  ) 
#     end 
#     Ξ_gpsindy_err = [] 
#     for i = 1 : n_vars 
#         push!( Ξ_gpsindy_err, norm( Ξ_true_stls[:,i] - Ξ_gpsindy[:,i] )  ) 
#     end 
#     Ξ_err = Ξ_err_struct( Ξ_sindy_stls_err, Ξ_sindy_lasso_err, Ξ_nn_err, Ξ_gpsindy_err ) 

# end 
