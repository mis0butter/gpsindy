using GaussianSINDy 
using Plots 
using LinearAlgebra


## ============================================ ##

fn = unicycle 

# generate true states 
x0, dt, t, x_true, dx_true, dx_fd, p, u = ode_states(fn, 0, 2)
if u == false 
    u_train = false ; u_test  = false 
else 
    u_train, u_test = split_train_test(u, 0.2, 5) 
end 

# truth coeffs 
λ = 0.1 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x_true, u ) 
Ξ_true = sindy_stls(x_true, dx_true, λ, u)

# ----------------------- #

x_vars, u_vars, poly_order, n_vars = size_x_n_vars( dx_true, u ) 
    
if isequal(u, false)      # if u_data = false 
    data   = x_true 
else            # there are u_data inputs 
    data   = [ x_true u ]
end 

# construct data library 
poly_order = 1 
Θx = pool_data_test(data, n_vars, poly_order) 

# SINDy 
Ξ_true = sparsify_dynamics_stls( Θx, dx_true, λ, x_vars ) 


## ============================================ ##

plot( x_true[:,1], x_true[:,2] ) 

Ξ_true_terms = pretty_coeffs( Ξ_true, x_true, u ) 

## ============================================ ##


# add noise 
noise = 0.01 
println("noise = ", noise)
x_noise  = x_true + noise * randn(size(x_true, 1), size(x_true, 2))
dx_noise = dx_true + noise * randn(size(dx_true, 1), size(dx_true, 2))
# dx_noise = dx_fd 

# split into training and test data 
test_fraction = 0.2
portion = 5
t_train, t_test               = split_train_test(t, test_fraction, portion)
x_train_true, x_test_true     = split_train_test(x_true, test_fraction, portion)
dx_train_true, dx_test_true   = split_train_test(dx_true, test_fraction, portion)
x_train_noise, x_test_noise   = split_train_test(x_noise, test_fraction, portion)
dx_train_noise, dx_test_noise = split_train_test(dx_noise, test_fraction, portion)

# ----------------------- # 
# standardize  
x_stand_noise  = stand_data(t_train, x_train_noise)
x_stand_true   = stand_data(t_train, x_train_true)
dx_stand_true  = dx_true_fn(t_train, x_stand_true, p, fn)
dx_stand_noise = dx_stand_true + noise * randn(size(dx_stand_true, 1), size(dx_stand_true, 2))

# set training data for GPSINDy 
x_train  = x_stand_noise 
dx_train = dx_stand_noise

## ============================================ ##
# SINDy vs. NN vs. GPSINDy 

# SINDy by itself 
Ξ_sindy_stls  = sindy_stls(x_train, dx_train, λ, u_train)
Ξ_sindy_lasso = sindy_lasso(x_train, dx_train, λ, u_train)

# GPSINDy (first) 
x_train_GP  = gp_post(t_train, 0 * x_train, t_train, 0 * x_train, x_train)
dx_train_GP = gp_post(x_train_GP, 0 * dx_train, x_train_GP, 0 * dx_train, dx_train)
Ξ_gpsindy   = sindy_lasso(x_train_GP, dx_train_GP, λ, u_train)


# ----------------------- # 
# validate data 

x_vars = size(x_true, 2)
# dx_fn_true        = build_dx_fn( poly_order, x_vars, u_vars, Ξ_true ) 
dx_fn_sindy_stls  = build_dx_fn(poly_order, x_vars, u_vars, Ξ_sindy_stls)
dx_fn_sindy_lasso = build_dx_fn(poly_order, x_vars, u_vars, Ξ_sindy_lasso)
dx_fn_gpsindy     = build_dx_fn(poly_order, x_vars, u_vars, Ξ_gpsindy)

# # integrate !! 
x0 = x_test_true[1,:] 
# x_test_true        = integrate_euler( dx_fn_true, x0, t_test, u_test )
x_sindy_stls_test  = integrate_euler( dx_fn_sindy_stls, x0, t_test, u_test ) 
x_sindy_lasso_test = integrate_euler( dx_fn_sindy_lasso, x0, t_test, u_test ) 
x_gpsindy_test     = integrate_euler( dx_fn_gpsindy, x0, t_test, u_test ) 