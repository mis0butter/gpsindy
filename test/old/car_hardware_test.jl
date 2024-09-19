using GaussianSINDy 


## ============================================ ##
# setup 

# load data 
t, x, u = extract_car_data() 
x_vars, u_vars, poly_order, n_vars = size_vars( x, u ) 
x, dx_fd = unroll( t, x ) 

# split into training and test data 
test_fraction = 0.2 
portion       = 5 
u_train,  u_test  = split_train_test( u, test_fraction, portion ) 
t_train,  t_test  = split_train_test( t, test_fraction, portion ) 
x_train,  x_test  = split_train_test( x, test_fraction, portion ) 
dx_train, dx_test = split_train_test( dx_fd, test_fraction, portion ) 


## ============================================ ##
# SINDy vs. GPSINDy 

λ = 100.0 
Ξ_sindy, Ξ_gpsindy, Ξ_gpsindy_x2, Ξ_sindy_terms, Ξ_gpsindy_terms, Ξ_gpsindy_x2_terms = gpsindy_Ξ_fn( t_train, x_train, dx_train, λ, u_train )  

# ----------------------- #
# validate 

x_vars = size(x_train, 2) 
u_vars = size(u_train, 2) 
poly_order = x_vars 

# build dx functions 
dx_fn_sindy      = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy ) 
dx_fn_gpsindy    = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
dx_fn_gpsindy_x2 = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy_x2 ) 

# integrate 
x0 = x_test[1,:] 
x_unicycle_test   = integrate_euler_unicycle(unicycle_realistic, x0, t_test, u_test) 
x_sindy_test      = integrate_euler( dx_fn_sindy, x0, t_test, u_test ) 
x_gpsindy_test    = integrate_euler( dx_fn_gpsindy, x0, t_test, u_test ) 
x_gpsindy_x2_test = integrate_euler( dx_fn_gpsindy_x2, x0, t_test, u_test ) 

# plot smoothed data and validation test data 
plot_validation_test( t_test, x_test, x_unicycle_test, x_sindy_test, x_gpsindy_x2_test) 


