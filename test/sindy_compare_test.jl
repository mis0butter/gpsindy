using GaussianSINDy 


## ============================================ ##

fn = unicycle 
data_train, data_test = ode_train_test( fn ) 

x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_true, data_train.u ) 

if isequal(data_train.u, false)     # if u_data = false 
    data   = data_train.x_true 
else                                # there are u_data inputs 
    data   = [ data_train.x_true data_train.u ]
end 

# construct data library 
Θx = pool_data_test(data, n_vars, poly_order) 

# SINDy 
Ξ_test  = sparsify_dynamics_stls( Θx, data_train.dx_true, λ, x_vars ) 
# Ξ = sparsify_dynamics_cstrnd( Θx, dx, λ, x_vars ) 
Ξ_lasso = sparsify_dynamics_lasso( Θx, data_train.dx_true, λ, x_vars ) 


