# this is a script for tuning lambda for each dx based on the error from the TRAINING DATA 

using GaussianSINDy 
using LinearAlgebra 

## ============================================ ##

fn = unicycle 

λ = 0.1 
data_train, data_test = ode_train_test( fn ) 
Ξ_true       = sindy_stls( data_train.x_true, data_train.dx_true, λ, data_train.u ) 
Ξ_true_terms = pretty_coeffs( Ξ_true, data_train.x_true, data_train.u ) 

x_train  = data_train.x_noise 
dx_train = data_train.dx_noise 
u_train  = data_train.u 
t_train  = data_train.t 


## ============================================ ##

# get sizes 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x_train, u_train ) 

# first - smooth measurements with Gaussian processes 
x_GP_train  = gp_post( t_train, 0*x_train, t_train, 0*x_train, x_train ) 
dx_GP_train = gp_post( x_GP_train, 0*dx_train, x_GP_train, 0*dx_train, dx_train ) 

# get x0 from smoothed data 
x0 = x_GP_train[1,:] 

# SINDy (STLS) 
λ = 0.1 
Ξ_sindy_stls  = sindy_stls( x_train, dx_train, λ, u_train ) 
dx_fn_sindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_stls ) 
x_sindy_train = integrate_euler( dx_fn_sindy, x0, t_train, u_train ) 

# integrate unicycle 
# x_unicycle_train = integrate_euler_unicycle( fn, x0, t_train, u_train ) 


## ============================================ ##

λ_vec = [ 1e-4 ] 
while λ_vec[end] < 10 
    push!( λ_vec, 10 * λ_vec[end] ) 
end 
while λ_vec[end] < 500 
    push!( λ_vec, 10 + λ_vec[end] ) 
end 


err_Ξ_vec = [] 
err_x_vec = [] 
for j = 1 : x_vars 
    
    err_ξ_vec  = [] 
    err_xj_vec = [] 
    for i = 1 : length(λ_vec) 
        # j = 1 
    
        λ = λ_vec[i] 
        println( "i = ", i, ". λ = ", λ ) 

        Ξ_gpsindy = sindy_lasso( x_GP_train, dx_GP_train, λ, u_train ) 

        if sum(Ξ_gpsindy[:,j]) == 0 
            break 
        end 

        Ξ_gpsindy_terms = pretty_coeffs( Ξ_gpsindy, x_GP_train, u_train ) 
        display(Ξ_gpsindy_terms) 

        dx_fn_gpsindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
        x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0, t_train, u_train ) 
        err_ξ_norm      = norm( Ξ_true[:,j] - Ξ_gpsindy[:,j] ) 
        err_x_norm      = norm( x_gpsindy_train[:,j] - x_train[:,j] ) 

        # plot_validation_test( t_train, x_train, x_unicycle_train, x_sindy_train, x_gpsindy_train ) 

        push!( err_ξ_vec, err_ξ_norm ) 
        push!( err_xj_vec, err_x_norm ) 

    end 

    push!( err_Ξ_vec, err_ξ_vec ) 
    push!( err_x_vec, err_xj_vec ) 

end 

