## ============================================ ##

# export build_dx_fn 
# function build_dx_fn(poly_order, x_vars, u_vars, z_fd) 

#     n_vars = x_vars + u_vars 

#     # define pool_data functions 
#     fn_vector = pool_data_vecfn_test(n_vars, poly_order) 

#     # numerically evaluate each function at x and return a vector of numbers
#     𝚽( xu, fn_vector ) = [ f(xu) for f in fn_vector ]

#     # create vector of functions, each element --> each state 
#     dx_fn_vec = Vector{Function}(undef,0) 
#     for i = 1 : x_vars 
#         # define the differential equation 
#         push!( dx_fn_vec, (xu,p,t) -> dot( 𝚽( xu, fn_vector ), z_fd[:,i] ) ) 
#     end 

#     dx_fn(xu,p,t) = [ f(xu,p,t) for f in dx_fn_vec ] 

#     return dx_fn 

# end 


## ============================================ ##
# test without control 

using GaussianSINDy 

fn = predator_prey 

x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2) 

λ = 0.1 
Ξ_true = SINDy_test( x_true, dx_true, λ ) 

z_fd       = Ξ_true 
x_vars     = size(x0, 1) 
poly_order = x_vars 
u_vars     = 0 

# ----------------------- #
dx_fn_true = build_dx_fn(poly_order, x_vars, u_vars, z_fd) 

# n_vars = x_vars + u_vars 

# # define pool_data functions 
# fn_vector = pool_data_vecfn_test(n_vars, poly_order) 

# # numerically evaluate each function at x, u and return a vector of numbers
# 𝚽( xu, fn_vector ) = [ f(xu) for f in fn_vector ]

# # create vector of functions, each element --> each state 
# dx_fn_vec = Vector{Function}(undef,0) 
# for i = 1 : x_vars 
#     # define the differential equation 
#     push!( dx_fn_vec, (xu,p,t) -> dot( 𝚽( xu, fn_vector ), z_fd[:,i] ) ) 
# end 

# dx_fn(xu,p,t) = [ f(xu,p,t) for f in dx_fn_vec ] 



## ============================================ ##
# test with control 


