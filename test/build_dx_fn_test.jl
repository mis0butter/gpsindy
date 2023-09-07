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
#         push!( dx_fn_vec, (xu,t) -> dot( 𝚽( xu, fn_vector ), z_fd[:,i] ) ) 
#     end 

#     dx_fn(xu,t) = [ f(xu,t) for f in dx_fn_vec ] 

#     return dx_fn 

# end 

## ============================================ ##
# test without control 

using GaussianSINDy 

fn = predator_prey 

x0, dt, t, x_true, dx_true, dx_fd, p, u = ode_states(fn, 0, 2) 

λ = 0.1 
Ξ_true = SINDy_test( x_true, dx_true, λ ) 

z_fd       = Ξ_true 
x_vars     = size(x_true, 2) 
poly_order = x_vars 

if u == false 
    u_vars     = 0 
else 
    u_vars = size(u, 2) 
end 

# build function 
dx_fn_true = build_dx_fn(poly_order, x_vars, u_vars, z_fd) 

# test that it works 
dx0_test = dx_fn_true( x0, 0 ) 
println( "dx0_true = ", dx_true[1,:] )
println( "dx0_test = ", dx0_test )  
dxf_test = dx_fn_true( x_true[end,:], 0 ) 
println( "dxf_true = ", dx_true[end,:] )
println( "dxf_test = ", dxf_test )  


## ============================================ ##
# test with control 

fn = predator_prey_forcing 

x0, dt, t, x_true, dx_true, dx_fd, p, u = ode_states(fn, 0, 2) 

λ = 0.1 
# function SINDy_test( x, dx, λ, u = false )
Ξ_true = SINDy_test( x_true, dx_true, λ, u ) 

z_fd       = Ξ_true 
x_vars     = size(x_true, 2) 
poly_order = x_vars 

if u == false 
    u_vars     = 0 
else 
    u_vars = size(u, 2) 
end 

# build function 
dx_fn_true_ctrl = build_dx_fn(poly_order, x_vars, u_vars, z_fd) 

xu0 = copy( x0 ) 
push!( xu0, u[1] ) 
# test that it works 
dx0_test = dx_fn_true_ctrl( xu0, 0 ) 
println( "dx0_true = ", dx_true[1,:] )
println( "dx0_test = ", dx0_test )  

xuf = x_true[end,:] 
push!( xuf, u[end] ) 
dxf_test = dx_fn_true_ctrl( xuf, 0 ) 
println( "dxf_true = ", dx_true[end,:] )
println( "dxf_test = ", dxf_test )  

