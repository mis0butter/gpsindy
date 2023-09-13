using GaussianSINDy
using CSV, DataFrames 

# single run 
fn = unicycle 
# fn = predator_prey 

noise = 0.01 
λ = 0.1 
x_test_hist = x_struct( [], [], [], [], [], [] ) 
x_err_hist  = x_err_struct( [], [], [], [] )
Ξ_hist      = Ξ_struct( [], [], [], [], [] ) 
Ξ_err_hist  = Ξ_err_struct( [], [], [], [] )
Ξ_hist, Ξ_err_hist, x_hist, x_err_hist = sindy_nn_gpsindy( fn, noise, λ, Ξ_hist, Ξ_err_hist, x_test_hist, x_err_hist ) 

x_true      = x_hist.truth[1] 
x_sindy     = x_hist.sindy_lasso[1] 
x_nn        = x_hist.nn[1] 
x_gpsindy   = x_hist.gpsindy[1] 
t_test      = x_hist.t[1] 
x_test      = x_hist.truth[1] 
plot_x_sindy_nn_gpsindy( t_test, x_test, x_sindy, x_nn, x_gpsindy)  

# check coeffs 
Ξ_true    = Ξ_hist.truth[1] 
Ξ_sindy   = Ξ_hist.sindy_lasso[1] 
Ξ_gpsindy = Ξ_hist.gpsindy[1] 
if fn == unicycle 
    u_temp = zeros( length(t_test), 2 )
elseif fn == predator_prey 
    u_temp = false 
end 
Ξ_true_terms    = pretty_coeffs( Ξ_true, x_true, u_temp )
Ξ_sindy_terms   = pretty_coeffs( Ξ_sindy, x_true, u_temp )
Ξ_gpsindy_terms = pretty_coeffs( Ξ_gpsindy, x_true, u_temp )


# ## ============================================ ##
# # save data 

if fn == unicycle 
    header = [ "t", "x1_sindy", "x2_sindy", "x3_sindy", "x4_sindy", "x1_gpsindy", "x2_gpsindy", "x3_gpsindy", "x4_gpsindy", "x1_nn", "x2_nn", "x3_nn", "x4_nn" ] 
elseif fn == predator_prey
    header = [ "t", "x1_sindy", "x2_sindy", "x1_gpsindy", "x2_gpsindy", "x1_nn", "x2_nn" ] 
end 
data   = [ t_test x_sindy x_gpsindy x_nn ]
df     = DataFrame( data,  :auto ) 
CSV.write(string(string(fn), "_single", ".csv"), df, header=header)

