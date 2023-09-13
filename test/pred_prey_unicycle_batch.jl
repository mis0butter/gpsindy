using GaussianSINDy
using CSV, DataFrames 

# choose ODE, plot states --> measurements 
# fn = unicycle 
fn = predator_prey 

# set up noise vec 
noise_vec      = []
if fn == unicycle 
    noise_vec_iter = 0.01 : 0.01 : 0.05  
elseif fn == predator_prey  
    noise_vec_iter = 0.05 : 0.05 : 0.25  
end 
for i in noise_vec_iter
    for j = 1:5 
        push!(noise_vec, i)
    end
end 
# noise_vec = [ 0.01 ] 

## ============================================ ##
# start MC loop 

λ = 0.1 
x_test_hist = x_struct( [], [], [], [], [], [] ) 
x_err_hist  = x_err_struct( [], [], [], [] )
Ξ_hist      = Ξ_struct( [], [], [], [], [] ) 
Ξ_err_hist  = Ξ_err_struct( [], [], [], [] )
for noise = noise_vec 
    Ξ_hist, Ξ_err_hist, x_hist, x_err_hist = sindy_nn_gpsindy( fn, noise, λ, Ξ_hist, Ξ_err_hist, x_test_hist, x_err_hist ) 
end 

# ----------------------- #
# plot quartiles 

Ξ_sindy_stls_err  = Ξ_err_hist.sindy_stls
Ξ_sindy_lasso_err = Ξ_err_hist.sindy_lasso 
Ξ_gpsindy_err     = Ξ_err_hist.gpsindy 
Ξ_nn_err          = Ξ_err_hist.nn 

plot_med_quarts_sindy_nn_gpsindy(Ξ_sindy_lasso_err, Ξ_nn_err, Ξ_gpsindy_err, noise_vec)

x_sindy_stls_err  = x_err_hist.sindy_stls
x_sindy_lasso_err = x_err_hist.sindy_lasso 
x_gpsindy_err     = x_err_hist.gpsindy 
x_nn_err          = x_err_hist.nn 


## ============================================ ##
# save outputs as csv 

data   = [ noise_vec Ξ_sindy_lasso_err Ξ_gpsindy_err Ξ_nn_err x_sindy_lasso_err x_gpsindy_err x_nn_err ]
header = [ "noise_vec", "Ξ_sindy_err", "Ξ_gpsindy_err", "Ξ_nn_err", "x_sindy_err", "x_gpsindy_err", "x_nn_err" ] 
df     = DataFrame( data,  :auto ) 
CSV.write(string(string(fn), "_batch.csv"), df, header=header)


