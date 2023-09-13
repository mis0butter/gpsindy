using GaussianSINDy
using CSV, DataFrames 

# choose ODE, plot states --> measurements 
fn = unicycle 

# set up noise vec 
noise_vec      = []
noise_vec_iter = 0.01 : 0.01 : 0.05  
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
x_err_hist  = x_err_struct([], [], [], [])
Ξ_hist      = Ξ_struct([], [], [], [], []) 
Ξ_err_hist  = Ξ_err_struct([], [], [], [])
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

# plot_med_quarts_sindy_nn_gpsindy(x_sindy_lasso_err, x_gpsindy_err, x_nn_err, noise_vec)

# ----------------------- #
# save outputs as csv 
data   = [ noise_vec Ξ_sindy_lasso_err Ξ_gpsindy_err Ξ_nn_err x_sindy_lasso_err x_gpsindy_err x_nn_err ]
header = [ "noise_vec", "Ξ_sindy_err", "Ξ_gpsindy_err", "Ξ_nn_err", "x_sindy_err", "x_gpsindy_err", "x_nn_err" ] 
df     = DataFrame( data,  :auto ) 
CSV.write(string(string(fn), "_batch.csv"), df, header=header)


## ============================================ ##
# single run 

fn = unicycle 

noise = 0.0 
λ = 0.1 
x_test_hist = x_struct( [], [], [], [], [], [] ) 
x_err_hist  = x_err_struct([], [], [], [])
Ξ_hist      = Ξ_struct([], [], [], [], []) 
Ξ_err_hist  = Ξ_err_struct([], [], [], [])
Ξ_hist, Ξ_err_hist, x_hist, x_err_hist = sindy_nn_gpsindy( fn, noise, λ, Ξ_hist, Ξ_err_hist, x_test_hist, x_err_hist ) 

x_true = x_hist.truth[1] 
x_sindy = x_hist.sindy_lasso[1] 
x_nn = x_hist.nn[1] 
x_gpsindy = x_hist.gpsindy[1] 

t_test = x_hist.t[1] 
x_test = x_hist.truth[1] 
plot_x_sindy_nn_gpsindy( t_test, x_test, x_sindy, x_nn, x_gpsindy)  

if fn == unicycle 
    header = [ "t", "x1_sindy", "x2_sindy", "x3_sindy", "x4_sindy", "x1_gpsindy", "x2_gpsindy", "x3_gpsindy", "x4_gpsindy", "x1_nn", "x2_nn", "x3_nn", "x4_nn" ] 
elseif fn == predator_prey
    header = [ "t", "x1_sindy", "x2_sindy", "x1_gpsindy", "x2_gpsindy", "x1_nn", "x2_nn" ] 
end 
data   = [ t_test x_sindy x_gpsindy x_nn ]
df     = DataFrame( data,  :auto ) 
CSV.write(string(string(fn), "_single", ".csv"), df, header=header)


# ## ============================================ ##
# # save data 

# using JLD2

# timestamp = Dates.format(now(), "YYYYmmdd-HHMMSS")
# dir_name  = joinpath(@__DIR__, "outputs", "runs_$timestamp")
# @assert !ispath(dir_name) "Somebody else already created the directory"
# if !ispath(dir_name)
#     mkdir(dir_name)
# end 

# # save 
# jldsave(string( dir_name, "\\Ξ_err_hist.jld2" ); Ξ_err_hist)

