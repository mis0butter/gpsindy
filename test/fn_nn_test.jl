using GaussianSINDy 
using LinearAlgebra

## ============================================ ##

# generate data 
fn    = predator_prey  
# fn    = unicycle 


# set up noise vec 
noise_vec = [] 
noise_vec_iter = 0.05 : 0.05 : 0.25  
for i in noise_vec_iter 
    for j = 1:5 
        push!(noise_vec, i)
    end 
end 


## ============================================ ## 
# the big one 

Ξ_err = Ξ_err_struct( [], [], [], [] ) 
x_err = x_err_struct( [], [], [], [] ) 
# for i = eachindex(noise_vec) 
x_hist = [] 
noise = 0.25 
for i = [10, 20] 
    noise = noise_vec[i] 
    data_train, data_test, data_train_stand = ode_train_test( fn, noise ) 
    x, Ξ = x_Ξ_fn( data_train, data_test, data_train_stand ) 
    push!( x_hist, x ) 
    Ξ_err, x_err = err_Ξ_x( Ξ, x, Ξ_err, x_err )     
end 


