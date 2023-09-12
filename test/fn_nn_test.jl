using GaussianSINDy 
using LinearAlgebra

## ============================================ ##

# generate data 
# fn    = predator_prey  
fn    = unicycle 
noise = 0.01 

## ============================================ ## 
# the big one 

data_train, data_test, data_train_stand = ode_train_test( fn, noise ) 
x, Ξ = x_Ξ_fn( data_train, data_test, data_train_stand ) 
Ξ_err, x_err = err_Ξ_x( x, Ξ ) 


