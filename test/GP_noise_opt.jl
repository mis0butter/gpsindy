using GaussianSINDy 
using LinearAlgebra 
using CSV, DataFrames 
using CairoMakie 


## ============================================ ##
# extract data 

path          = "test/data/jake_car_csvs_ctrlshift/10hz/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end 

# for i = eachindex(csv_files_vec) 
# for i = [ 4 ]
    i = 1 
    csv_file = csv_files_vec[i] 
    df       = CSV.read(csv_file, DataFrame) 
    data     = Matrix(df) 
    header   = names(df) 

    # get data 
    t = data[:,1] ; x = data[:,2:5] ; u = data[:,6:7] 


## ============================================ ##
# gp_post function: 
# gp_post( x_prior, μ_prior, x_train, μ_train, y_train ) 

using GaussianProcesses 
using LineSearches 
using Optim 

# okay, this part is so confusing .. but basically: 
# inputs:   t (time vector)  --> x (GP)  
# outputs:  x (state vector) --> y (GP) 
n_vars    = size(x, 2) 
prior_row = size(t)[1] 
train_row = size(x)[1]   

# massage inputs 
x_prior = t 
μ_prior = zeros( prior_row, n_vars ) 

x_train = t 
μ_train = zeros( train_row, n_vars ) 
y_train = x 

# set up posterior (will have same dimension as the PRIOR / test output points) 
y_post = zeros( prior_row, n_vars ) 


## ============================================ ##
# manually tuning the hyperparameters 

σ_f = 1.0   # signal variance 
l   = 1.0   # length scale 
σ_n = 0.0   # signal noise 

hp  = [ σ_f, l ] 

hp_opt( ( σ_f, l ) ) = log_p( σ_f, l, σ_n, x_train, y_train[:,i], μ_train[:,i] )
od       = OnceDifferentiable( hp_opt, hp ; autodiff = :forward ) 
result   = optimize( od, hp, LBFGS() ) 
hp       = result.minimizer 

μ_post, Σ_post = post_dist( x_train, y_train[:,i], x_prior, hp[1], hp[2], σ_n ) 



## ============================================ ##
# using GP toolbox 

using Plots 


# choose a state 
i = 3 

σ_f = 1.0  
l   = 1.0 
σ_n = 0.1 

# kernel  
mZero     = MeanZero()                  # zero mean function 
kern      = SE( log( σ_f ), log( l ) )  # squared eponential kernel (hyperparams on LOG scale)  
log_noise = log( σ_n )                  # (optional) log std dev of obs 

gp      = GP( x_train', y_train[:,i] - μ_train[:,i], mZero, kern, log_noise ) 
# gp      = GP( x_train', y_train[:,i] - μ_train[:,i], mZero, kern ) 

optimize!( gp; method = LBFGS( linesearch = LineSearches.BackTracking() ), noise = false ) 

μ_post  = predict_y( gp, x_prior' )[1] 

# return HPs 
σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt.( gp.kernel.ℓ2 ) ; σ_n = exp( gp.logNoise.value ) 
hp  = [ σ_f, l, σ_n ] 
println( "hp = ", hp ) 

p = Plots.plot(gp) 

# use cairomakie to plot scatter 
f = Figure() 
    ax = Axis( f[1,1] ) 
    CairoMakie.scatter!( ax, x_train, y_train[:,i] ) 
    lines!( ax, x_prior, μ_post ) 
f 







