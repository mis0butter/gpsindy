using GaussianSINDy 
using LinearAlgebra 
using GLMakie 

## ============================================ ##
# single run (good) 

path          = "test/data/jake_car_csvs/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end

x_err_hist  = x_err_struct([], [], [], []) 

norm_err_vec = [  ] 

for i = eachindex(csv_files_vec) 
# i = 1 

    csv_file = csv_files_vec[i] 

    # get x and dx training data 
    data_train, data_test = car_data_struct( csv_file ) 
    t_train  = data_train.t[:,1]
    x_train  = data_train.x_noise 
    dx_train = data_train.dx_noise 

    # smooth with GPs 
    x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 

    x_err = norm( x_train_GP - x_train ) 
    push!( norm_err_vec, x_err ) 

end 


## ============================================ ##
# i = 17 error 

i = 17 
csv_file = csv_files_vec[i] 

# get x and dx training data 
data_train, data_test = car_data_struct( csv_file ) 
t_train  = data_train.t[:,1] 
x_train  = data_train.x_noise 
dx_train = data_train.dx_noise 

t, x, u = extract_car_data( csv_file ) 

## ============================================ ##
# debug unroll 

    # use forward finite differencing 
    dx_fd = fdiff(t, x, 1) 

    # massage data, generate rollovers  
    rollover_up_ind = findall( x -> x > 100, dx_fd[:,4] ) 
    rollover_dn_ind = findall( x -> x < -100, dx_fd[:,4] ) 

    # @infiltrate 

    for i in eachindex(rollover_up_ind) 

        i0   = rollover_up_ind[i] + 1 
        ifin = rollover_dn_ind[i] 
        rollover_rng = x[ i0 : ifin , 4 ]
        dθ = π .- rollover_rng 
        θ  = -π .- dθ 
        x[ i0 : ifin , 4 ] = θ

    end 

## ============================================ ##

t, x, u = extract_car_data( csv_file ) 

# debug car_data_struct 

    # use forward finite differencing 
    dx_fd = fdiff(t, x, 1) 

    # massage data, generate rollovers  
    rollover_up_ind = findall( x -> x > 100, dx_fd[:,4] ) 
    rollover_dn_ind = findall( x -> x < -100, dx_fd[:,4] ) 

    # @infiltrate 

    # for i in eachindex(rollover_up_ind) 
    i = 1 

        if rollover_up_ind[i] < rollover_dn_ind[i] 
            i0   = rollover_up_ind[i] + 1 
            ifin = rollover_dn_ind[i]     
            rollover_rng = x[ i0 : ifin , 4 ]
            dθ = π .- rollover_rng 
            θ  = -π .- dθ 
        else 
            i0   = rollover_dn_ind[i] + 1 
            ifin = rollover_up_ind[i]     
            rollover_rng = x[ i0 : ifin , 4 ]
            dθ = π .+ rollover_rng 
            θ  = π .+ dθ     
        end 
        x[ i0 : ifin , 4 ] = θ

    # end 

plot( t, x[:,4] )


## ============================================ ## 

# smooth with GPs 
x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 


## ============================================ ## 
# plot train and train_GP data 

fig = Figure() 
for i = 1:4 
    Axis( fig[i, 1], xlabel = "t", ylabel = string( "x",i ) ) 
        lines!( fig[i,1] , t_train, x_train[:,i], label = "raw" ) 
        # lines!( fig[i,1] , t_train, x_train_GP[:,i], linestyle = :dash, label = "GP" ) 
    Axis( fig[i, 2], xlabel = "t", ylabel = string( "dx",i ) ) 
        lines!( fig[i,2] , t_train, dx_train[:,i], label = "raw" ) 
        # lines!( fig[i,2] , t_train, dx_train_GP[:,i], linestyle = :dash, label = "GP" ) 
end       
fig  


## ============================================ ##

fig = Figure() 

    Axis( fig[1:2, 1], xlabel = "x", ylabel = "y" ) 
        lines!( fig[1:2,1] , x_train[:,1], x_train[:,2], label = "raw" ) 
    Axis( fig[3, 1], xlabel = "t", ylabel = "v" ) 
        lines!( fig[3,1] , t_train, x_train[:,3], label = "v" ) 
    Axis( fig[4, 1], xlabel = "t", ylabel = "θ" ) 
        lines!( fig[4,1] , t_train, x_train[:,4], label = "θ" ) 

    Axis( fig[1, 2], ylabel = "xdot" ) 
        lines!( fig[1,2] , t_train, dx_train[:,1], label = "raw" ) 
    Axis( fig[2, 2], ylabel = "ydot" ) 
        lines!( fig[2,2] , t_train, dx_train[:,2], label = "raw" ) 
    Axis( fig[3, 2], ylabel = "vdot" ) 
        lines!( fig[3,2] , t_train, dx_train[:,3], label = "v" ) 
    Axis( fig[4, 2], xlabel = "t", ylabel = "θdot" ) 
        lines!( fig[4,2] , t_train, dx_train[:,4], label = "θ" ) 

fig  

## ============================================ ##









