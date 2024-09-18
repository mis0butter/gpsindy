using GaussianSINDy 
using LinearAlgebra 
using CairoMakie 

## ============================================ ##
# single run (good) 

path          = "test/data/jake_car_csvs_control_adjust_10hz/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end
N = length(csv_files_vec) 

norm_err_vec = [  ] 

# for i = 1 : N 
# i = 1 

    csv_file = csv_files_vec[i] 
    println( "i_csv = ", i, ". csv_file = ", csv_file ) 

    # get x and dx training data 
    data_train, data_test = make_data_structs( csv_file ) 
    t_train  = data_train.t[:,1]
    x_train  = data_train.x_noise 
    dx_train = data_train.dx_noise 

    # smooth with GPs 
    x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = smooth_data_gp( data_train, data_test ) 

    # make fig 
    fig = plot_car_raw_GP( t_train, x_train, dx_train, x_train_GP, dx_train_GP) 

    # save fig to folder 
    figstring = replace( csv_file, "data" => "images" ) 
    figstring = replace( figstring, ".csv" => ".png" ) 
    save( figstring, fig ) 

    x_err = norm( x_train_GP - x_train ) 
    push!( norm_err_vec, x_err ) 

# end 


## ============================================ ##
# i = 40 error 

i = 40 
csv_file = csv_files_vec[i] 

# get x and dx training data 
data_train, data_test = make_data_structs( csv_file ) 
t_train  = data_train.t[:,1] 
x_train  = data_train.x_noise 
dx_train = data_train.dx_noise 


# ----------------------- #
# smooth with GPs 

x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = smooth_data_gp( data_train, data_test ) 


## ============================================ ## 
# plot x_train and dx_train data  

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
# unroll errors? 

t, x, u = extract_car_data( csv_file ) 

# x, dx = unroll( t, x ) 

# use forward finite differencing 
dx = fdiff(t, x, 1) 

# massage data, generate rollovers  
rollover_up_ind = findall( x -> x > 100, dx[:,4] ) 
rollover_dn_ind = findall( x -> x < -100, dx[:,4] ) 

for i in eachindex(rollover_up_ind) 
    # i = 1 

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

    end 

# @infiltrate 

# use central finite differencing now  
dx = fdiff(t, x, 2) 


## ============================================ ##


f = Figure() 

    Axis( f[1:2, 1], xlabel = "x", ylabel = "y" ) 
        lines!( f[1:2,1] , x[:,1], x[:,2], label = "raw" ) 
    Axis( f[3, 1], xlabel = "x", ylabel = "y" ) 
        lines!( f[3,1] , t, x[:,3], label = "raw" ) 
    Axis( f[4, 1], xlabel = "x", ylabel = "y" ) 
        lines!( f[4,1] , t, x[:,4], label = "raw" ) 

    Axis( f[1, 2], ylabel = "xdot" ) 
        lines!( f[1,2] , t, dx[:,1], label = "raw" ) 
    Axis( f[2, 2], ylabel = "ydot" ) 
        lines!( f[2,2] , t, dx[:,2], label = "raw" ) 
    Axis( f[3, 2], ylabel = "vdot" ) 
        lines!( f[3,2] , t, dx[:,3], label = "v" ) 
    Axis( f[4, 2], xlabel = "t", ylabel = "θdot" ) 
        lines!( f[4,2] , t, dx[:,4], label = "θ" ) 

f 





