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
N = length(csv_files_vec) 

norm_err_vec = [  ] 

for i = 1 : N 
# i = 1 

    csv_file = csv_files_vec[i] 
    println( "i_csv = ", i, ". csv_file = ", csv_file ) 

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
# i = 40 error 

i = 40 
csv_file = csv_files_vec[i] 

# get x and dx training data 
data_train, data_test = car_data_struct( csv_file ) 
t_train  = data_train.t[:,1] 
x_train  = data_train.x_noise 
dx_train = data_train.dx_noise 


# ----------------------- #
# smooth with GPs 

x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 


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
# plot x_train, dx_train, GP and raw data  

fig = Figure() 

Axis( fig[1:2, 1], xlabel = "x", ylabel = "y" ) 
    lines!( fig[1:2,1] , x_train[:,1], x_train[:,2], label = "raw" ) 
    lines!( fig[1:2,1] , x_train_GP[:,1], x_train_GP[:,2], linestyle = :dash, label = "GP" ) 
    axislegend() 

Axis( fig[3, 1], xlabel = "t", ylabel = "v" ) 
    lines!( fig[3,1] , t_train, x_train[:,3], label = "v" ) 
    lines!( fig[3,1] , t_train, x_train_GP[:,3], linestyle = :dash, label = "v" ) 
Axis( fig[4, 1], xlabel = "t", ylabel = "θ" ) 
    lines!( fig[4,1] , t_train, x_train[:,4], label = "θ" ) 
    lines!( fig[4,1] , t_train, x_train_GP[:,4], linestyle = :dash, label = "θ" ) 

Axis( fig[1, 2], ylabel = "xdot" ) 
    lines!( fig[1,2] , t_train, dx_train[:,1], label = "raw" ) 
    lines!( fig[1,2] , t_train, dx_train_GP[:,1], label = "raw" ) 
Axis( fig[2, 2], ylabel = "ydot" ) 
    lines!( fig[2,2] , t_train, dx_train[:,2], label = "raw" ) 
    lines!( fig[2,2] , t_train, dx_train_GP[:,2], label = "raw" ) 
Axis( fig[3, 2], ylabel = "vdot" ) 
    lines!( fig[3,2] , t_train, dx_train[:,3], label = "v" ) 
    lines!( fig[3,2] , t_train, dx_train_GP[:,3], label = "v" ) 
Axis( fig[4, 2], xlabel = "t", ylabel = "θdot" ) 
    lines!( fig[4,2] , t_train, dx_train[:,4], label = "θ" ) 
    lines!( fig[4,2] , t_train, dx_train_GP[:,4], label = "θ" ) 

fig  





