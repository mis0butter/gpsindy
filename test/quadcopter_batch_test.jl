using GaussianSINDy 
using LinearAlgebra 
using CSV, DataFrames 
using BenchmarkTools 
using GLMakie 

## ============================================ ##
# get states and inputs 

path = "test/data/cyrus_quadcopter_csvs/" 
csv_files_vec = readdir( path ) 

# for i_csv in eachindex(csv_files_vec) 
i_csv = 1 

    csv_file = string( path, csv_files_vec[i_csv] ) 
    df   = CSV.read(csv_file, DataFrame) 
    x    = Matrix(df) 

i_csv = 2 

    csv_file = string( path, csv_files_vec[i_csv] ) 
    println( csv_file ) 

    df   = CSV.read(csv_file, DataFrame) 
    u    = Matrix(df) 

# end 

N  = size(x, 1) 

# time vector ( probably dt = 0.01 s? )
dt = 0.01 
t  = collect( range(0, step = dt, length = N) ) 

# get derivatives 
x, dx_fd = unroll( t, x ) 


## ============================================ ##
# plot entire trajectory 

# state vars: px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz 

fig = plot_line3d( x[:,1], x[:,2], x[:,3] ) 


## ============================================ ##
# split into training and testing 

N_train = Int( round( N * 0.8 ) ) 
N_train = 200 

t_train  = t[ 1:N_train ] 
x_train  = x[ 1:N_train, : ] 
dx_train = dx_fd[ 1:N_train, : ] 
u_train  = @btime u[ 1:N_train, : ] 

t_test   = t[ N_train+1:end ] 
x_test   = x[ N_train+1:end, : ] 
dx_test  = dx_fd[ N_train+1:end, : ] 
u_test   = u[ N_train+1:end, : ] 


x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x, u )

## ============================================ ##
# try sindy stls and lasso 

# try sindy 
λ = 0.1 

# println( "Ξ_stls time " ) 
# @btime Ξ_stls  = sindy_stls( x_train, dx_train, λ, u_train ) 

println( "Ξ_lasso time " ) 

start   = time() 
Ξ_lasso = @btime sindy_lasso( x_train, dx_train, λ, u_train ) 
elapsed = time() - start 
println( "btime elapsed = ", elapsed ) 

start   = time() 
Ξ_lasso = @time sindy_lasso( x_train, dx_train, λ, u_train ) 
elapsed = time() - start 
println( "time elapsed = ", elapsed ) 


## ============================================ ##
# try gp stuff 

# first - smooth measurements with Gaussian processes 

println( "x_GP time" )
start        = time() 
x_GP         = gp_post( t_train, 0*x_train, t_train, 0*x_train, x_train ) 
x_GP_elapsed = time() - start 

println( "dx_GP time" ) 
start         = time() 
dx_GP         = gp_post( x_GP, 0*dx_train, x_GP, 0*dx_train, dx_train ) 
dx_GP_elapsed = time() - start 

Ξ_GP_lasso = sindy_lasso( x_GP, dx_GP, λ, u_train ) 


## ============================================ ##
# now test on training data 

x0_train_GP = x_GP[1,:] 

# build dx fn 
dx_fn_gpsindy = build_dx_fn( poly_order, x_vars, u_vars, Ξ_GP_lasso ) 
x_train_pred  = integrate_euler( dx_fn_gpsindy, x0_train_GP, t_train, u_train )  


## ============================================ ## 
# debug build_dx_fn 

z_fd = Ξ_GP_lasso 

n_vars = x_vars + u_vars 

# define pool_data functions 
poly_order = 3 
fn_vector  = pool_data_vecfn_test(n_vars, poly_order) 

# numerically evaluate each function at x and return a vector of numbers
𝚽( xu, fn_vector ) = [ f(xu) for f in fn_vector ]

# create vector of functions, each element --> each state 
dx_fn_vec = Vector{Function}(undef,0) 
for i = 1 : x_vars 
    # define the differential equation 
    push!( dx_fn_vec, (xu,p,t) -> dot( 𝚽( xu, fn_vector ), z_fd[:,i] ) ) 
end 

dx_fn(xu,p,t) = [ f(xu,p,t) for f in dx_fn_vec ] 



## ============================================ ##


    # initialize empty vector of functions 
    Θx = Vector{Function}(undef,0) 

    # fil out 1st column of Θ with ones (poly order = 0) 
    ind  = 1 
    push!(Θx, x -> 1) 

    # poly order 1 
    a = 0 
    for i = 1 : n_vars 
        ind  += 1 
        a += 1 
        push!( Θx, x -> x[i] ) 
    end 

    # ind += 1 
    # push!( Θ, x[1] .* x[2] )

    # poly order 2 
    b = 0 
    if poly_order >= 2 
        for i = 1 : n_vars 
            for j = i:n_vars 
                ind += 1 ; 
                b += 1 
                push!( Θx, x -> x[i] .* x[j] ) 
            end 
        end 
    end 

    # poly order 3 
    c = 0 
    if poly_order >= 3 
        for i = 1 : n_vars 
            for j = i : n_vars 
                for k = j : n_vars 
                    ind += 1 ;                     
                    c += 1 
                    push!( Θx, x -> x[i] .* x[j] .* x[k] )
                end 
            end 
        end 
    end 

    # sine functions 
    d = 0 
    for i = 1 : n_vars 
        ind  += 1
        d += 1 
        push!(Θx, x -> sin.( x[i] ) )
    end 

    # sine functions 
    e = 0 
    for i = 1 : n_vars 
        ind  += 1
        e += 1 
        push!(Θx, x -> cos.( x[i] ) ) 
    end 

    # nonlinear combinations with sine functions 
    
    # initialize empty vector of functions 
    f = 0 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind += 1 
            f += 1 
            push!( Θx, x -> x[i] .* sin.( x[j] ) ) 
        end 
    end 

    # nonlinear combinations with cosine functions 
    g = 0 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind += 1 
            g += 1 
            push!( Θx, x -> x[i] .* cos.( x[j] ) ) 
        end 
    end 

    # poly order 2 nonlinear combination with sine functions 
    h = 0 
    if poly_order >= 2 
        for i = 1 : n_vars 
            for j = i : n_vars 
                for k = 1 : n_vars 
                    ind += 1 
                    h += 1 
                    push!( Θx, x -> x[i] .* x[j] .* sin.( x[k] ) ) 
                end
            end 
        end 
    end 

    # poly order 2 nonlinear combination with cosine functions 
    i_ind = 0 
    if poly_order >= 2 
        for i = 1 : n_vars 
            for j = i : n_vars 
                for k = 1 : n_vars 
                    ind += 1 
                    i_ind += 1 
                    push!( Θx, x -> x[i] .* x[j] .* cos.( x[k] ) ) 
                end
            end 
        end 
    end
    
    


## ============================================ ##
# poly order n_vars test 

a = check_vars_poly_deg( n_vars, 1 )
b = check_vars_poly_deg( n_vars, 2 ) 
c = check_vars_poly_deg( n_vars, 3 ) 
d = check_vars_poly_deg( n_vars, 1 ) 
e = check_vars_poly_deg( n_vars, 1 ) 
f = n_vars * n_vars 
g = n_vars * n_vars 
h = check_vars_poly_deg( n_vars, 2 ) * n_vars  
i = check_vars_poly_deg( n_vars, 2 ) * n_vars  

1 + a + b + c + d + e + f + g + h + i 














## ============================================ ##
## ============================================ ##
# below is all jake's car data stuff 


## ============================================ ##
# single run (good) 

path          = "test/data/jake_car_csvs/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end

x_err_hist  = x_err_struct([], [], [], [])
for i = eachindex(csv_files_vec) 
# for i = [ 4, 5 ]
    # i = 4 
    csv_file = csv_files_vec[i] 
    t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy = cross_validate_gpsindy( csv_file, 1 ) 

    push!( x_err_hist.sindy_lasso, norm( x_test_noise - x_test_sindy )  ) 
    push!( x_err_hist.gpsindy,     norm( x_test_noise - x_test_gpsindy )  ) 
end 

## ============================================ ## 
# single run (good) 


csv_file = "test/data/rollout_4_mod_u.csv"

Ξ_gpsindy = [] 
x_gpsindy = [] 
x_sindy   = [] 
t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy = cross_validate_gpsindy( csv_file, 1 )

using Plots 
plot( x_test_noise[:,1], x_test_noise[:,2] ) 
plot!( x_test_sindy[:,1], x_test_sindy[:,2] ) 
plot!( x_test_gpsindy[:,1], x_test_gpsindy[:,2] ) 


## ============================================ ##
# save outputs as csv 
header = [ "t", "x1_test", "x2_test", "x3_test", "x4_test", "x1_sindy", "x2_sindy", "x3_sindy", "x4_sindy", "x1_gpsindy", "x2_gpsindy", "x3_gpsindy", "x4_gpsindy" ] 

data   = [ t_test x_test_noise x_test_sindy x_test_gpsindy ]
df     = DataFrame( data,  :auto ) 
CSV.write(string("car_hw_single", ".csv"), df, header=header)


## ============================================ ##
data_noise = [ t_test x_test_noise ] 
header     = [ "t", "x1_test", "x2_test", "x3_test", "x4_test" ]
data       = [ t_test x_test_noise ]  
df         = DataFrame( data,  :auto ) 
CSV.write(string("car_hw_single_test_data", ".csv"), df, header=header)

