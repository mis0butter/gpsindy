using Statistics 
using CSV 
using DataFrames 

## ============================================ ##
# extract Jake's car data 

export extract_car_data 
function extract_car_data(  ) 

    csv_file = "test/data/jake_robot_data.csv" 

    # wrap in data frame --> Matrix 
    df   = CSV.read(csv_file, DataFrame) 
    data = Matrix(df) 
    
    # extract variables 
    t = data[:,1] 
    x = data[:,2:end-2]
    u = data[:,end-1:end] 

    return t, x, u 
end 

## ============================================ ##
# get sizes of things 

export size_x_n_vars 
function size_x_n_vars( x, u ) 

    x_vars = size(x, 2)
    u_vars = size(u, 2) 
    poly_order = x_vars 
    n_vars = x_vars + u_vars 

    return x_vars, u_vars, poly_order, n_vars 
end 

## ============================================ ##
# rollover indices 

export unroll 
function unroll( t, x ) 

    # use forward finite differencing 
    dx_fd = fdiff(t, x, 1) 

    # massage data, generate rollovers  
    rollover_up_ind = findall( x -> x > 100, dx_fd[:,4] ) 
    rollover_dn_ind = findall( x -> x < -100, dx_fd[:,4] ) 
    for i = 1 : length(rollover_up_ind) 

        i0   = rollover_up_ind[i] + 1 
        ifin = rollover_dn_ind[i] 
        rollover_rng = x[ i0 : ifin , 4 ]
        dθ = π .- rollover_rng 
        θ  = -π .- dθ 
        x[ i0 : ifin , 4 ] = θ

    end 
    
    # use central finite differencing now  
    dx_fd = fdiff(t, x, 2) 

    return x, dx_fd 
end 

## ============================================ ##
# standardize data for x and dx 

export stand_data 
function stand_data( t, x ) 

    n_vars = size(x, 2) 
    
    # loop through states 
    x_stand = 0 * x 
    for i = 1:n_vars 
        x_stand[:,i] = ( x[:,i] .- mean( x[:,i] ) ) ./ std( x[:,i] )
    end 
    
    return x_stand 
end 

## ============================================ ##
# convert vector of vectors into matrix 

export vv2m 
function vv2m( vecvec )

    mat = mapreduce(permutedims, vcat, vecvec)

    return mat 
end 

## ============================================ ##
# split into training and validation data 

export split_train_test 
function split_train_test(x, test_fraction, portion)

    # if test data = LAST portion 
    if portion == 1/test_fraction 

        ind = Int(round( size(x,1) * (1 - test_fraction) ))   

        x_train = x[1:ind,:]
        x_test  = x[ind:end,:] 

    # if test data = FIRST portion 
    elseif portion == 1 

        ind = Int(round( size(x,1) * (test_fraction) ))  

        x_train = x[ind:end,:] 
        x_test  = x[1:ind,:]

    # test data is in MIDDLE portion 
    else 

        ind1 = Int(round( size(x,1) * (test_fraction*( portion-1 )) )) 
        ind2 = Int(round( size(x,1) * (test_fraction*( portion )) )) 

        x_test  = x[ ind1:ind2,: ]
        x_train = [ x[ 1:ind1,: ] ; x[ ind2:end,: ] ]

    end 

    return x_train, x_test 

end 


## ============================================ ##

export min_d_max
function min_d_max( x )

    xmin = round( minimum(x), digits = 1 )  
    xmax = round( maximum(x), digits = 1 )
    # xmin = minimum(x) 
    # xmax = maximum(x)  
    # dx   = round( ( xmax - xmin ) / 2, digits = 1 ) 
    dx = ( xmax - xmin ) / 2 

    return xmin, dx, xmax  

end 


## ============================================ ##
# derivatives: finite difference  

export fdiff 
function fdiff(t, x, fd_method) 

    # forward finite difference 
    if fd_method == 1 

        dx_fd = 0*x 
        for i = 1 : length(t)-1
            dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / ( t[i+1] - t[i] )
        end 

        # deal with last index 
        dx_fd[end,:] = ( x[end,:] - x[end-1,:] ) / ( t[end] - t[end-1] )

    # central finite difference 
    elseif fd_method == 2 

        dx_fd = 0*x 
        for i = 2 : length(t)-1
            dx_fd[i,:] = ( x[i+1,:] - x[i-1,:] ) / ( t[i+1] - t[i-1] )
        end 

        # deal with 1st index 
        i = 1 
        dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / ( t[i+1] - t[i] )

        # deal with last index 
        dx_fd[end,:] = ( x[end,:] - x[end-1,:] ) / ( t[end] - t[end-1] )

    # backward finite difference 
    else 

        dx_fd = 0*x 
        for i = 2 : length(t)
            dx_fd[i,:] = ( x[i,:] - x[i-1,:] ) / ( t[i] - t[i-1] )
        end 

        # deal with 1st index 
        i = 1 
        dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / ( t[i+1] - t[i] )

    end 

    return dx_fd 

end 

