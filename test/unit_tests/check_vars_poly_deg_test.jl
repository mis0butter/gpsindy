using GaussianSINDy 

## ============================================ ##
# function to check the number of variables in the polynomial expansion 

ind = 0 
n_vars = 5 
poly_order = 4 

    # poly order 4 
    for i = 1 : n_vars 
        for j = i : n_vars 
            for k = j : n_vars 
                for l = k  : n_vars 
                    ind += 1            
                end 
            end 
        end 
    end 

fn_check = check_vars_poly_deg( n_vars, poly_order )   

println( "fn_check = ", fn_check ) 
println( "ind = ", ind ) 

## ============================================ ##

ind = 0 
n_vars = 10 
poly_order = 3 

    # poly order 3 
    for i = 1 : n_vars 
        for j = i : n_vars 
            for k = j : n_vars 
                ind += 1 
            end 
        end 
    end 

fn_check = check_vars_poly_deg( n_vars, poly_order )   

println( "fn_check = ", fn_check ) 
println( "ind = ", ind ) 

## ============================================ ##

ind = 0 
n_vars = 19  
poly_order = 2 

    # poly order 2 
    for i = 1 : n_vars 
        for j = i : n_vars 
            ind += 1 
        end 
    end 

fn_check = check_vars_poly_deg( n_vars, poly_order )   

println( "fn_check = ", fn_check ) 
println( "ind = ", ind ) 



## ============================================ ##

n_vars      = 19 
poly_order  = 3 


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



