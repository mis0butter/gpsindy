## ============================================ ##
# putting it together (no control) 

export sindy_stls 
function sindy_stls( x, dx, λ, u = false )

    x_vars, u_vars, poly_order, n_vars = size_x_n_vars( dx, u ) 
    poly_order = 1 

    
    if isequal(u, false)      # if u_data = false 
        data   = x 
    else            # there are u_data inputs 
        data   = [ x u ]
    end 

    # construct data library 
    Θx = pool_data_test(data, n_vars, poly_order) 

    # SINDy 
    Ξ = sparsify_dynamics_stls( Θx, dx, λ, x_vars ) 

    return Ξ

end 

## ============================================ ##
# putting it together (no control) 

export sindy_lasso 
function sindy_lasso( x, dx, λ, u = false )

    x_vars, u_vars, poly_order, n_vars = size_x_n_vars( dx, u ) 
    
    if isequal(u, false)      # if u_data = false 
        data = x 
    else            # there are u_data inputs 
        data = [ x u ] 
    end 

    # construct data library 
    Θx = pool_data_test(data, n_vars, poly_order) 

    # SINDy 
    Ξ = sparsify_dynamics_lasso( Θx, dx, λ, x_vars ) 

    return Ξ

end 


# ## ============================================ ##
# # putting it together (with control) 

# export SINDy_c_test 
# function SINDy_c_test( x, u, dx, λ )

#     x_vars = size(x, 2)
#     u_vars = size(u, 2) 
#     n_vars = x_vars + u_vars 
#     poly_order = x_vars 

#     # construct data library 
#     Θx = pool_data_test( [x u], n_vars, poly_order ) 

#     # first cut - SINDy 
#     # Ξ = sparsify_dynamics_stls( Θx, dx, λ, x_vars ) 
#     Ξ = sparsify_dynamics_cstrnd( Θx, dx, λ, x_vars ) 

#     return Ξ 

# end 


## ============================================ ##
# solve sparse regression 

export sparsify_dynamics_lasso 
function sparsify_dynamics_lasso( Θx, dx, λ, n_vars ) 
# ----------------------- #
# Purpose: Solve for active terms in dynamics through sparse regression 
# 
# Inputs: 
#   Θx     = data matrix (of input states) 
#   dx     = state derivatives 
#   lambda = sparsification knob (threshold) 
#   n_vars = # elements in state 
# 
# Outputs: 
#   Ξ      = sparse coefficients of dynamics 
# ----------------------- #

    # first perform least squares 
    Ξ = Θx \ dx 

    # for each element in state 
    for j = 1 : n_vars 

        x, z, hist = lasso_admm( Θx, dx[:,j], λ ) 
        Ξ[:, j]    = z  

    end 

    return Ξ

end 


## ============================================ ##
# solve sparse regression 

export sparsify_dynamics_stls 
function sparsify_dynamics_stls( Θx, dx, λ, n_vars ) 
# ----------------------- #
# Purpose: Solve for active terms in dynamics through sparse regression 
# 
# Inputs: 
#   Θx     = data matrix (of input states) 
#   dx     = state derivatives 
#   lambda = sparsification knob (threshold) 
#   n_vars = # elements in state 
# 
# Outputs: 
#   Ξ      = sparse coefficients of dynamics 
# ----------------------- #

    # first perform least squares 
    Ξ = Θx \ dx 

    # sequentially thresholded least squares = LASSO. Do 10 iterations 
    for k = 1 : 10 

        # for each element in state 
        for j = 1 : n_vars 

            # small_inds = rows of |Ξ| < λ
            small_inds = findall( <(λ), abs.(Ξ[:,j]) ) 

            # set elements < λ to 0 
            Ξ[small_inds, j] .= 0 

            # big_inds --> select columns of Θx
            big_inds = findall( >=(λ), abs.( Ξ[:,j] ) ) 

            # regress dynamics onto remaining terms to find sparse Ξ
            Ξ[big_inds, j] = Θx[:, big_inds] \ dx[:,j] 

        end 

    end 
    
    return Ξ
end 


## ============================================ ##
# define min function 

export min_Ax_b 
function min_Ax_b( A, b, bound = 10 ) 

    x0 = 0 * ( A \ b ) 

    # Optim stuff 
    upper = bound * ones(size(x0)) 
    lower = -upper 
    f_test(x) = 1/2 * norm(A*x - b)^2 

    # optimization 
    od     = OnceDifferentiable( f_test, x0 ; autodiff = :forward ) 
    result = optimize( od, lower, upper, x0, Fminbox(LBFGS()) ) 
    x_opt  = result.minimizer 

    return x_opt 

end 


## ============================================ ##
# solve sparse regression 

export sparsify_dynamics_cstrnd 
function sparsify_dynamics_cstrnd( Θx, dx, λ, x_vars ) 
# ----------------------- #
# Purpose: Solve for active terms in dynamics through sparse CONSTRAINED regression 
# 
# Inputs: 
#   Θx     = data matrix (of input states) 
#   dx     = state derivatives 
#   lambda = sparsification knob (threshold) 
#   n_vars = # elements in state 
# 
# Outputs: 
#   Ξ      = sparse coefficients of dynamics 
# ----------------------- #

    # first perform least squares 
    Ξ = Θx \ dx 

    # sequentially thresholded least squares = LASSO. Do 10 iterations 
    for k = 1 : 10 

        # for each element in state 
        for j = 1 : x_vars 

            # small_inds = rows of |Ξ| < λ
            small_inds = findall( <(λ), abs.(Ξ[:,j]) ) 

            # set elements < λ to 0 
            Ξ[small_inds, j] .= 0 

            # big_inds --> select columns of Θx
            big_inds = findall( >=(λ), abs.( Ξ[:,j] ) ) 

            # regress dynamics onto remaining terms to find sparse Ξ
            A     = Θx[:, big_inds] 
            b     = dx[:,j] 
            x_opt = min_Ax_b( A, b ) 
            Ξ[big_inds, j] = x_opt 

        end 

    end 
        
    return Ξ

end 


## ============================================ ##
# build data matrix 

export pool_data_test
function pool_data_test(xu_mat, n_vars, poly_order) 
# ----------------------- #
# Purpose: Build data matrix based on possible functions 
# 
# Inputs: 
#   xu_mat      = data input 
#   n_vars      = # elements in state and/or control  
#   poly_order  = polynomial order (goes up to order 3) 
# 
# Outputs: 
#   Θx          = data matrix passed through function library 
# ----------------------- #

    # turn x into matrix and get length 
    # xmat = mapreduce(permutedims, vcat, x) 
    l = size(xu_mat, 1) 

    # # fill out 1st column of Θx with ones (poly order = 0) 
    ind = 1 ; 
    Θx  = ones(l, ind) 
    
    # poly order 1 
    for i = 1 : n_vars 
        ind += 1 
        Θx   = [ Θx xu_mat[:,i] ]
    end 

    # # poly order 2 
    # if poly_order >= 2 
    #     for i = 1 : n_vars 
    #         for j = i : n_vars 
    #             ind += 1 ; 
    #             vec  = xu_mat[:,i] .* xu_mat[:,j] 
    #             Θx   = [Θx vec] 
    #         end 
    #     end 
    # end 

    # # poly order 3 
    # if poly_order >= 3 
    #     for i = 1 : n_vars 
    #         for j = i : n_vars 
    #             for k = j : n_vars 
    #                 ind += 1 ;                     
    #                 vec  = xmat[:,i] .* xmat[:,j] .* xmat[:,k] 
    #                 Θx   = [Θx vec] 
    #             end 
    #         end 
    #     end 
    # end 

    # sine functions 
    for i = 1 : n_vars 
        ind  += 1 
        vec   = sin.(xu_mat[:,i]) 
        Θx    = [Θx vec] 
    end 

    # cos functions 
    for i = 1 : n_vars 
        ind  += 1 
        vec   = cos.(xu_mat[:,i]) 
        Θx    = [Θx vec] 
    end 

    # nonlinear combination with sine functions 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind  += 1 
            vec   = xu_mat[:,i] .* sin.(xu_mat[:,j]) 
            Θx    = [Θx vec]     
        end 
    end 

    # nonlinear combination with cosine functions 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind  += 1 
            vec   = xu_mat[:,i] .* cos.(xu_mat[:,j]) 
            Θx    = [Θx vec]     
        end 
    end 

    # # poly order 2 nonlinear combination with sine functions 
    # if poly_order >= 2 
    #     for i = 1 : n_vars 
    #         for j = i : n_vars 
    #             for k = 1 : n_vars 
    #                 ind  += 1 
    #                 vec   = xmat[:,i] .* xmat[:,j] .* sin.(xmat[:,k]) 
    #                 Θx    = [Θx vec]     
    #             end
    #         end 
    #     end 
    # end 

    # # poly order 2 nonlinear combination with cosine functions 
    # if poly_order >= 2 
    #     for i = 1 : n_vars 
    #         for j = i : n_vars 
    #             for k = 1 : n_vars 
    #                 ind  += 1 
    #                 vec   = xmat[:,i] .* xmat[:,j] .* cos.(xmat[:,k]) 
    #                 Θx    = [Θx vec]     
    #             end
    #         end 
    #     end 
    # end 

    return Θx  

end 


## ============================================ ##
# build data matrix 

export pool_data_vecfn_test
function pool_data_vecfn_test(n_vars, poly_order) 
# ----------------------- #
# Purpose: Build data vector of functions  
# 
# Inputs: 
#   n_vars      = # elements in state 
#   poly_order  = polynomial order (goes up to order 3) 
# 
# Outputs: 
#   Θ       = data matrix passed through function library 
# ----------------------- #
    
    # initialize empty vector of functions 
    Θx = Vector{Function}(undef,0) 

    # fil out 1st column of Θ with ones (poly order = 0) 
    ind  = 1 
    push!(Θx, x -> 1) 

    # poly order 1 
    for i = 1 : n_vars 
        ind  += 1 
        push!( Θx, x -> x[i] ) 
    end 

    # # poly order 2 
    # if poly_order >= 2 
    #     for i = 1 : n_vars 
    #         for j = i:n_vars 
    #             ind += 1 ; 
    #             push!( Θx, x -> x[i] .* x[j] ) 
    #         end 
    #     end 
    # end 

    # # poly order 3 
    # if poly_order >= 3 
    #     for i = 1 : n_vars 
    #         for j = i : n_vars 
    #             for k = j : n_vars 
    #                 ind += 1 ;                     
    #                 push!( Θx, x -> x[i] .* x[j] .* x[k] )
    #             end 
    #         end 
    #     end 
    # end 

    # sine functions 
    for i = 1 : n_vars 
        ind  += 1
        push!(Θx, x -> sin.( x[i] ) )
    end 

    # sine functions 
    for i = 1 : n_vars 
        ind  += 1
        push!(Θx, x -> cos.( x[i] ) ) 
    end 

    # nonlinear combinations with sine functions 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind += 1 
            push!( Θx, x -> x[i] .* sin.( x[j] ) ) 
        end 
    end 

    # nonlinear combinations with cosine functions 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind += 1 
            push!( Θx, x -> x[i] .* cos.( x[j] ) ) 
        end 
    end 

    # # poly order 2 nonlinear combination with sine functions 
    # if poly_order >= 2 
    #     for i = 1 : n_vars 
    #         for j = i : n_vars 
    #             for k = 1 : n_vars 
    #                 ind += 1 
    #                 push!( Θx, x -> x[i] .* x[j] .* sin.( x[k] ) ) 
    #             end
    #         end 
    #     end 
    # end 

    # # poly order 2 nonlinear combination with cosine functions 
    # if poly_order >= 2 
    #     for i = 1 : n_vars 
    #         for j = i : n_vars 
    #             for k = 1 : n_vars 
    #                 ind += 1 
    #                 push!( Θx, x -> x[i] .* x[j] .* cos.( x[k] ) ) 
    #             end
    #         end 
    #     end 
    # end
    
    return Θx 

end 


## ============================================ ##
# export terms (with control)

export nonlinear_terms 
function nonlinear_terms( x_data, u_data = false ) 

    terms = [] 
    x_vars = size(x_data, 2) 
    u_vars = size(u_data, 2) 

    var_string = [] 
    for i = 1 : x_vars 
        push!( var_string, string("x", i) )
    end  
    if isequal(u_data, false)      # if u_data = false 
        n_vars = x_vars 
    else            # there are u_data inputs 
        n_vars = x_vars + u_vars 
        for i = 1 : u_vars 
            push!( var_string, string("u", i) ) 
        end 
    end 
    
    # first one 
    ind = 1  
    push!( terms, 1 )
    
     # poly order 1 
    for i = 1 : n_vars 
        ind += 1 
        push!( terms, var_string[i] ) 
    end 
    
    # poly order 2 
    for i = 1 : n_vars 
        for j = i : n_vars 
            ind += 1 
            push!( terms, string( var_string[i], var_string[j] ) ) 
        end 
    end 
    
     # poly order 3 
    for i = 1 : n_vars 
        for j = i : n_vars 
            for k = j : n_vars 
                ind += 1 
                push!( terms, string( var_string[i], var_string[j], var_string[k] ) )     
            end 
        end 
    end 
    
     # sine functions 
    for i = 1 : n_vars 
        ind += 1 
        push!( terms, string( "sin(", var_string[i], ")" ) ) 
    end 
    
     for i = 1 : n_vars 
        ind += 1 
        push!( terms, string( "cos(", var_string[i], ")" ) ) 
    end 
    
     # cosine functions 
    for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind += 1 
            push!( terms, string( var_string[i], "sin(", var_string[j], ")") ) 
        end 
    end 
    
     for i = 1 : n_vars 
        for j = 1 : n_vars 
            ind += 1 
            push!( terms, string( var_string[i], "cos(", var_string[j], ")") ) 
        end 
    end 

    # poly order 2 nonlinear combination with sine functions 
    for i = 1 : n_vars 
        for j = i : n_vars 
            for k = 1 : n_vars 
                ind += 1 
                push!( terms, string( var_string[i], var_string[j], "sin(", var_string[k], ")") ) 
            end
        end 
    end 

    # poly order 2 nonlinear combination with cosine functions 
    for i = 1 : n_vars 
        for j = i : n_vars 
            for k = 1 : n_vars 
                ind += 1 
                push!( terms, string( var_string[i], var_string[j], "cos(", var_string[k], ")") ) 
            end
        end 
    end 

    return terms 
end 


## ============================================ ##

export pretty_coeffs 
function pretty_coeffs(Ξ_true, x_true, u = false)

    # compute all nonlinear terms 
    terms = nonlinear_terms( x_true, u ) 

    # header 
    n_vars = size(x_true, 2) 
    header = [ "term" ] 
    for i = 1 : n_vars 
        push!( header, string( "x", i, "dot" ) ) 
    end 
    header = permutedims( header[:,:] ) 

    # unique inds of nonzero rows 
    n_vars = size(x_true, 2) 
    inds = [] 
    for i = 1 : n_vars 
        ind  = findall( x -> abs(x) > 0, Ξ_true[:,i] ) 
        inds = [ inds ; ind ]
    end 
    inds = sort(unique(inds)) 

    # save nzero rows 
    Ξ_nzero     = Ξ_true[inds, :]
    terms_nzero = terms[inds, :]
    
    # build Ξ with headers of nonzero rows 
    sz      = size(Ξ_nzero) 
    Ξ_terms = Array{Any}( undef, sz .+ (1,1) ) 
    Ξ_terms[1,:]          = header 
    Ξ_terms[2:end, 1]     = terms_nzero 
    Ξ_terms[2:end, 2:end] = Ξ_nzero   

    return Ξ_terms 

end 


