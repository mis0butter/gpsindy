using Optim 


## ============================================ ##

function min_Ax_b( A, b ) 

    x0 = 0 * ( A \ b ) 

    # Optim stuff 
    upper = 10 * ones(size(x0)) 
    lower = -upper 
    f_test(x) = 1/2 * norm(A*x - b)^2 

    # optimization 
    od     = OnceDifferentiable( f_test, x0 ; autodiff = :forward ) 
    result    = optimize( od, lower, upper, x0, Fminbox(LBFGS()) ) 
    # result = optimize( od, x0, LBFGS() ) 
    x_opt  = result.minimizer 

    return x_opt 

end 


## ============================================ ##
# setup 

m  = 15            # number of examples 
n  = 50            # number of features 
p  = 10/n           # sparsity density 
A  = randn(m,n) 
b  = randn(m,1) 

λ  = 0.1 
x  = rand(n,1) 

x_opt = min_Ax_b( A, b ) 


## ============================================ ##


m  = 15            # number of examples 
n  = 50            # number of features 
p  = 10/n           # sparsity density 
Θx = randn(m,n) 
dx = randn(m,1) 

λ  = 0.1 

# first perform least squares 
Ξ = Θx \ dx 
# sequentially thresholded least squares = LASSO. Do 10 iterations 
for k = 1 : 10 

    # for each element in state 
    for j = 1 : n_vars 
    # j = 1 
    # k = 1 

        # small_inds = rows of |Ξ| < λ
        small_inds = findall( <(λ), abs.(Ξ[:,j]) ) 

        # set elements < λ to 0 
        Ξ[small_inds, j] .= 0 

        # big_inds --> select columns of Θx
        big_inds = findall( >=(λ), abs.( Ξ[:,j] ) ) 

        # regress dynamics onto remaining terms to find sparse Ξ
        A  = Θx[:, big_inds] 
        b  = dx[:,j] 
        
        x_opt = min_Ax_b( A, b ) 

        Ξ[big_inds, j] = x_opt 

    end 

end 


## ============================================ ##


