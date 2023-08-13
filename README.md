# GaussianSINDy (GPSINDy)

Discovering Equations of Motion from Data 

# Research Notes

## Aug 11, 2023 

- use double pendulum hardware data 
- fix validation script and plotting 
- start writing: update the methods section 
- try MeanLin with beta transpose 

paper: 
- motivate the problem diagram: have a robot or satellite where we want to learn the dynamics, think about where modeling is useful 
- show SINDy is bad 
- plot 1: motivating example (something with a robot) 
- save data, figure out best plot later 


## Aug 6, 2023 

Double Pendulum Chaotic dataset: 
  - https://ibm.github.io/double-pendulum-chaotic-dataset/ 
  - Extract and save data to folder test/double-pendulum-chaotic/


## July 28 
- (suggestion) For step 2: implement the mean function from Adam's white board as the mean function 
    - subtract Theta(x)*xi from training points dx_noise 

submit paper: 
- learned models have uncertainty 
- can take learned models with uncertainties into parametric form 
- experiments, learn dynamics 
- treat uncertainty of the learned system as disturbance on data 

this weekend: 
- try on real data 

my idea: 
- kalman filter GP-SINDy? condition learned model on new data? 


## July 27 

David: 
- implement the picture you took with your phone (GP-SINDy-GP-SINDy etc) 
- create nice plot from Lasse (fig 3) *DONE 
- table it: actually implement LASSO 
- write problem in a way to say that we used coordinate descent (with separate primal variables) 
- say that we stacked unknown variables (xi coefficients and sigma hyperparameters) into 1 unknown vector 
- and then say we used coordinate descent to split unknown into 2 split variables 
- and then do that thing that we wrote on the white board 
- try on Jake's data, just see how it looks *DONE (it looks terrible with SINDy alone) 

deadline: 
- nicer to have journal 
- decide later 

email Chante about work ending Aug 4 
email her again about lunch reimbursement 
she is responsive 


## July 26 

Somi: 
1. smooth dx_noise with x_noise as training inputs --> dx_GP
    - smooth x with t first --> x_GP (SE kernel)  
    - smooth dx with x --> dx_GP (SE kernel) 
    - try diff kernel? SE *DONE 
    - set up kernel to be function of x *DONE 
2. plug in dx_GP and x_noise into SINDy, good plots? *DONE 
3. fix ADMM stuff, only update hyperparameters at the start *DONE 
4. compare SINDy and GPSINDy *DONE 


## July 24-28  

Somi: (2)  
- try Matern52 kernel within SINDy-GP-ADMM (and maybe Matern32 kernel) 

David: (4) 
- table it for now: figure right way to do sparsity, soft thresholding? hard thresholding?
- iterate: GP --> SINDy --> GP --> SINDy --> repeat  
    - GP takes in (t,x) as input, outputs smoothed x 
    - Coefficients * function library to generate dx ? 
- (tangential) try SINDy with soft thresholding 
- create pull request, merge into main 

            # new dx 
            # dx_new = Θx * Ξ_gpsindy 

            # combine dx_noise and dx_new 
            # dx_combine = ... ? 

            # dx_GP2 = post_dist_M52I( t, t_test, dx_new ) 
            # Ξ_gpsindy2 = SINDy_test( x_GP, dx_GP2, λ ) 

Adam: (1) --> GPs 
- standardize x, add noise --> GP --> derivative of GP *set up GP dx = f(x)
- set up kernel to be function of x *DONE 

Yue: (3) --> l1 norm min, ADMM 
- remove relative tolerances for ADMM *DONE 
- SINDy, keep it the way it is  
- don't change hyperparameters every step for ADMM (do not change objective function at every iteration), maybe let ADMM run for 1000 iterations and then update hyperparameters *DONE 
- try taking out log(det(Ky)) of obj fn *DONE 

Junette: 
- just do Jake's car data *DONE SET UP SANDBOX 


## July 17-21 

- look into sparsity: increase lambda, compare gpsindy and sindy 
- decrease samples and increase noise, gpsindy works better? 
- use GP to smooth data and use as input into sindy 

fixed kernel?  
- input should be time, not x or dx as previously specified in the paper? 
- 

## June 19, 2023 
- put predator_prey plot into function in utils.jl 
- added metrics to end of predator_prey_test.jl (just opnorm) for truth vs. sindy vs. gpsindy 
- also tried just turning off one small coefficient for sindy per David's suggestion - it makes trajectory propagate much differently 
- now just put everything that you've found into a latex document ... and run more experiments 
