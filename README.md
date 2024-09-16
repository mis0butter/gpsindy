# GaussianSINDy (GPSINDy)

Discovering Equations of Motion from Data 

# Research Notes 

Junette (TODAY): 
1. work on intro (1 hr) 
2. send to David (0 hr)
3. tell Somi to improve plots (1/2 hr)
4. investigate car robot delayed control input (2 hrs)
5. go through Adam's comments and address feedback (2 hrs)
6. double-check template (1/2 hr)
7. clean up refs (1 hr)
8. ping Adam, send updated draft to him, get feedback (1 hr)
9. send updated draft to everyone 

right now: at 8 hours 

TOMORROW: 
- Adam is available until 5pm tmrw 

## 

Sept 14, 2023  

Adam: 
- work on plots, big one 
  - 3.25 in is standardize size for figure width in IEEE 
- update paper title (noise-tolerant?) 
- funding blurb - typically, IEEE puts it on bottom or in the acknowledgements 
- check the template 
- references ( will take some time ): 
  - clean up references 
  - IEEE guidelines give format, use template as closely as possible 
  - if can't find ICRA style template, default to IEEE tran template, style for bibliography 
  - remove things like URLs, links, DOIs 
  - things needed are: name, (only) year, titles, journal/publisher  
- focus on flow and a nice logical argument 
  - all the content is basically there 
- register for submission website (Adam shared site)
  - include PINs for co-authors 
  - make sure you know process - David is supposed to help me out  
  - Adam will give me more info on registration process stuff 
- after ICRA submission, put paper on ArXiV  

Junette: 
- explore delayed control input with robot car 

for Somi: 
- predator-prey x1 and x2 plots should have time at horizontal axis time (t)
- background should be white  


- 

## 

Adam: 
- Finish experiments section 
- Finish intro 
- Related work section needs more work 
- Read papers by Adam and put into related work 
  - identify one key statement that summarizes paper 
  - identify downfall of method 
  - "this is a problem that has been identified, see this papers. They propose to handle via this technique," do that over and over again 
- get to a point where we can say "our proposed approach is different and better" 
- send paper to Adam first, (ALSO FIX ABSTRACT) then send to DFK/Somi 

## Sept 9, 2023 

Somi: 
- maybe choose 20 or 25 out of 45 collections of Jake's car data 
- overlay MSE on top of error  
- box plots show mean and quartile range 
- maybe out of 45 collections of Jake's car data, one of them looks really good 

Predator and prey AND simulated unicycle: 
- 5 different noise levels 
- 5 seeds for each noise level 
- MSE 

cherry-pick best 2 examples for predator-prey and unicycle: 

## Sept 8, 2023 

Adam: 
- gradient based optimizer can be very slow, probably not good for our problem 
- discuss what SINDy is actually doing 
  - one way of promoting sparsity is by doing LASSO 
  - however LASSO is actually rarely sparse 
  - another thing to do is sequentially thresholded least squares 
  - discuss problems with STLS, including manually tuning lambda at each step to determine magnitude so that you don't cut out useful terms in your dynamics 
- car data 
  - tune lambda from 1e-5 to 1e4 increments of 10 
  - right now GPSINDy is overfitting 
  - cross-validation 
    - find different lambdas that look good, cut value of lambda in half 
  - optimizing for hyperparameters and lambda 
    - use default hyperparameters of 1.0, 1.0, 0.1, or take hyperparameters from unicycle sim test 
    - and then cross-validation, optimize for lambda 

Yue: 
- to make x smaller, add lambda_2 ||x||^2 penalty to obj fn 

Somi: 
- he'll get quadcopter data, and he'll make the plots 

## Sept 7, 2023 

Adam: 
- 4 plots: 
  - 1: predator_prey single case 
  - 2: predator_prey batch median quartile plots 
  - 3: unicycle sim 
  - 4: Jake's car data 

Somi: 
- Use neural network as another baseline (can fit into fig 1 and 2) 

## Sept 5, 2023 

Junette: 
- integrating ODEs with discrete control inputs ... 

## Aug 23, 2023 

Somi: 
- work on car data 

Adam: 
- writing process: 
- need ~25 refs 
- shared his writing guide on Slack 
- most important ref: SINDy 
  - name some other symbolic regression algorithms, problems with SINDy 
- a problem is not enough references 
  - make sure you cover all your bases 
  - bulk of related work is 2-3 paragraphs in intro 
- what template to use? ICRA or RA-L? 
  - find the right template !!!   
- read RA-L and ICRA site submission requirements CAREFULLY 
  - take notes 
- want ~4-5 figures for 6 pages 
  - title figure (motivating example), more artistic, illustrating the problem, nice cartoon  
  - tools: mathcha --> exports to latex 
  - be conservative, use what works 
  - exporting to png is fine 
  - tikz works well with latex 
  - 2-3 figures for results, at least 1 per experiment 
  - at least 2 experiments 
- if you use tables, highlight or color numbers that you want people to pay attention to 
- reviewers: 
  - if they have theoretical problem, pay attention 
  - REBUTTAL IS IMPORTANT - cross that bridge when you get there  
- get better at anticipating process, review other papers
  - the way you get asked to review is by submitting papers 
- aim for 1 conference paper and 1 journal paper per year   

## Aug 21, 2023 

ICRA DEADLINE: SEPT 15 

- PRIORITY: hitting the hardware data now: 
  - quadcopter data 
  - try single pendulum or unicycle 
  - keep trying Jake's car data !!! 
- future work? iterative GPSINDy 
  - try on actual hardware    
  - what will better mean help with? 
  - generalization  
- try real LASSO 
- SECOND: compare against baseline models 
  - Baseline A: SINDy directly 
  - Baseline B: moving average / low pass filter --> SINDy  
  - baseline: compare against polynomial fitting ? 
  - baseline: neural network 
  - baseline: "prior" dynamics 
  - does choice of GP make a difference?  
- KEEP WRITING 

## Aug 18, 2023 

- Finish methods section by this weekend 
- Start writing results section, filling out  

Somi: 
- try planar car dynamics
  - can use box car dynamics (driven by rear wheel axle from Estimation) 

## Aug 17, 2023 

My own thoughts: 
- Added headers to Xi matrix, easy to tell what is what now 
- Try unicycle dynamics again, but in simulation. Then try Jake's car data again 
- motivating examples? Use homework problems from Estimation, read more papers on robotics and use same style 

## Aug 16, 2023 

Adam: 
- said same thing as Somi: motivating example 
  - give persuasive argument why SINDy is bad and we want to use GPSINDy instead 
  - why having a good/better model is **important** 
  - application areas: dynamics fudginess causes issues 
  - safety applications    

## Aug 11, 2023 

- use double pendulum hardware data *ONGOING 
- fix validation script and plotting *DONE 
- start writing: update the methods section *ONGOING 
- try MeanLin with beta transpose *DONE (still doesn't work?)

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
            # Ξ_gpsindy2 = sindy_stls( x_GP, dx_GP2, λ ) 

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
