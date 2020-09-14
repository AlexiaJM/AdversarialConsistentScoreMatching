## Calculate number of steps using Technique 2 from https://arxiv.org/abs/2006.09011
## sigma_min should always be .01
## Use "main.py --compute_approximate_sigma_max" to get sigma_max
## D is the dimension
## Goal should be .50 or .99

# input
D = 3*32*32
sigma_min = .01
sigma_max = 50
goal = .50

# number of steps
get_num_steps = function(goal, D, sigma_min, sigma_max){
	x = seq(0,1,by=.000001)
	i = which.min((pnorm(sqrt(2*D)*(x-1)+3*x)-pnorm(sqrt(2*D)*(x-1)-3*x)-goal)^2)
	alpha = x[i]
	n_steps = log(sigma_min/sigma_max)/log(alpha)
	return(n_steps)
}

## CIFAR-10
get_num_steps(.5, 3*32*32, .01, 50) # around 225 steps for C = .50
get_num_steps(.99, 3*32*32, .01, 50) # around 1k steps for C = .99

## LSUN-Churches
get_num_steps(.5, 3*64*64, .01, 140) # around 500 steps for C = .50
get_num_steps(.99, 3*64*64, .01, 140) # around 2.2k steps for C = .99
