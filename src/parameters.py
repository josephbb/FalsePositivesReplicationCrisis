

#Simulation parameters
res = 40
n = 100

#General parameters
alpha = .05

#Figure parameters
fig_2_tau = .2
fig_3_tau = .2

#Centralized place to retitle and relabel figures or change color schemes
sign_error_cmap = "ch:start=2.8,rot=.3"
magnitude_error_cmap = "ch:start=1.3,rot=-.1"
pub_cmap = "mako"
rep_cmap = "rocket"

#labels
tau_label = r"$\tau$" + " (effect size)"
sigma_label = r"$\sigma$" + " (varying effects)"
sample_size_label = "Sample Size"

#Titles
sign_title = "Pr(Sign Error)"
magnitude_title = "Magnitude Error"
publish_title = "Pr(publish)\n"
rep_title = "Pr(replicate)\n"

#Phacking simulation parameters
phacking_sim_size = 10000
p_hacking_mu = .05
p_hacking_sigma = .1 
p_hacking_n = 20

font_scale=1.25