import numpy as np
from scipy.optimize import root, minimize
from scipy.stats import norm
from scipy.stats import qmc

# CSTR equations with reaction kinetics k0 exp(Ea/RT) Ca^n
def concentration_compute(CA, k, ca_in=0.0, Temp=0.0, n=1.0, tau=5.0):
    return (
        1.0 / tau * (ca_in - CA[0])
        - k[0] * np.exp(-k[1] / (8.314 * Temp)) * CA[0]**n
    )

# #non-linear solver to compute the outlet concentrations for CSTR 
def CSTR_model(ca_in, Temp, tau=5.0, n=1.0, k=(0.0, 0.0)):
    sol = root(
        lambda CA: concentration_compute(CA, k, ca_in=ca_in, Temp=Temp, n=n, tau=tau),
        x0=[0.1],
        tol=1e-15
    )
    return sol.x


#random points generator for one variable. For more variables enhance the size of lb and ub
def random_points_generator(Nexps, lb=(0.0,), ub=(1.0,)):
    sampler = qmc.Halton(d=len(lb), scramble=False)
    sample = sampler.random(Nexps)
    return qmc.scale(sample, lb, ub).flatten()

#add noise to the outlet concentrations if required
def ca_exp(
    ca_ins,
    Temp,
    Nexps=0,
    k=(1.0, 20000.0),
    add_noise=False,
    sigma=1e-6,
    N_repeats=5,
    order=1.0
):
    ca_out_without_noise = np.zeros(Nexps)
    ca_out_matrix = np.zeros((N_repeats, Nexps))

    for i in range(Nexps):
        ca_out_without_noise[i] = CSTR_model(
            ca_ins[i], Temp[i], k=k, n=order
        )[0]

        noise = norm.rvs(0, sigma if add_noise else 0.0, size=N_repeats)
        ca_out_matrix[:, i] = ca_out_without_noise[i] + noise

    return np.mean(ca_out_matrix, axis=0)

#model CSTR equations 
def ca_model(k, ca_in=0.0, Temp=0.0, n=1.0, tau=5.0):
    sol = root(
        lambda CA: concentration_compute(CA, k, ca_in=ca_in, Temp=Temp, n=n, tau=tau),
        x0=[0.1],
        tol=1e-15
    )
    return sol.x

#Function to estimate the reaction parameters  
def parameter_estimator(Nexps=10,
    ca_exp=None,
    ca_in=None,
    Temp=None,
    order=1,
    initial_guess=(0.0, 0.0)):

    def objective(k):
        if k[0] <= 0 or k[1] <= 0:
            return 1e20

        err = 0.0
        for i in range(Nexps):
            CA = ca_model(k, ca_in=ca_in[i], Temp=Temp[i], n=order)[0]
            if np.isnan(CA):
                return 1e20
            err += (ca_exp[i] - CA)**2
        return err

    res = minimize(
        objective,
        x0=initial_guess,
        method="Nelder-Mead",
        options={"maxiter": 5000, "disp": True}
    )

    print("Estimated parameters:", res.x)
    return res.x #changed by me to support step 10

def run_parameter_estimation(
    Nexps=5,
    add_noise=True,
    sigma=1e-10,
    order=1
):
    Temp = random_points_generator(Nexps, lb=(200.0,), ub=(400.0,)) #create random values of temperature for different experiments
    ca_ins = random_points_generator(Nexps, lb=(0.1,), ub=(1.0,)) #create random values of ca_ins for different experiments

    ca_exper = ca_exp(
        ca_ins,
        Temp,
        Nexps=Nexps,
        k=(1.0, 20000.0),
        add_noise=add_noise,
        sigma=sigma,
        order=order
    ) #replicate experiments by adding noise 
    
    parameter_estimator(
        Nexps=Nexps,
        ca_exp=ca_exper,
        ca_in=ca_ins,
        Temp=Temp,
        order=order,
        initial_guess=(0.1, 10.0)
    ) #estimate the paraemteres using the above computed outlet concentrations 

if __name__ == "__main__":
    run_parameter_estimation(
        Nexps=5,
        add_noise=False,
        sigma=1e-3,
        order=1
    )


