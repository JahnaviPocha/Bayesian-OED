# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:12:18 2026

@author: jahna
"""

# ============================================================
# PART 1 — INTERFACE (YOUR JULIA BINDINGS)
# ============================================================


import numpy as np
import matplotlib.pyplot as plt


from call_to_KPE_code import (
    rpg,
    experiments,
    parameter_estimator
)

# ============================================================
# GLOBAL SETTINGS (EASY TO MODIFY)
# ============================================================

NSPEC = 3   # number of species (change anytime)

# Bounds from Julia main.jl
# lb = [0.1, 0.1, 300]
# ub = [0.5, 0.5, 600]

Y_BOUNDS = [(0.1, 0.5) for _ in range(NSPEC - 1)]
TEMP_BOUNDS = (300.0, 600.0)

# Target species index (0-based Python indexing)
# For output (3rd species)
TARGET_SPECIES_INDEX = 2

# True kinetic parameters (used in experiments)
TRUE_K = [4000.0, 4000.0]

# Fixed parameters (same as Julia)
P_TOTAL = 50
RATIO = 0.1
STD_DATA = 1e-6
N_REPEATS = 10


STOICHIOMETRY = np.array([[-2, -1, 2]])
IG = np.array([1000.0, 1000.0])

# ============================================================
# HELPER: COMPLETE MASS FRACTION VECTOR
# ============================================================

def complete_Y_in(Y_partial):
    """
    Convert partial species vector → full Y_in
    Enforces sum(Y) = 1
    """
    last = 1.0 - np.sum(Y_partial)
    
    # Safety check
    if last < 0:
        return None
    
    return np.append(Y_partial, last)


# ============================================================
# HELPER: CONVERT BO VECTOR → PHYSICAL INPUT
# ============================================================

def decode_design_vector(x):
    """
    x = [Y1, Y2, ..., Y(n-1), Temp]
    """
    Y_partial = x[:-1]
    Temp = x[-1]

    Y_full = complete_Y_in(Y_partial)

    if Y_full is None:
        return None, None

    return Y_full, Temp


# ============================================================
# PART 2 — FISHER INFORMATION OBJECTIVE
# ============================================================

def fisher_information_objective(x, n_repeats=10):

    Y_in, Temp = decode_design_vector(x)

    if Y_in is None:
        return -1e12

    Y_in = np.array(Y_in).reshape(NSPEC, 1)
    Temp = np.array([Temp])

    k_samples = []

    for _ in range(n_repeats):

        try:
            Yexp = experiments(
                Y_in=Y_in,
                Temp=Temp,
                P_total=P_TOTAL,
                Nexps=1,
                ratio=RATIO,
                N_repeats=N_REPEATS,
                std_data=STD_DATA,
                Nspec=NSPEC,
                k_true=TRUE_K
            )

            k = parameter_estimator(
                ratio=RATIO,
                nspec=NSPEC,
                Y_in=Y_in,
                Temp=Temp,
                P_total=P_TOTAL,
                St=STOICHIOMETRY,
                nref=2500,
                nreac=1,
                Nexps=1,
                Y_out=Yexp,
                unknown_parameters=2,
                IG=IG,
                N_repeats=N_REPEATS,
                σ_data=STD_DATA,
                RBS_full=False
            )

            k = np.array(k, dtype=float)

            if np.any(np.isnan(k)) or np.any(np.isinf(k)):
                continue

            k_samples.append(k)

        except Exception:
            continue

    if len(k_samples) < 2:
        return -1e12

    K = np.array(k_samples)

    cov = np.cov(K.T)

    cov = cov + 1e-10 * np.eye(cov.shape[0])

    # Fisher proxy (stable form)
    try:
        score = np.log(np.linalg.det(cov))
    except:
        return -1e12

    return -score


# ============================================================
# PART 3 — BO SETUP (Gaussian Process + EI)
# ============================================================

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm


def scale_X(X):

    X = np.array(X)
    Xs = np.zeros_like(X)

    # Y1
    Xs[:,0] = (X[:,0] - 0.1) / (0.5 - 0.1)

    # Y2
    Xs[:,1] = (X[:,1] - 0.1) / (0.5 - 0.1)

    # Temp
    Xs[:,2] = (X[:,2] - 300) / (600 - 300)

    return Xs
# ============================================================
# PART 4 — GP MODEL
# ============================================================

def train_gp(X, y):

    Xs = scale_X(X)

    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(3))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=3
    )

    gp.fit(Xs, y)

    return gp


# ============================================================
# PART 5 — EXPECTED IMPROVEMENT
# ============================================================

def EI(Xcand, gp, y_best):

    Xs = scale_X(Xcand)

    mu, sigma = gp.predict(Xs, return_std=True)

    sigma = np.maximum(sigma, 1e-9)

    Z = (mu - y_best) / sigma

    ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)

    ei[sigma == 0] = 0

    return ei


# ============================================================
# PART 6 — CANDIDATE GENERATION
# ============================================================


def generate_candidates(n_candidates=100):

    candidates = []

    while len(candidates) < n_candidates:

        Y_partial = [
            np.random.uniform(*Y_BOUNDS[i])
            for i in range(NSPEC - 1)
        ]

        Temp = np.random.uniform(*TEMP_BOUNDS)

        x = np.array(Y_partial + [Temp])

        Y_full, _ = decode_design_vector(x)

        if Y_full is not None:
            candidates.append(x)

    return np.array(candidates)

# ============================================================
# PART 7 — SELECT NEXT EXPERIMENT
# ============================================================
def select_next(gp, X, y):

    Xcand = generate_candidates()

    ei = EI(Xcand, gp, np.max(y))

    return Xcand[np.argmax(ei)]


# ============================================================
# PART 8 — estimates parameters
# ============================================================

def estimate_parameters_from_design(X):

    Nexps = len(X)

    Y_in_all = []
    Temp_all = []

    for x in X:

        Yin, Temp = decode_design_vector(x)

        Y_in_all.append(Yin)
        Temp_all.append(Temp)

    Y_in_all = np.array(Y_in_all).T
    Temp_all = np.array(Temp_all)

    Y_out = experiments(
        Y_in=Y_in_all,
        Temp=Temp_all,
        P_total=P_TOTAL,
        Nexps=Nexps,
        ratio=RATIO,
        N_repeats=N_REPEATS,
        std_data=STD_DATA,
        Nspec=NSPEC,
        k_true=TRUE_K
    )

    k = parameter_estimator(
        ratio=RATIO,
        nspec=NSPEC,
        Y_in=Y_in_all,
        Temp=Temp_all,
        P_total=P_TOTAL,
        St=STOICHIOMETRY,
        nref=2500,
        nreac=1,
        Nexps=Nexps,
        Y_out=Y_out,
        unknown_parameters=2,
        IG=IG,
        N_repeats=N_REPEATS,
        σ_data=STD_DATA,
        RBS_full=False
    )

    return np.array(k,dtype=float)

# ============================================================
# PART 9 — BO LOOP (CORE)
# ============================================================
def BO(
        N_init=3,
        N_iter=10
):

    X = generate_candidates(N_init)

    y = []

    param_history = []

    print("\n=== INITIAL EXPERIMENTS ===")

    for x in X:

        val = fisher_information_objective(x)

        y.append(val)

        print("x =", x)
        print("info =", val)

    y = np.array(y)

    # Initial parameter estimate
    k = estimate_parameters_from_design(X)

    param_history.append(k)

    print("Initial k =", k)

    history = [np.copy(y)]

    print("\n=== BO START ===")

    for i in range(N_iter):

        gp = train_gp(X,y)

        x_new = select_next(gp,X,y)

        y_new = fisher_information_objective(x_new)

        X = np.vstack([X,x_new])

        y = np.append(y,y_new)

        # Estimate parameters again
        k = estimate_parameters_from_design(X)

        param_history.append(k)

        history.append(np.copy(y))

        print("\n--------------------------------")
        print(f"Iteration {i+1}")
        print("--------------------------------")
        print("New design =",x_new)
        print("Information =",y_new)
        print("k1 =",k[0])
        print("k2 =",k[1])

    return X,y,history,np.array(param_history)

# ============================================================
# PART 10 — Noise Study Driver
# ============================================================

def run_noise_study():

    noises = [3,5,8,13]

    results = {}

    global N_REPEATS

    for noise in noises:

        print("\n")
        print("="*60)
        print(f"RUNNING NOISE CASE {noise}")
        print("="*60)

        N_REPEATS = noise

        X,y,hist,param_hist = BO(
            N_init=3,
            N_iter=10
        )

        results[noise] = {
            "X":X,
            "y":y,
            "history":hist,
            "param_history":param_hist
        }

        final_k = param_hist[-1]

        best_idx = np.argmax(y)

        print("\n===== FINAL RESULTS SUMMARY =====")
        print()
        print("Total experiments used:",len(X))
        print()
        print("Final estimated parameters:")
        print(f"k1 = {final_k[0]:.6f}")
        print(f"k2 = {final_k[1]:.6f}")
        print()
        print("Best experiment:")
        print(X[best_idx])
        print()
        print(f"Information = {y[best_idx]:.6f}")

    return results

# ============================================================
# PART 11 — PLOTS
# ============================================================


# ============================================================
# PART 11.1 — k1 Plot
# ============================================================ 

def plot_k1_convergence_all(results):

    plt.figure(figsize=(9,6))

    for noise,data in results.items():

        params = data["param_history"]

        plt.plot(
            np.arange(1,len(params)+1),
            params[:,0],
            marker='o',
            linewidth=2,
            label=f"Noise={noise}"
        )

    plt.axhline(
        4000,
        color='black',
        linestyle='--',
        linewidth=2,
        label='True k1'
    )

    plt.xlabel("Experiments Used")
    plt.ylabel("Estimated k1")
    plt.title("k1 Convergence")
    plt.legend()
    plt.grid(True)

    plt.show()

# ============================================================
# PART 11.2 — k2 Plot
# ============================================================ 
    
    
def plot_k2_convergence_all(results):

    plt.figure(figsize=(9,6))

    for noise,data in results.items():

        params = data["param_history"]

        plt.plot(
            np.arange(1,len(params)+1),
            params[:,1],
            marker='s',
            linewidth=2,
            label=f"Noise={noise}"
        )

    plt.axhline(
        4000,
        color='black',
        linestyle='--',
        linewidth=2,
        label='True k2'
    )

    plt.xlabel("Experiments Used")
    plt.ylabel("Estimated k2")
    plt.title("k2 Convergence")
    plt.legend()
    plt.grid(True)

    plt.show()
    
# ============================================================
# PART 11.3 — Relative Error Plot
# ============================================================ 
    
    
def plot_parameter_error(results):

    plt.figure(figsize=(9,6))

    true_k = np.array([4000,4000])

    for noise,data in results.items():

        params = np.array(data["param_history"])

        errors = []

        for p in params:

            err = np.linalg.norm(
                p-true_k
            )/np.linalg.norm(true_k)

            errors.append(err)

        plt.plot(
            errors,
            marker='o',
            linewidth=2,
            label=f"Noise={noise}"
        )

    plt.yscale("log")

    plt.xlabel("Experiments Used")
    plt.ylabel("Relative Parameter Error")
    plt.title("Parameter Error Convergence")
    plt.legend()
    plt.grid(True)

    plt.show()

# ============================================================
# PART 11.4 — information gain
# ============================================================     
    
def plot_information_gain(results):

    plt.figure(figsize=(8,6))

    for noise,data in results.items():

        info = data["y"]

        best_info = np.maximum.accumulate(info)

        plt.plot(
            best_info,
            marker='o',
            label=f"σ={noise}"
        )

    plt.xlabel("Number of Experiments")
    plt.ylabel("Best Information")

    plt.title("Bayesian OED Information Gain")

    plt.legend()
    plt.grid(True)

    plt.show()
    
# ============================================================
# PART 12 — RUN
# ============================================================

if __name__ == "__main__":

    results = run_noise_study()

    plot_k1_convergence_all(results)

    plot_k2_convergence_all(results)

    plot_parameter_error(results)

    plot_information_gain(results)
    
    
    
    