import numpy as np
import copy
import argparse
import numba

## RBM
@numba.jit(nopython=True)
def logsumexp(u):
    m = np.max(u.real)
    s = 0.0 + 0.0j
    for x in u:
        s += np.exp(x - m)
    return m + np.log(s)

@numba.jit(nopython=True)
def log_psi(v, b, W):
    u = np.sum(b + W * v[np.newaxis, :], axis=1)
    return logsumexp(u)

@numba.jit(nopython=True)
def local_energy(v, b, W, J, h):
#    E_diag = -J * np.sum(v * np.roll(v, -1)) ## periodic
    E_diag = -J * np.sum(v[:-1] * v[1:]) ## open
    E_off = 0.0
    for i in range(v.shape[0]):
        v_flip = v.copy()
        v_flip[i] *= -1
        E_off += -h * np.exp(log_psi(v_flip, b, W) - log_psi(v, b, W))
    return E_diag + E_off

@numba.jit(nopython=True)
def metropolis_step(v, b, W):
    i = np.random.randint(v.shape[0])
    v_new = v.copy()
    v_new[i] *= -1
    ratio = np.abs(np.exp(2 * (log_psi(v_new, b, W) - log_psi(v, b, W))))
    if np.random.rand() < min(1.0, ratio):
        return v_new
    return v

@numba.jit(nopython=True)
def run_vmc(n,J,h,q,n_samples,n_iter,n_ave,lr,diag_shift,b,W,seed):
    # parameter initialization
    np.random.seed(seed=seed)
    E_means = np.zeros(n_iter,dtype=np.complex128)
    b_ave = np.zeros(b.shape,dtype=np.complex128)
    W_ave = np.zeros(W.shape,dtype=np.complex128)
    for it in range(n_iter):
        # sampling
        v = np.where(np.random.rand(n) < 0.5, 1, -1)
        vs = np.zeros((n_samples, n))
        for j in range(n_samples): # samples for thermalization
            for _ in range(n):
                v = metropolis_step(v, b, W)
        for j in range(n_samples): # store samples after thermalization
            for _ in range(n):
                v = metropolis_step(v, b, W)
                vs[j] = v
        # calculate gradient and local energy
        O = np.zeros((n_samples, 2 * q * n),dtype=np.complex128)
        E_loc = np.zeros(n_samples,dtype=np.complex128)
        for idx in range(n_samples):
            v = vs[idx]
            E_loc[idx] = local_energy(v, b, W, J, h)
            if lr > 1e-12: # for updating wave function
                # softmax p_r
                scores = np.sum(b + W * v[np.newaxis, :], axis=1)
                stable_scores = scores - np.max(scores.real)
                exps = np.exp(stable_scores)
                Z = exps.sum()
                p_r = exps / Z
                # d log psi / d b, d log psi / d W
                db_flat = np.repeat(p_r, n)
                dW_flat = (p_r[:, None] * v[None, :]).flatten()
                O[idx, :q*n] = db_flat
                O[idx, q*n:] = dW_flat
        E_mean = E_loc.mean()
        E_means[it] = E_mean
        if lr > 1e-12: # for updating wave function
            O_mean = np.sum(O,axis=0)/O.shape[0]
            # SR matrix and vector update
            S = (O.T @ O) / n_samples - np.outer(O_mean, O_mean)
            F = (O.T @ E_loc) / n_samples - E_mean * O_mean
            S += diag_shift * np.eye(2 * q * n)
            dp = np.linalg.solve(S, -F)
            maxdp = np.max(np.abs(dp))
            if maxdp > 1.0: # rescale updated vector if it is too large
                dp /= maxdp   
            db = dp[:q*n].reshape(q, n)
            dW = dp[q*n:].reshape(q, n)
            b  += lr * db
            W  += lr * dW
            # average b, W for last num_ave steps
            if it >= n_iter - n_ave:
                b_ave += b
                W_ave += W
        print(it,E_mean.real,E_mean.imag)
    # average b, W for last num_ave steps
    b_ave /= n_ave
    W_ave /= n_ave
    return b_ave, W_ave, E_means
## end of RBM

def main():
    parser = argparse.ArgumentParser(description="dmrg and cp decomposition")
    parser.add_argument("--L", type=int, default=16, help="number of sites")
    parser.add_argument("--g", type=float, default=1.0, help="interaction")
#    parser.add_argument("--rank", type=int, default=8, help="cp rank")
    parser.add_argument("--rank", type=int, default=40, help="cp rank")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    args = parser.parse_args()

## set parameters
    L = args.L
    g = args.g
#    D = L//4
    D = L//2
    d = 2
#    rank = 4*D**2
    rank = args.rank
    seed = args.seed
#    np.random.seed(seed)
    print("L",L)
    print("g",g)
    print("MPS bond dim D",D)
    print("CPD rank",rank)
    print()

## prepare multinomial RBM parameters
    print("initial RBM parameters")
    string = "_L" + "{}".format(L) \
        + "_g" + "{:.10f}".format(g) \
        + "_D" + "{}".format(D) \
        + "_rank" + "{}".format(rank) \
        + "_seed" + "{}".format(seed)
    dir = "./input/"
    enes = np.loadtxt(dir+"dat_cp_enes"+string)
    e_exact = enes[5]
    e_dmrg = enes[6]
    e_cp = enes[7]
    b0s = np.loadtxt(dir+"dat_cp_b0s"+string,dtype=np.complex128)
    W0s = np.loadtxt(dir+"dat_cp_W0s"+string,dtype=np.complex128)
    print("e_exact",e_exact)
    print("e_dmrg",e_dmrg)
    print("e_cp",e_cp)
    print("b0s",b0s.shape)
#    print(b0s)
    print("W0s",W0s.shape)
#    print(W0s)
    print()

## run RBM VMC
    n = L
    J = 1.0
    h = g
    q = rank
    diag_shift = 1e-4
#
    print("run RBM VMC opt")
    n_samples = 10000
    n_iter = 2000
    n_ave = 100
    lr = 0.01
    bs, Ws, E_means = run_vmc(n,J,h,q,n_samples,n_iter,n_ave,lr,diag_shift,b0s,W0s,seed)
    np.savetxt("dat_vmc_bs"+string,bs)
    np.savetxt("dat_vmc_Ws"+string,Ws)
    np.savetxt("dat_vmc_enes_opt"+string,E_means)
    print()
#
    print("run RBM VMC aft")
    n_samples = 100000
    n_iter = 32
    n_ave = 0
    lr = 0.00
    _, _, E_means = run_vmc(n,J,h,q,n_samples,n_iter,n_ave,lr,diag_shift,bs,Ws,seed+1)
    np.savetxt("dat_vmc_enes_aft"+string,E_means)
    print()
#
    Eave = np.mean(E_means.real)
    Eerr = np.std(E_means.real)/np.sqrt(len(E_means)-1)
    np.savetxt("dat_vmc_enes_ave_err"+string,
        np.array([
            L,g,D,rank,seed,e_exact,e_dmrg,e_cp,
            Eave,Eerr,
            np.abs((Eave-e_exact)/e_exact),np.abs((Eerr)/e_exact)
        ]).reshape(1,-1),
        header="L g D rank seed exact dmrg cpd vmc vmc_err (vmc-exact)/exact vmc_err/exact")

if __name__ == "__main__":
    main()
