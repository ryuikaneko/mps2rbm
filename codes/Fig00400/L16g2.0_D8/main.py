import numpy as np
import copy
import argparse
from tenpy_toycodes.d_dmrg import example_DMRG_tf_ising_finite
from tenpy_toycodes.tfi_exact import finite_gs_energy

#==== begin: calculate exact energy
def ground_state_energy_tfim(N, h, J=1.0, bc='periodic'):
    """
    Calculate the ground state energy of the transverse-field Ising model
    with either periodic or open boundary conditions, using the Jordan-Wigner transformation.

    Parameters:
    - N: int, number of sites
    - h: float, transverse field strength
    - J: float, coupling constant (default 1)
    - bc: str, 'periodic' (default) or 'open' for boundary conditions

    Returns:
    - E0: float, total ground state energy
    - e0_per_site: float, ground state energy per site
    """
    lambda_ = h / J

    if bc == 'periodic':
        # k-values for periodic BC and even fermion parity
        k = (2 * np.arange(N) + 1) * np.pi / N
    elif bc == 'open':
        # Solve quantization condition sin((N+1)k) - lambda*sin(Nk) = 0 for k in (0, pi)
        def f(k):
#            return np.sin((N+1)*k) - lambda_ * np.sin(N*k)
            return np.sin((N+1)*k) - 1.0/lambda_ * np.sin(N*k)

        roots = []
        # bracket around m*pi/(N+1)
        for m in range(1, N+1):
            a = (m - 0.5) * np.pi / (N + 1)
            b = (m + 0.5) * np.pi / (N + 1)
            fa, fb = f(a), f(b)
            # if no sign change, widen bracket
            if fa * fb > 0:
                a = max(1e-6, (m * np.pi / (N + 1)) * 0.9)
                b = min(np.pi - 1e-6, (m * np.pi / (N + 1)) * 1.1)
                fa, fb = f(a), f(b)
            # bisection
            for _ in range(50):
                c = 0.5 * (a + b)
                fc = f(c)
                if fa * fc <= 0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
            roots.append(0.5 * (a + b))
        k = np.array(roots)
    else:
        raise ValueError("bc must be 'periodic' or 'open'")

    # dispersion: epsilon_k = 2J * sqrt(1 + lambda_^2 - 2*lambda_*cos(k))
    eps_k = 2 * J * np.sqrt(1 + lambda_**2 - 2 * lambda_ * np.cos(k))

    # ground state energy
    E0 = -0.5 * np.sum(eps_k)
    e0_per_site = E0 / N
    return E0, e0_per_site
#==== end: calculate exact energy

#==== begin: calculate fidelity of complex mpss
def create_random_mps(L, d, D):
    mps = []
    for i in range(L):
        Dl = 1 if i == 0 else D
        Dr = 1 if i == L - 1 else D
        tensor = np.random.randn(Dl, d, Dr) + 1j * np.random.randn(Dl, d, Dr)
        mps.append(tensor)
    return mps

def mps_inner_product_with_product_state(mps, product_state):
    env = np.tensordot(np.conj(mps[0]), product_state[0], axes=([1], [0]))  # (Dl, Dr)
    for i in range(1, len(mps)):
        env = np.tensordot(env, np.conj(mps[i]), axes=([1], [0]))           # (Dl, d, Dr)
        env = np.tensordot(env, product_state[i], axes=([1], [0]))          # (Dl, Dr)
    return env.squeeze()

def mps_inner_product(mps1, mps2):
    L = len(mps1)
    env = np.tensordot(np.conj(mps1[0]), mps2[0], axes=([1], [1]))  # [Dl1*, Dr1, Dl2, Dr2]
    env = np.transpose(env, (0, 2, 1, 3))  # [Dl1*, Dl2, Dr1, Dr2]
    for i in range(1, L):
        A1 = np.conj(mps1[i])  # [Dl1, d, Dr1]*
        A2 = mps2[i]           # [Dl2, d, Dr2]
        temp = np.tensordot(A1, A2, axes=([1], [1]))  # [Dl1*, Dr1, Dl2, Dr2]
        temp = np.transpose(temp, (0, 2, 1, 3))       # [Dl1*, Dl2, Dr1, Dr2]
        env = np.tensordot(env, temp, axes=([2, 3], [0, 1]))  # --> [Dr1', Dr2']
    return env.squeeze()

def mps_norm(mps):
    return np.sqrt(mps_inner_product(mps, mps))

def cp_inner_product(factors):
    """
    factors: list of factor matrices, each of shape (In, R)
    """
    R = factors[0].shape[1]
    G = np.ones((R, R), dtype=complex)
    for A in factors:
        Gn = A.conj().T @ A  # Adagger A --> shape (R, R)
        G *= Gn  # Hadamard product
    return np.real(np.sum(G))

def cp_norm(cp_factors):
    return np.sqrt(cp_inner_product(cp_factors))

def mps_cp_inner_product(mps, cp_factors):
    """
    mps: list of tensors [Dl, d, Dr]
    cp_factors: list of factor matrices (each shape: [d, R])
    """
    R = cp_factors[0].shape[1]
    result = 0.0 + 0.0j
    for r in range(R):
        product_state = [cp_factors[n][:, r] for n in range(len(cp_factors))]
        result += mps_inner_product_with_product_state(mps, product_state)
    return result

def fidelity(mps, cp_factors):
    # <A|B>
    inner_prod = mps_cp_inner_product(mps, cp_factors)
    # norm <A|A>, <B|B>
    norm_A = mps_norm(mps)
    norm_B = cp_norm(cp_factors)
    # Fidelity
#    fidelity_value = np.real( np.abs(inner_prod) / (norm_A * norm_B) )
    fidelity_value = np.real( np.abs(inner_prod) / (norm_A * norm_B) )**2
    return fidelity_value
#==== end: calculate fidelity of complex mpss

def get_spin(state,site):
    return (state>>site)&1

def flip_spin(state,site):
    return state^(1<<site)

def make_ham(L,g):
    Nstate = 2**L
    Ham = np.zeros((Nstate,Nstate),dtype=np.float64)
    for a in range(Nstate):
#        for i in range(L):
        for i in range(L-1):
            if get_spin(a,i) == get_spin(a,(i+1)%L):
                Ham[a,a] -= 1.0
            else:
                Ham[a,a] += 1.0
        for i in range(L):
            b = flip_spin(a,i)
            Ham[a,b] -= g
    return Ham

def reshape_MPS(psi,L,chi):
    Gs = []
    Gs.append(psi.Bs[0][0,:,:])
    for i in range(1,L-1):
        Gs.append(psi.Bs[i])
    Gs.append(psi.Bs[-1][:,:,0])
    return Gs

def khatri_rao(matrices, skip=None):
    """
    Compute the Khatri-Rao product of a list of factor matrices, optionally skipping one.
    matrices: list of arrays [I_k x R]
    skip: index to skip (for update of that factor)
    Returns: (\prod I_except_skip) x R array
    """
    if skip is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip]
    if len(matrices) == 1:
        return matrices[0]
    if matrices[0].ndim == 2:
        n_columns = matrices[0].shape[1]
    else:
        n_columns = 1
        matrices = [np.reshape(m, (-1, 1)) for m in matrices]
    for i, e in enumerate(matrices[1:]):
        if not i:
            res = matrices[0]
        s1, s2 = res.shape
        s3, s4 = e.shape
        a = np.reshape(res, (s1, 1, s2))
        b = np.reshape(e, (1, s3, s4))
        res = np.reshape(a * b, (-1, n_columns))
    return res

def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def mttkrp_mps(mpss, factors, mode):
    """
    Matricized tensor times Khatri-Rao product:
    Computes the Matrix-Vector product for a given mode of a tensor represented in MPS format.
    [input]
    mpss : list of np.ndarray
        The matrix product state (MPS) representation of the tensor.
        Typically, MPS takes the form of mpss[L][D,d,D],
        where L is the number of sites, D is the bond dimension,
        and d is the physical dimension.
        Note that, at the left edge, mpss[0][d,D],
        whereas at the right edge, mpss[L-1][D,d].
    factors : list of np.ndarray
        The factor matrices to be multiplied.
        Typically, factors takes the form of factors[L][d,R],
        where L is the number of sites, d is the physical dimension,
        and R is the rank of the factor matrix.
    mode : int
        The mode along which to perform the multiplication.
    [output]
    result: np.ndarray
        The resulting matrix after performing the MTTKRP operation.
        Typically, result takes the form of result[d,R],
        where d is the physical dimension and R is the rank of the factor matrix.
    """
    # delta_{ijk}
    rank = factors[0].shape[1]
    L = len(factors)
    delta = (np.arange(rank)[:, None, None] == np.arange(rank)[None, :, None]) & (np.arange(rank)[None, None, :] == np.arange(rank)[:, None, None])
    delta = delta.astype(np.float64)
    if mode==0:
        left = mpss[0] # [d,D]
        # right to left contraction
        right = np.tensordot(mpss[-1], factors[-1], axes=1) # [D,d1]*[d1,R]=[D,R]
        for i in range(L-2, mode, -1): # exclude mode=0
            right = np.tensordot(right, mpss[i], axes=([0],[-1])) # [D1,R]*[D,d,D1] = [R,D,d]
            right = np.tensordot(right, factors[i], axes=1) # [R,D,d1]*[d1,R]=[R,D,R]
            right = np.tensordot(right, delta, axes=([0,-1],[0,-1])) # [R1,D,R2]*[R1,R,R2]=[D,R]
        result = np.tensordot(left, right, axes=1) # [d,D1]*[D1,R]=[d,R]
    elif mode==L-1:
        # left to right contraction
        left = np.tensordot(mpss[0], factors[0], axes=([0],[0])) # [d1,D]*[d1,R]=[D,R]
        for i in range(1, mode): # exclude mode=L-1
            left = np.tensordot(left, mpss[i], axes=([0],[0])) # [D1,R]*[D1,d,D] = [R,d,D]
            left = np.tensordot(left, factors[i], axes=([1],[0])) # [R,d1,D]*[d1,R]=[R,D,R]
            left = np.tensordot(left, delta, axes=([0,-1],[0,-1])) # [R1,D,R2]*[R1,R,R2]=[D,R]
        right = mpss[-1] # [D,d]
        result = np.tensordot(left, right, axes=([0],[0])) # [D1,R]*[D1,d]=[R,d]
        result = result.T # [d,R]
    else:
        # left to right contraction
        left = np.tensordot(mpss[0], factors[0], axes=([0],[0])) # [d1,D]*[d1,R]=[D,R]
        for i in range(1, mode): # exclude mode
            left = np.tensordot(left, mpss[i], axes=([0],[0])) # [D1,R]*[D1,d,D] = [R,d,D]
            left = np.tensordot(left, factors[i], axes=([1],[0])) # [R,d1,D]*[d1,R]=[R,D,R]
            left = np.tensordot(left, delta, axes=([0,-1],[0,-1])) # [R1,D,R2]*[R1,R,R2]=[D,R]
        # right to left contraction
        right = np.tensordot(mpss[-1], factors[-1], axes=1) # [D,d1]*[d1,R]=[D,R]
        for i in range(L-2, mode, -1): # exclude mode
            right = np.tensordot(right, mpss[i], axes=([0],[-1])) # [D1,R]*[D,d,D1] = [R,D,d]
            right = np.tensordot(right, factors[i], axes=1) # [R,D,d1]*[d1,R]=[R,D,R]
            right = np.tensordot(right, delta, axes=([0,-1],[0,-1])) # [R1,D,R2]*[R1,R,R2]=[D,R]
        result = np.tensordot(left, mpss[mode], axes=([0],[0])) # [D1,R]*[D1,d,D]=[R,d,D]
        result = np.tensordot(result, right, axes=1) # [R,d,D1]*[D1,d]=[R,d,R]
        result = np.tensordot(result, delta, axes=([0,-1],[0,-1])) # [R1,d,R2]*[R1,R,R2]=[d,R]
    return result

def reconstruct_cp(factors):
    """
    Reconstruct a tensor from its CP decomposition factors.
    factors: list of arrays [I_k x R]
    Returns: full tensor of shape I_0 x I_1 x ... x I_{n-1}
    """
    rank = factors[0].shape[1]
    shapes = [mat.shape[0] for mat in factors]
    # Initialize approximation
    X_hat = np.zeros(shapes)
    # Sum over ranks
    for r in range(rank):
        outer_prod = factors[0][:, r] 
        for mat in factors[1:]:
            outer_prod = np.multiply.outer(outer_prod, mat[:, r])
        X_hat += outer_prod
    return X_hat

def normalize_factors(factors):
    """
    Normalize each factor matrix's columns to unit norm. Returns norms for each factor.
    factors: list of [I_k x R]
    Returns: normalized factors, list of lambda vectors (length R) per factor
    """
    lambdas = []
    for idx, A in enumerate(factors):
        norms = np.linalg.norm(A, axis=0)
        # avoid division by zero
        norms[norms == 0] = 1
        factors[idx] = A / norms
        lambdas.append(norms)
    return factors, lambdas

def cp_als(X, rank, n_iter=1000, tol=1e-6, verbose=False):
    """
    Compute CP decomposition of an n-way tensor X via Alternating Least Squares.
    Parameters:
    X : ndarray, shape (I_0, I_1, ..., I_{n-1})
    rank : int, target CP rank (number of components)
    n_iter : int, maximum number of ALS iterations
    tol : float, convergence tolerance on relative error
    verbose : bool, whether to print progress
    Returns:
    factors : list of n factor matrices [I_k x rank]
    """
    # Number of modes
    n_modes = X.ndim
    # Dimensions
    dims = X.shape
    # Initialize factor matrices with random values
    factors = [np.random.randn(dim, rank) for dim in dims]
    # Precompute unfoldings
    X_unfolds = [unfold(X, mode) for mode in range(n_modes)]
    diageps = 1e-10
    normX = np.linalg.norm(X)
    errors = []
    for it in range(n_iter):
        # Normalize factors
        factors, lambdas = normalize_factors(factors)
        # ALS updates for each mode
        for mode in range(n_modes):   
            # Khatri-Rao of all factors except current mode
            Z = khatri_rao(factors, skip=mode)
            # Compute Gramian
#            Y = Z.T @ Z
            Y = np.ones((rank, rank))
            for i, factor in enumerate(factors):
                if i == mode:
                    continue 
                Y *= factor.T @ factor  
            # Least squares update
            V = X_unfolds[mode] @ Z
            reg = diageps * np.max(np.abs(Y)) # Regularize Y 
            factors[mode] = V @ np.linalg.inv(Y + reg * np.eye(Y.shape[0]))
        # Compute reconstruction error
        if verbose or it == n_iter - 1:
            X_hat = reconstruct_cp(factors)
            error = np.linalg.norm(X - X_hat) / normX
            errors.append(error)
            if verbose:
#                print(f"Iteration {it+1:d}, error = {error:.2e}")
                if (it+1)%20==0:
                    print(f"Iteration {it+1:d}, error = {error:.2e}")
            if error < tol:
                if verbose:
                    print(f"Converged at iteration {it+1}, error: {error:.2e}")
                break
    return factors, errors

def cp_als_mps(mpss, rank, n_iter=1000, tol=1e-6, verbose=False, itskip=20):
    """
    Compute CP decomposition of an n-way tensor X given by matrix product states mpss via Alternating Least Squares.
    Parameters:
    mpss : list of np.ndarray. The matrix product state (MPS) representation of the tensor.
    rank : int, target CP rank (number of components)
    n_iter : int, maximum number of ALS iterations
    tol : float, convergence tolerance on relative error
    verbose : bool, whether to print progress
    Returns:
    factors : list of n factor matrices [I_k x rank]
    """
    # Number of modes
    n_modes = len(mpss)
    # Dimensions
#    dims = [mpss[mode].shape[1] for mode in range(n_modes)]
    dims = [mpss[mode].shape[0] if mode==0 else mpss[mode].shape[1] for mode in range(n_modes)]
    # Initialize factor matrices with random values
    factors = [np.random.randn(dim, rank) for dim in dims]
    # Precompute unfoldings
    diageps = 1e-10
    errors = []
    for it in range(n_iter):
        # Copy old factors
        factors_old = copy.deepcopy(factors)
        # Normalize factors
        factors, lambdas = normalize_factors(factors)
        # ALS updates for each mode
        for mode in range(n_modes):   
            # Compute Gramian
            Y = np.ones((rank, rank))
            for i, factor in enumerate(factors):
                if i == mode:
                    continue 
                Y *= factor.T @ factor
            # Least squares update
            V = mttkrp_mps(mpss, factors, mode) # Khatri-Rao of all factors except current mode
            reg = diageps * np.max(np.abs(Y)) # Regularize Y 
            factors[mode] = V @ np.linalg.inv(Y + reg * np.eye(Y.shape[0]))
        # Compute reconstruction error
        if verbose or it == n_iter - 1:
            ## Evaluate error without using contracted tensor X
#            error = 1.0
#            print("Fs.shape",*(F.shape for F in factors))
#            print("Fs.shape",*(F.shape for F in factors_old))
#            error = np.sum([np.linalg.norm(factors[mode]-factors_old[mode])/np.linalg.norm(factors_old[mode]) for mode in range(n_modes)])/n_modes/dims[0]/rank
#            error = np.sum([np.linalg.norm(factors[mode]-factors_old[mode])/np.linalg.norm(factors_old[mode]) for mode in range(n_modes)])/n_modes
#
#            print([mps.shape for mps in mpss])
#            print([mps[np.newaxis,:,:].shape if i==0 else mps[:,:,np.newaxis].shape if i==len(mpss)-1 else mps.shape for i,mps in enumerate(mpss)])
            mpssnew = [mps[np.newaxis,:,:] if i==0 else mps[:,:,np.newaxis] if i==len(mpss)-1 else mps for i,mps in enumerate(mpss)]
#            print([mps.shape for mps in mpssnew])
#            print([f.shape for f in factors])
            error = 1.0 - fidelity(mpssnew, factors)
            errors.append(error)
            if verbose:
#                print(f"Iteration {it+1:d}, error = {error:.2e}")
#                if (it+1)%20==0:
                if (it+1)%itskip==0:
                    print(f"Iteration {it+1:d}, error = {error:.2e}")
            if error < tol:
                if verbose:
                    print(f"Converged at iteration {it+1}, error: {error:.2e}")
                break
    return factors, errors


def main():
    parser = argparse.ArgumentParser(description="dmrg and cp decomposition")
    parser.add_argument("--L", type=int, default=16, help="number of sites")
    parser.add_argument("--g", type=float, default=1.0, help="interaction")
    parser.add_argument("--rank", type=int, default=8, help="cp rank")
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
    np.random.seed(seed)
    print("L",L)
    print("g",g)
    print("MPS bond dim D",D)
    print("CPD rank",rank)
    print()

## DMRG
#    E, psi, model = example_DMRG_tf_ising_finite(L, g)
    E, psi, model = example_DMRG_tf_ising_finite(L, g, chi_max=D)
    print("DMRG Energy", E)
    print()

## reshape MPS
#    Gs = []
#    Gs.append(np.random.randn(d,D))
#    for i in range(1,L-1):
#        Gs.append(np.random.randn(D,d,D))
#    Gs.append(np.random.randn(D,d))
    Gs = reshape_MPS(psi,L,D)
    print("Gs.shape",*(G.shape for G in Gs))
    print()

## Hamiltonian
    Ham = make_ham(L,g)
#    print("Hamiltonian",Ham)
#    print()
##    Eexact_, _ = ground_state_energy_tfim(L,g,bc="open") ## bug in code, works only for g>=1
    Eexact = finite_gs_energy(L, 1., g)
##    print("Analytical Energy", Eexact_)
    print("Analytical Energy", Eexact)
    print()

## contract MPS, CP decomposition, and reconstruction
#    print("CPD to contracted MPS")
#    X = Gs[0]
#    for i in range(1,L):
#        X = np.tensordot(X,Gs[i],axes=(-1, 0))
#    Fs = cp_als(X, rank, n_iter=200, tol=1e-4, verbose=True)
#    print("Fs.shape",*(F.shape for F in Fs))
#    X1 = reconstruct_cp(Fs)
#    error = np.linalg.norm(X-X1)/np.linalg.norm(X)
#    print("CPD error",error)
#    V = X1 / np.linalg.norm(X1.flatten())
#    VHV = V.flatten()@Ham@V.flatten()
#    print("<H>",VHV)
#    print("<H> error from DMRG",np.abs((VHV-E)/E))
#    print()

## run cp_als with MPS
    string = "_L" + "{}".format(L) \
        + "_g" + "{:.10f}".format(g) \
        + "_D" + "{}".format(D) \
        + "_rank" + "{}".format(rank) \
        + "_seed" + "{}".format(seed)
    print("CPD to MPS")
#    Fs, errors = cp_als_mps(Gs, rank, n_iter=200, tol=1e-4, verbose=True, itskip=1)
    Fs, errors = cp_als_mps(Gs, rank, n_iter=1000, tol=1e-6, verbose=True, itskip=1)
    print("Fs.shape",*(F.shape for F in Fs))
    np.savez_compressed("dat_cp_F0s"+string,Fs)
    np.savetxt("dat_cp_errors"+string,errors)
    X2 = reconstruct_cp(Fs)
#    error = np.linalg.norm(X-X2)/np.linalg.norm(X)
#    print("CPD error",error)
    V = X2 / np.linalg.norm(X2.flatten())
    VHV = V.flatten()@Ham@V.flatten()
    np.savetxt("dat_cp_enes"+string,
        np.array([
            L,g,D,rank,seed,Eexact,E,VHV,
            np.abs((E-Eexact)/Eexact),np.abs((VHV-Eexact)/Eexact),errors[-1]
        ]).reshape(1,-1),
        header="L g D rank seed exact dmrg cpd dmrg_error cpd_error 1-fidelity[-1]")
    print("<H>",VHV)
    print("<H> error from DMRG",np.abs((VHV-E)/E))
    print()

## prepare multinomial RBM parameters
    print("initial RBM parameters")
    logfactors = np.log(np.array(Fs) + 0.0j)
    b0s = 0.5*(logfactors[:,0,:]+logfactors[:,1,:]).T
    W0s = 0.5*(logfactors[:,0,:]-logfactors[:,1,:]).T
    np.savetxt("dat_cp_b0s"+string,b0s)
    np.savetxt("dat_cp_W0s"+string,W0s)
    print("b0s",b0s.shape)
#    print(b0s)
    print("W0s",W0s.shape)
#    print(W0s)
    print()  

## load test
#    dat_Fs = np.load("dat_cp_F0s.npz")["arr_0"]
#    print("original Fs",*(F.shape for F in Fs))
#    print("loaded Fs  ",*(F.shape for F in dat_Fs))

if __name__ == "__main__":
    main()
