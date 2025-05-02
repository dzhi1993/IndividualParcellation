try:
    from tqdm.cli import tqdm
except:
    tqdm = lambda x:x
import numpy as np
from scipy import linalg
import pickle, time
import torch as pt


pt.cuda.is_available = lambda : True
if pt.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)

def move_to_cuda(data):
    if isinstance(data, np.ndarray):
        # Convert NumPy array to a PyTorch tensor and move it to GPU
        return pt.tensor(data, dtype=pt.get_default_dtype())
    elif isinstance(data, pt.Tensor):
        # If already a PyTorch tensor, move it to GPU
        return data.to(dtype=pt.get_default_dtype(), device=DEVICE)
    elif isinstance(data, (int, float)):
        # Scalars are returned as-is (no GPU move needed for single values)
        return pt.tensor(data, dtype=pt.get_default_dtype())
    elif isinstance(data, list):
        # Recursively move each element in the list
        return [move_to_cuda(item) for item in data]
    elif isinstance(data, dict):
        # Recursively move each item in the dictionary
        return {k: move_to_cuda(v) for k, v in data.items()}
    else:
        # If it's not a tensor or array, return as-is (e.g., int, float)
        return data
    
def xpfun(data_dt, return_data=False, n_iter = 20):
    new_dt = {}
    itr_lst = [elem for _ in range(int(n_iter/2)) for elem in data_dt.keys()]
    for itr in tqdm(itr_lst):
        # This loop iterates over
        if not itr in new_dt: new_dt[itr] = dict()
        for i_reg in data_dt[itr].keys():
                
            # Hard unpack the dict here:
            dt = data_dt[itr][i_reg]
            D, psi, idx_reg = dt['D'], dt['psi'], dt['idx_reg']
            beta_tilde, sigma, n_eff = dt['beta_tilde'], dt['sigma'], dt['n_eff']
            s2, Di, Dih, s = dt['s2'], dt['Di'], dt['Dih'], dt['s']
            beta = dt['beta']

            if return_data:
                name_lst = ['D','psi','idx_reg','beta_tilde','sigma','n_eff','s2','Di','Dih','s','beta']
                new_dt[itr][i_reg] = {name:var for name, var in data_dt[itr][i_reg].items() if name in name_lst}

            ## One way to do it (which does involve a funny addition inside of the cholesky solve...)

            # A part the also would need to happen on the gpu:
            dinvt = D + np.diag(1.0/psi[idx_reg].T[0])

            # The solve part:
            dinvt_chol = linalg.cholesky(dinvt)

            beta_tmp = (linalg.solve_triangular(dinvt_chol, beta_tilde, trans='T') +
                        np.sqrt(sigma/n_eff)*np.random.randn(len(D), 1))
            beta[idx_reg] = linalg.solve_triangular(dinvt_chol, beta_tmp, trans='N')

            ## Another way to do it involving a cleaner solve:

            # Stuff that can happen earlier. just send some small vector to the gpu:
            t = psi[idx_reg].T[0]*float(s2/n_eff)
            u     = np.random.randn(len(Di),1)*np.sqrt(t[:,np.newaxis])
            seedk = np.random.randn(len(Di),1)
            k     = Dih@seedk
            alpha = Di@beta_tilde-u-(s/np.sqrt(n_eff))*k

            # Stuff that would need to happen on the gpu:
            # <- mod the T here
            T     = np.diag(t) # This step could prob. be done faster: np.fill_diagonal(T,t)
            W     = (T + (s2/n_eff)*Di) # This step too by moving (s2/n_eff) to another spot
            R     = linalg.solve(W, alpha) # <- The solve
            beta[idx_reg] = u + T@R
            
    if return_data:
        return new_dt


# Function rewritten in PyTorch
def xpfun_torch(data_dt, return_data=False, n_iter=20):
    new_dt = {}
    itr_lst = [elem for _ in range(int(n_iter/2)) for elem in data_dt.keys()]
    
    for itr in tqdm(itr_lst):
        if itr not in new_dt: 
            new_dt[itr] = dict()
        
        for i_reg in data_dt[itr].keys():
            dt = data_dt[itr][i_reg]
            D, psi, idx_reg = dt['D'], dt['psi'], dt['idx_reg']
            beta_tilde, sigma, n_eff = dt['beta_tilde'], dt['sigma'], dt['n_eff']
            s2, Di, Dih, s = dt['s2'], dt['Di'], dt['Dih'], dt['s']
            beta = dt['beta']

            if return_data:
                name_lst = ['D', 'psi', 'idx_reg', 'beta_tilde', 'sigma', 'n_eff', 's2', 'Di', 'Dih', 's', 'beta']
                new_dt[itr][i_reg] = {name: var for name, var in data_dt[itr][i_reg].items() if name in name_lst}
            
            # Perform operations on the GPU using PyTorch
            dinvt = D + pt.diag(1.0 / psi[idx_reg].T[0])
            dinvt_chol = pt.linalg.cholesky(dinvt)

            beta_tmp = (pt.linalg.solve_triangular(dinvt_chol, beta_tilde, upper=False, left=True) +
                        pt.sqrt(sigma / n_eff) * pt.randn(len(D), 1))
            beta[idx_reg] = pt.linalg.solve_triangular(dinvt_chol, beta_tmp, upper=False, left=True)

            # Second approach
            t = psi[idx_reg].T[0] * (s2 / n_eff).item()
            u = pt.randn(len(Di), 1) * pt.sqrt(t[:, None])
            seedk = pt.randn(len(Di), 1)
            k = Dih @ seedk
            alpha = Di @ beta_tilde - u - (s / pt.sqrt(n_eff)) * k

            T = pt.diag(t)
            W = T + (s2 / n_eff) * Di
            R = pt.linalg.solve(W, alpha)
            beta[idx_reg] = u + T @ R

    if return_data:
        return new_dt
    

# Function rewritten in PyTorch
def xpfun_torch_parallel(data_dt, return_data=False, n_iter=20):
    new_dt = {}
    itr_lst = [elem for _ in range(int(n_iter/2)) for elem in data_dt.keys()]
    
    for itr in itr_lst:
        # Gather all necessary data for each iteration into batches for parallel processing
        all_D, all_psi, all_beta_tilde, all_sigma, all_n_eff, all_s2, all_Di, all_Dih, all_s = [], [], [], [], [], [], [], [], []


        for i_reg, dt in data_dt[itr].items():
            all_D.append(dt['D'].to('cpu'))
            all_psi.append(dt['psi'][dt['idx_reg']])
            all_beta_tilde.append(dt['beta_tilde'])
            all_sigma.append(dt['sigma'])
            all_n_eff.append(dt['n_eff'])
            all_s2.append(dt['s2'])
            all_Di.append(dt['Di'])
            all_Dih.append(dt['Dih'])
            all_s.append(dt['s'])

        # Stack data tensors to create a batch dimension
        D = pt.block_diag(*all_D).to_sparse().to('cuda') # shape: [batch_size, *D_shape]
        psi = pt.stack(all_psi)  # shape: [batch_size, *psi_shape]
        beta_tilde = pt.stack(all_beta_tilde)  # shape: [batch_size, *beta_tilde_shape]
        sigma = pt.tensor(all_sigma).to(D.device)  # shape: [batch_size]
        n_eff = pt.tensor(all_n_eff).to(D.device)  # shape: [batch_size]
        s2 = pt.tensor(all_s2).to(D.device)  # shape: [batch_size]
        Di = pt.stack(all_Di)
        Dih = pt.stack(all_Dih)
        s = pt.tensor(all_s).to(D.device)
                
        # Perform operations on the GPU using PyTorch
        dinvt = D + pt.diag(1.0 / psi[idx_reg].T[0])
        dinvt_chol = pt.linalg.cholesky(dinvt)

        beta_tmp = (pt.linalg.solve_triangular(dinvt_chol, beta_tilde, upper=False, left=True) +
                    pt.sqrt(sigma / n_eff) * pt.randn(len(D), 1))
        beta[idx_reg] = pt.linalg.solve_triangular(dinvt_chol, beta_tmp, upper=False, left=True)

        # Second approach
        t = psi[idx_reg].T[0] * (s2 / n_eff).item()
        u = pt.randn(len(Di), 1) * pt.sqrt(t[:, None])
        seedk = pt.randn(len(Di), 1)
        k = Dih @ seedk
        alpha = Di @ beta_tilde - u - (s / pt.sqrt(n_eff)) * k

        T = pt.diag(t)
        W = T + (s2 / n_eff) * Di
        R = pt.linalg.solve(W, alpha)
        beta[idx_reg] = u + T @ R

    if return_data:
        return new_dt
    

with open('/data/tge/menno/repos/sparse-prscs/nbs/gpu-speedup_da+menno.pkl', 'rb') as f:
    xp_dt = pickle.load(f)

# 1. normal numpy operation
# tic = time.perf_counter()
# xpfun(xp_dt)
# toc = time.perf_counter()
# print(f'Scipy version: {toc - tic:0.4f} seconds!')

# 2. pytorch gpu
# xp_dt = move_to_cuda(xp_dt)
# tic = time.perf_counter()
# xpfun_torch(xp_dt)
# toc = time.perf_counter()
# print(f'PyTorch version: {toc - tic:0.4f} seconds!')



# Define separate CUDA streams
s1 = pt.cuda.Stream()
s2 = pt.cuda.Stream()
# Initialise cuda tensors here. E.g.:
A = pt.randn(1000, 1000, device = 'cuda')
B = pt.randn(1000, 1000, device = 'cuda')
# Wait for the above tensors to initialise.
pt.cuda.synchronize()

tic = time.perf_counter()
with pt.cuda.stream(s1):
    for i in range(1000):
        C = pt.mm(A, A)

with pt.cuda.stream(s2):
    for i in range(1000):
        D = pt.mm(B, B)
# Wait for C and D to be computed.
pt.cuda.synchronize()
# Do stuff with C and D.
toc = time.perf_counter()
print(f'parallel: {toc - tic:0.4f} seconds!')

tic = time.perf_counter()
A = pt.randn(1000, 1000, device='cuda')
B = pt.randn(1000, 1000, device='cuda')
for i in range(1000):
    C = pt.mm(A, A)

for i in range(1000):
    C = pt.mm(B, B)

toc = time.perf_counter()
print(f'sequential: {toc - tic:0.4f} seconds!')

# 3. pytorch gpu vectorize
xp_dt = move_to_cuda(xp_dt)
tic = time.perf_counter()
xpfun_torch_parallel(xp_dt)
toc = time.perf_counter()
print(f'PyTorch version: {toc - tic:0.4f} seconds!')

# OR! :
# p1 = %lprun -r -f fun fun(xp_dt) # for granular line profiling