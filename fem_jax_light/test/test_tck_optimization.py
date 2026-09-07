#thickness optimization of a spar of a wing
import os
# os.environ["JAX_LOG_COMPILES"] = "1"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np 
from fem_jax_light.FEM_jax import FEM_study
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jax.ops as jops
from scipy.optimize import minimize, OptimizeResult
import matplotlib.pyplot as plt

import time as t 


#test_case = "wing_spar"
#test_case  = "wing_spar_ribs"
test_case  = "wing_full"
mesh_file = "meshes/"+test_case+".msh"
# Material and element properties
E = 6825.0*1e7
nu = 0.3
h = 0.04
element_type = {'tri': 'DKT_jax'}
if test_case == "wing_spar":
    n_var = 30
elif test_case == "wing_spar_ribs":
    n_var = 60
elif test_case == "wing_full":
    n_var = 90    
element_property = [[h]]*n_var
material = [[E,nu]]*n_var 
# Create FEM study
fem = FEM_study(mesh_file,element_type,element_property,material)
#RHS 
# we apply an elliptic load distribution along the span of the wing
nodes = fem.nodes
Lift = 4000.0 
y_sections = nodes[:,2]


f_y = Lift/(np.pi*y_sections.max())*np.sqrt(1-(y_sections/y_sections.max())**2) 

F = np.zeros((len(nodes)*6,))
F[2::6] = f_y
F = jnp.array(F)
rhs = fem.set_rhs(F)
# clamped boundary conditions at the root of the wing y == 0

ind = nodes[:,2]==0.0
# Boundary conditions
nodes_y0 = nodes[ind,0]
nodes_sets = [nodes_y0]
l_dof_clamped = [[0,1,2,3,4,5]]
l_dof = [l_dof_clamped]
fem.create_constrained_DOFs(nodes_sets, l_dof)

#compute the surfaces
surfaces = jnp.array([fem.element_dict['element_sets'][i+1]['surfaces'].sum() for i in range(n_var)])


#objective function: mass
def obj_fun(X):
    # X is the vector that parameterized the shape of the beam thickness using element properties
    mass = (X*surfaces).sum()
    return mass 

def h_sparse(X):
    # X is the vector that parameterized the shape of the beam thickness using element properties
    element_property = [[h] for h in X]
    K = fem.assembling_K_parametric_sparse(nodes[:,1:], element_property, material)
    rhs = fem.set_rhs(F)
    K, rhs = fem.boundary_conditions_sparse(K, rhs)
    Us = fem.solve_sparse(K,rhs)
    return (2.0-Us[2::6].max())


def h(X):
    # X is the vector that parameterized the shape of the beam thickness using element properties
    element_property = [[h] for h in X]
    K = fem.assembling_K_parametric(nodes[:,1:], element_property, material)
    rhs = fem.set_rhs(F)
    K, rhs = fem.boundary_conditions(K, rhs)
    Us = fem.solve(K,rhs)
    return (2.0-Us[2::6].max())


def h_VM(X):
    # X is the vector that parameterized the shape of the beam thickness using element properties
    element_property = X[:, None]
    K = fem.assembling_K_parametric(nodes[:,1:], element_property, material)
    rhs = fem.set_rhs(F)
    K, rhs = fem.boundary_conditions(K, rhs)
    Us = fem.solve(K,rhs)
    Us_r = Us.reshape((-1,6))
    strain_and_stress = fem.compute_strain_and_stress(nodes[:,0],Us_r,nodes[:,1:],element_property,material)
    stress = strain_and_stress[1]
    vm = fem.compute_vonMises(stress[:,:,-3:].T)/200e6
    vm_mean = vm.mean(axis=0) #mean over the gauss points

    # 2. Trouver le maximum local de CHAQUE groupe (en 1 seule passe XLA !)
    # max_per_group aura la taille (num_groups,)
    max_per_group = jops.segment_max(vm_mean, fem.element_type_ids, num_segments=90)
    
    # 3. "Diffuser" ce max sur chaque élément pour la stabilisation numérique
    # (Chaque élément récupère le max de son propre groupe)
    max_broadcast = max_per_group[fem.element_type_ids]
    
    # 4. Calcul du terme exponentiel stabilisé
    exp_term = jnp.exp(20 * (vm_mean - max_broadcast))
    
    # 5. Somme des exponentielles PAR GROUPE
    sum_exp_per_group = jops.segment_sum(exp_term, fem.element_type_ids, num_segments=90)
    
    # 6. Recombinaison de la formule LogSumExp par groupe
    ks_per_group = max_per_group + (1.0 / 20.0) * jnp.log(sum_exp_per_group)
    
    return 1.0 - ks_per_group


X = jnp.array([0.1]*n_var)
mass = obj_fun(X)
grad_f = jax.grad(obj_fun)
grad_h = jax.grad(h)
grad_h_VM = jax.jacrev(h_VM)
# Compiler les fonctions avec JIT
f_jit = jax.jit(obj_fun)
grad_f_jit = jax.jit(grad_f)
h_jit = jax.jit(h)
h_VM_jit = jax.jit(h_VM)
grad_h_jit = jax.jit(grad_h) 
grad_h_VM_jit = jax.jit(grad_h_VM)

#computation time before jit
t6 = t.time()
U_max = h_jit(X)
t7 = t.time()
print(f"Time to compute the maximum deflection before JIT: {t7-t6:.6f} seconds")
t8 = t.time()
d_U_max = grad_h_jit(X)
t9 = t.time()
print(f"Time to compute the gradient of the maximum deflection before JIT: {t9-t8:.6f} seconds")
d_vm_max = grad_h_VM_jit(X)
t10 = t.time()
print(f"Time to compute the gradient of the maximum vm stress before JIT: {t10-t9:.6f} seconds")
#computation time after jit
t6 = t.time()
U_max = h_jit(X).block_until_ready()
t7 = t.time()
print(f"Time to compute the maximum deflection after JIT: {t7-t6:.6f} seconds")
t8 = t.time()
d_U_max = grad_h_jit(X).block_until_ready()
t9 = t.time()
print(f"Time to compute the gradient of the maximum deflection after JIT: {t9-t8:.6f} seconds")
d_vm_max = grad_h_VM_jit(X).block_until_ready()
t10 = t.time()
print(f"Time to compute the gradient of the maximum vm stress after JIT: {t10-t9:.6f} seconds")

# #calcul d'un champ de déplacement et des contrainte de VM associées
# element_property = [[h] for h in X]
# K = fem.assembling_K_parametric_sparse(nodes[:,1:], element_property, material)
# rhs = fem.set_rhs(F)
# K, rhs = fem.boundary_conditions_sparse(K, rhs)
# Us = fem.solve_sparse(K,rhs)
# Us_r = Us.reshape((-1,6))
# strain_and_stress = fem.compute_strain_and_stress(nodes[:,0],Us_r,nodes[:,1:],element_property,material)
# stress = strain_and_stress[1]
# vm = fem.compute_vonMises(stress[:,:,-3:].T) #vm stress at each gauss points
# vm_mean = vm.mean(axis=0) #mean over the gauss points
# fem.post_processing_VM(vm_mean,'vm_before_opt')

#conversion to scipy to use scipy optimize SLSQP 
def f_numpy(x):
    # Ensure input to JAX is float64 and output is float64
    return np.float64(f_jit(jnp.array(x, dtype=jnp.float64)))

def grad_f_numpy(x):
    # Ensure input to JAX is float64 and output numpy array is float64
    return np.asarray(grad_f_jit(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)

def h_numpy(x):
    # Ensure input to JAX is float64 and output is float64
    return np.float64(h_jit(jnp.array(x, dtype=jnp.float64)))

def h_VM_numpy(x):
    # Ensure input to JAX is float64 and output is float64
    return np.float64(h_VM_jit(jnp.array(x, dtype=jnp.float64)))

def grad_h_numpy(x):
    # Ensure input to JAX is float64 and output numpy array is float64
    return np.asarray(grad_h_jit(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)

def grad_h_VM_numpy(x):
    # Ensure input to JAX is float64 and output numpy array is float64
    return np.asarray(grad_h_VM_jit(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)

# Point initial
x0 = np.array(X, dtype=np.float64) # Explicitly set x0 to float64

# Définir la contrainte au format scipy
constraint = {
    'type': 'ineq',
    'fun': h_VM_numpy,
    'jac': grad_h_VM_numpy
}

# callback function to monitor optimization progress
# Historique
history = []
def callback(intermediate_result: OptimizeResult):
    xk = intermediate_result.x
    fk = intermediate_result.fun  

    print(f"Iteration: x = {xk}, f(x) = {fk}")
    history.append((xk.copy(), fk))

# Optimisation avec SLSQP

result = minimize(
    fun=f_numpy,
    x0=x0,
    method='SLSQP',
    jac=grad_f_numpy,
    constraints=constraint,
    bounds = [(1e-3,0.1)]*n_var, # bounds on the tck distribution
    options={'disp': True,'maxiter': 50,'ftol': 1e-4},
    callback=callback
)

hist_f = [h[1] for h in history]
hist_g = []
for i in range(len(hist_f)):
    x = history[i][0]
    hist_g.append(h_VM_numpy(x))

# plot the convergence history
fig, ax1 = plt.subplots(figsize=(8, 5))
x = np.arange(len(hist_f))
# objective function history
line1 = ax1.plot(x, hist_f, label='f_obj', color='tab:blue')
ax1.set_xlabel('iterations')
ax1.set_ylabel('f_obj', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# constraint history
ax2 = ax1.twinx()
line2 = ax2.plot(x, -np.array(hist_g), label='g', color='tab:red')
ax2.set_ylabel('g', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Axe logarithmique sur le second axe Y
ax2.set_yscale('log')

plt.legend(line1 + line2, [l.get_label() for l in line1 + line2], loc='best')
plt.title('Convergence history')

plt.savefig(f"{test_case}_optimization_history.pdf")


element_property = [[h] for h in result.x]
K = fem.assembling_K_parametric_sparse(nodes[:,1:], element_property, material)
rhs = fem.set_rhs(F)
K, rhs = fem.boundary_conditions_sparse(K, rhs)
Us = fem.solve_sparse(K,rhs)
Us_r = Us.reshape((-1,6))
strain_and_stress = fem.compute_strain_and_stress(nodes[:,0],Us_r,nodes[:,1:],element_property,material)
stress = strain_and_stress[1]
vm = fem.compute_vonMises(stress[:,:,-3:].T) #vm stress at each gauss points
vm_mean = vm.mean(axis=0) #mean over the gauss points
fem.post_processing_VM(vm_mean,'vm_after_opt')