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
from scipy.optimize import minimize, OptimizeResult
import matplotlib.pyplot as plt

test_case = "wing_spar"
mesh_file = "meshes/"+test_case+".msh"
# Material and element properties
E = 6825.0*1e7
nu = 0.3
h = 0.04
element_type = {'tri': 'DKT_jax'}
element_property = [[h]]*30
material = [[E,nu]]*30 
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

#solve the initial problem 
K = fem.assembling_K_parametric(nodes[:,1:], element_property, material)
K, rhs = fem.boundary_conditions(K, rhs)
Us = fem.solve(K,rhs)
fem.post_processing(Us,"meshes/wing_spar_U")

#compute the surfaces
surfaces = jnp.array([fem.element_dict['element_sets'][i+1]['surfaces'].sum() for i in range(30)])


#objective function: mass
def obj_fun(X):
    # X is the vector that parameterized the shape of the beam thickness using element properties
    mass = (X*surfaces).sum()
    return mass 

def h(X):
    # X is the vector that parameterized the shape of the beam thickness using element properties
    element_property = [[h] for h in X]
    K = fem.assembling_K_parametric(nodes[:,1:], element_property, material)
    rhs = fem.set_rhs(F)
    K, rhs = fem.boundary_conditions(K, rhs)
    Us = fem.solve(K,rhs)
    return (2.0-Us[2::6].max())


X = jnp.array([0.04]*30)
mass = obj_fun(X)
U_max = h(X)
print(f"mass = {mass:.6e} m^2")
print(f"Maximum deflection at the top edge = {U_max:.6e} m")
grad_f = jax.grad(obj_fun)
grad_h = jax.grad(h)

# Compiler les fonctions avec JIT
f_jit = jax.jit(obj_fun)
grad_f_jit = jax.jit(grad_f)
h_jit = jax.jit(h)
grad_h_jit = jax.jit(grad_h)

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

def grad_h_numpy(x):
    # Ensure input to JAX is float64 and output numpy array is float64
    return np.asarray(grad_h_jit(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)

# Point initial
x0 = np.array(X, dtype=np.float64) # Explicitly set x0 to float64

# Définir la contrainte au format scipy
constraint = {
    'type': 'ineq',
    'fun': h_numpy,
    'jac': grad_h_numpy
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
    bounds = [(1e-3,0.1)]*30, # bounds on the tck distribution
    options={'disp': True,'maxiter': 50,'ftol': 1e-4},
    callback=callback
)

hist_f = [h[1] for h in history]
hist_g = []
for i in range(len(hist_f)):
    x = history[i][0]
    hist_g.append(h_numpy(x))

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

plt.savefig("wing_spar_optimization_history.pdf")
