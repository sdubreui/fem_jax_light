#optimization of a cantilever beam
import numpy as np 
from fem_jax_light.FEM_jax import FEM_study
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from scipy.optimize import minimize, OptimizeResult

# optimal shape of a cantilever beam 
test_case = "cantilever_beam"
mesh_file = "meshes/"+test_case+".msh"
# Material and element properties
E = 6825.0*1e7
nu = 0.3
h = 0.04
element_type = {'tri': 'DKT_jax'}
element_property = [[h]]
material = [[E,nu]] 
# Create FEM study
fem = FEM_study(mesh_file,element_type,element_property,material)
# Right hand side
nodes = fem.nodes
#top nodes 
ind = jnp.where(nodes[:,1]==1.0)
# we apply a non uniform load at the top edge 
y_sections = nodes[ind][:,2]
f_y = np.exp(-((y_sections-8.0)/2.0)**2)
F = np.zeros((len(nodes)*6,))
F[np.array(6*(nodes[ind][:,0]-1)+1,dtype=int)] = f_y
F = jnp.array(F*1e6)
fem.set_rhs(F)


def obj_fun(X):
    # X is the vector that parameterized the shape of the beam thickness 
    y_sections_norm = jnp.array(y_sections/y_sections.max())
    ind_y = jnp.argsort(y_sections_norm)
    y_sections_norm = y_sections_norm[ind_y]
    DT = X[0]*y_sections_norm*(1.0-y_sections_norm)**3 + X[1]*y_sections_norm**2*(1.0-y_sections_norm)**2 + X[2]*y_sections_norm**3*(1.0-y_sections_norm)+X[3]*y_sections_norm**4
    # mesh deformation
    new_nodes = jnp.array(nodes).copy()
    nx = int(nodes.shape[0]/y_sections.shape[0])
    ind_y = jnp.argsort(new_nodes[:,2])
    new_nodes_sorted = new_nodes[ind_y]
    for i in range(y_sections.shape[0]):
        dx = new_nodes_sorted[i*nx:(i+1)*nx,1]
        DY_section = dx+(-DT[i]*dx+DT[i]) 
        new_nodes_sorted = new_nodes_sorted.at[i*nx:(i+1)*nx,1].set(DY_section)
    new_nodes = jnp.zeros_like(new_nodes)    
    new_nodes = new_nodes.at[ind_y].set(new_nodes_sorted)
    #compute the surface
    #from FEM we access the connectivity of the mesh for all elements
    nodes_element_index = fem.elements_tot[:,3:6].astype(int)-1
    coord_nodes_tri = new_nodes[nodes_element_index]
    n_elem = fem.elements_tot.shape[0]

    M1 = jnp.ones((n_elem, 3, 3))
    M2 = jnp.ones((n_elem, 3, 3))
    M3 = jnp.ones((n_elem, 3, 3))
    M1 = M1.at[:, 0, :].set(coord_nodes_tri[:, :, 0])
    M1 = M1.at[:, 1, :].set(coord_nodes_tri[:, :, 1])
    M2 = M2.at[:, 0, :].set(coord_nodes_tri[:, :, 1])
    M2 = M2.at[:, 1, :].set(coord_nodes_tri[:, :, 2])
    M3 = M3.at[:, 0, :].set(coord_nodes_tri[:, :, 2])
    M3 = M3.at[:, 1, :].set(coord_nodes_tri[:, :, 0])
    surfaces_tri = 0.5 * jnp.sqrt(jnp.linalg.det(M1)**2 + jnp.linalg.det(M2)**2 + jnp.linalg.det(M3)**2)
    #Regularisation to avoid 0/sqrt(0) in the derivative
    M1 = M1 + 1e-12
    M2 = M2 + 1e-12
    M3 = M3 + 1e-12
    surface = jnp.sum(surfaces_tri)
    return surface

def h(X):
    # X is the vector that parameterized the shape of the beam thickness 
    y_sections_norm = jnp.array(y_sections/y_sections.max())
    ind_y = jnp.argsort(y_sections_norm)
    y_sections_norm = y_sections_norm[ind_y]
    DT = X[0]*y_sections_norm*(1.0-y_sections_norm)**3 + X[1]*y_sections_norm**2*(1.0-y_sections_norm)**2 + X[2]*y_sections_norm**3*(1.0-y_sections_norm)+X[3]*y_sections_norm**4
    # mesh deformation
    new_nodes = jnp.array(nodes).copy()
    nx = int(nodes.shape[0]/y_sections.shape[0])
    ind_y = jnp.argsort(new_nodes[:,2])
    new_nodes_sorted = new_nodes[ind_y]
    for i in range(y_sections.shape[0]):
        dx = new_nodes_sorted[i*nx:(i+1)*nx,1]
        DY_section = dx+(-DT[i]*dx+DT[i]) 
        new_nodes_sorted = new_nodes_sorted.at[i*nx:(i+1)*nx,1].set(DY_section)
    new_nodes = jnp.zeros_like(new_nodes)    
    new_nodes = new_nodes.at[ind_y].set(new_nodes_sorted)
    # # Assembling the rigidity matrix
    K = fem.assembling_K_parametric(fem.nodes[:,0],new_nodes[:,1:], element_property, material)
    # Set the rhs
    fem.set_rhs(F)
    # Boundary conditions
    ind = fem.nodes[:,2] == 0.0
    nodes_x0 = fem.nodes[ind,0]
    fem.nodes_sets = [nodes_x0]
    l_dof_clamped = [[0,1,2,3,4,5]]
    fem.l_dof = [l_dof_clamped]
    # Apply boundary conditions
    fem.boundary_conditions(fem.nodes_sets,fem.l_dof)
    # solve 
    Us = fem.solve()
    c = Us[0::6].min()+1.5
    return (c)


X = jnp.array([0.0,0.0,0.0,0.0])
surface = obj_fun(X)
U_max = h(X)
print(f"Surface = {surface:.6e} m^2")
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
    return np.array(grad_f_jit(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)

def h_numpy(x):
    # Ensure input to JAX is float64 and output is float64
    return np.float64(h_jit(jnp.array(x, dtype=jnp.float64)))

def grad_h_numpy(x):
    # Ensure input to JAX is float64 and output numpy array is float64
    return np.array(grad_h_jit(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)

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
    bounds = [(0.0,0.9)]*4, # bounds on the twist distribution
    options={'disp': True,'maxiter': 15},
    callback=callback
)


# #Post processing
import matplotlib.pyplot as plt
for j, (xk, fk) in enumerate(history):

    X = jnp.array(xk)
    y_sections_norm = jnp.array(y_sections/y_sections.max())
    ind_y = jnp.argsort(y_sections_norm)
    y_sections_norm = y_sections_norm[ind_y]
    DT = X[0]*y_sections_norm*(1.0-y_sections_norm)**3 + X[1]*y_sections_norm**2*(1.0-y_sections_norm)**2 + X[2]*y_sections_norm**3*(1.0-y_sections_norm)+X[3]*y_sections_norm**4
    # mesh deformation
    new_nodes = jnp.array(nodes).copy()
    nx = int(nodes.shape[0]/y_sections.shape[0])
    ind_y = jnp.argsort(new_nodes[:,2])
    new_nodes_sorted = new_nodes[ind_y]
    for i in range(y_sections.shape[0]):
        dx = new_nodes_sorted[i*nx:(i+1)*nx,1]
        DY_section = dx+(-DT[i]*dx+DT[i]) 
        new_nodes_sorted = new_nodes_sorted.at[i*nx:(i+1)*nx,1].set(DY_section)

    plt.plot(new_nodes_sorted[:,2],new_nodes_sorted[:,1],'+')
plt.axis('equal')

plt.show()