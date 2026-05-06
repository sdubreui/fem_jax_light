import numpy as np 
from fem_jax_light.FEM_jax import FEM_study
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

#circular plate with clamped boundary conditions and load at the center 
test_case = "circular_plate"
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
ind = np.sqrt(nodes[:,1]**2+nodes[:,2]**2+nodes[:,3]**2).argmin()
nodes_x1 = nodes[ind,0]
F = jnp.zeros((len(nodes)*6,))
F = F.at[int(6*(nodes_x1-1)+2)].set(1000.0)
rhs = fem.set_rhs(F)
# Assembling the rigidity matrix
K = fem.assembling_K_parametric(nodes[:,1:], element_property, material)
# Boundary conditions
ind = np.sqrt(nodes[:,1]**2+nodes[:,2]**2+nodes[:,3]**2) >= 0.48
nodes_x0 = nodes[ind,0]
nodes_sets = [nodes_x0]
l_dof_clamped = [[0,1,2,3,4,5]]
l_dof = [l_dof_clamped]
fem.create_constrained_DOFs(nodes_sets, l_dof)
# Apply boundary conditions
K, rhs = fem.boundary_conditions(K, rhs)
# solve 
Us = fem.solve(K,rhs)
# comparison with reference solution
a = 0.5
F1 = 1000.0
w = 3.0*(1.0-nu**2)*a**2*F1/(4.0*np.pi*E*h**3)
U_max = Us[2:-1:6].max()
epsilon = abs((w-U_max)/w)
print(f"Test case: {test_case}")
print(f"Maximum deflection at the center = {U_max:.6e} m")
print(f"Reference solution for maximum deflection = {w:.6e} m")
print(f"Relative error = {epsilon:.2%}")

# sparse solver 
test_case = "circular_plate"
mesh_file = "meshes/"+test_case+".msh"
# Material and element properties
E = 6825.0*1e7
nu = 0.3
h = 0.04
element_type = {'tri': 'DKT_jax'}
element_property = [[h]]
material = [[E,nu]] 
# Create FEM study
fem_sp = FEM_study(mesh_file,element_type,element_property,material)
# Right hand side
nodes = fem_sp.nodes
ind = np.sqrt(nodes[:,1]**2+nodes[:,2]**2+nodes[:,3]**2).argmin()
nodes_x1 = nodes[ind,0]
F = jnp.zeros((len(nodes)*6,))
F = F.at[int(6*(nodes_x1-1)+2)].set(1000.0)
rhs = fem_sp.set_rhs(F)
# Assembling the rigidity matrix
K_sp = fem_sp.assembling_K_parametric_sparse(nodes[:,1:], element_property, material)
# Boundary conditions
fem_sp.create_constrained_DOFs(nodes_sets, l_dof)
# Apply boundary conditions
K_sp, rhs = fem_sp.boundary_conditions_sparse(K_sp, rhs)
# solve 
Us_sparse = fem_sp.solve_sparse(K_sp, rhs)
# comparison with reference solution
a = 0.5
F1 = 1000.0
w = 3.0*(1.0-nu**2)*a**2*F1/(4.0*np.pi*E*h**3)
U_max = Us_sparse[2:-1:6].max()
epsilon = abs((w-U_max)/w)
print(f"Test case sparse solver: {test_case}")
print(f"Maximum deflection at the center = {U_max:.6e} m")
print(f"Reference solution for maximum deflection = {w:.6e} m")
print(f"Relative error = {epsilon:.2%}")




#circular plate simply supported and load at the center 
test_case = "circular_plate"
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
ind = np.sqrt(nodes[:,1]**2+nodes[:,2]**2+nodes[:,3]**2).argmin()
nodes_x1 = nodes[ind,0]
F = jnp.zeros((len(nodes)*6,))
F = F.at[int(6*(nodes_x1-1)+2)].set(1000.0)
rhs = fem.set_rhs(F)
# Assembling the rigidity matrix
K = fem.assembling_K_parametric(nodes[:,1:], element_property, material)
# Boundary conditions
ind = np.sqrt(nodes[:,1]**2+nodes[:,2]**2+nodes[:,3]**2) >= 0.48
nodes_x0 = nodes[ind,0]
nodes_sets = [nodes_x0]
l_dof_simply_supported = [[2]]
l_dof = [l_dof_simply_supported]
fem.create_constrained_DOFs(nodes_sets, l_dof)
# Apply boundary conditions
K, rhs = fem.boundary_conditions(K, rhs)
# solve 
Us = fem.solve(K,rhs)
# Post-processing
fem.post_processing(Us,"test")
# comparison with reference solution
a = 0.5
F1 = 1000.0
w = 3.0*(3.0+nu)*(1-nu)*a**2*F1/(4.0*np.pi*E*h**3)
U_max = Us[2:-1:6].max()
epsilon = abs((w-U_max)/w)
print(f"Test case: {test_case}")
print(f"Maximum deflection at the center = {U_max:.6e} m")
print(f"Reference solution for maximum deflection = {w:.6e} m")
print(f"Relative error = {epsilon:.2%}")


# sparse solver 
test_case = "circular_plate"
mesh_file = "meshes/"+test_case+".msh"
# Material and element properties
E = 6825.0*1e7
nu = 0.3
h = 0.04
element_type = {'tri': 'DKT_jax'}
element_property = [[h]]
material = [[E,nu]] 
# Create FEM study
fem_sp = FEM_study(mesh_file,element_type,element_property,material)
# Right hand side
nodes = fem_sp.nodes
ind = np.sqrt(nodes[:,1]**2+nodes[:,2]**2+nodes[:,3]**2).argmin()
nodes_x1 = nodes[ind,0]
F = jnp.zeros((len(nodes)*6,))
F = F.at[int(6*(nodes_x1-1)+2)].set(1000.0)
rhs = fem_sp.set_rhs(F)
# Assembling the rigidity matrix
K_sp = fem_sp.assembling_K_parametric_sparse(nodes[:,1:], element_property, material)
# Boundary conditions
fem_sp.create_constrained_DOFs(nodes_sets, l_dof)
# Apply boundary conditions
K_sp, rhs = fem_sp.boundary_conditions_sparse(K_sp, rhs)
# solve 
Us_sparse = fem_sp.solve_sparse(K_sp, rhs)
# comparison with reference solution
a = 0.5
F1 = 1000.0
w = 3.0*(3.0+nu)*(1-nu)*a**2*F1/(4.0*np.pi*E*h**3)
U_max = Us_sparse[2:-1:6].max()
epsilon = abs((w-U_max)/w)
print(f"Test case sparse solver: {test_case}")
print(f"Maximum deflection at the center = {U_max:.6e} m")
print(f"Reference solution for maximum deflection = {w:.6e} m")
print(f"Relative error = {epsilon:.2%}")


# #sphere
# test_case = "quarter_sphere"
# mesh_file = "meshes/"+test_case+".msh"
# # Material and element properties
# E = 6825.0*1e7
# nu = 0.3
# h = 0.04
# element_type = {'tri': 'DKT_jax'}
# element_property = [[h]]
# material = [[E,nu]] 
# # Create FEM study
# fem = FEM_study(mesh_file,element_type,element_property,material)
# # Right hand side
# nodes = fem.nodes
# ind_x_max = nodes[:,1].argmax()
# ind_z_max = nodes[:,3].argmax()  
# x_max = int(nodes[ind_x_max,0])
# z_max = int(nodes[ind_z_max,0])
# F = jnp.zeros((len(nodes)*6,))
# F = F.at[6*(x_max-1)].set(2000.0)
# F = F.at[6*(z_max-1)+2].set(-2000)
# fem.set_rhs(F)
# # Assembling the rigidity matrix
# K = fem.assembling_K_parametric(nodes[:,0],nodes[:,1:], element_property, material)
# # Boundary conditions
# nodes_sets = []
# #find y max node
# ind = nodes[:,2].argmax()   
# node_top = nodes[ind,0]
# nodes_sets.append([node_top])
# #find nodes such as x == 0
# ind = nodes[:,1] == 0.0
# nodes_x0 = nodes[ind,0]
# nodes_sets.append(nodes_x0)
# #find nodes such as z == 0
# ind = nodes[:,3] == 0.0
# nodes_z0 = nodes[ind,0]
# nodes_sets.append(nodes_z0)
# #constraints dof for each node set
# l_dof = [[0,1,2,3,4,5],[0,4,5],[2,3,4]]
# fem.nodes_sets = nodes_sets
# fem.l_dof = l_dof
# # Apply boundary conditions
# fem.boundary_conditions(fem.nodes_sets,fem.l_dof)
# # solve 
# Us = fem.solve()
# # Post-processing
# fem.post_processing(Us,"test_sphere")
