import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import jit, vmap, grad, jacfwd, jacrev
import jax.scipy.linalg as jlinalg
import jax.scipy.sparse.linalg as jspalinalg
from typing import Dict, List, Tuple, Any, Callable
from fem_jax_light.dkt_element_jax import DKT_element
import time as t

"""
Main class for FEM using JAX
"""

class FEM_study():
    """
    Construction:
    
    Inputs:
    mesh_file: fichier de maillage .msh
    element_type: dict avec clés type d'élément et formulation associée
    element_property: dict des propriétés géométriques
    material: dict des propriétés matériaux
    mode: 'full' ou 'sparse' pour la matrice de rigidité
    
    Types d'éléments disponibles:
    element_type['tri'] = "DKT_jax"

    """
    def __init__(self, mesh_file: str, element_type: Dict, element_property: Dict, 
                 material: Dict, mode: str = 'full'):
        self.mesh_file = mesh_file
        self.element_type = element_type
        self.element_property = element_property
        self.material = material
        self.mode = mode
        
        
        # Attributs initialisés plus tard
        self.nodes = None
        self.K = None
        self.rhs = None
        self.element_dict = None
        self.elements_tot = None
        self.surfaces_tri = None
        self.assembler = None
        self.elem_nodes = []
        self.fake_nodes = []
 
        # Lecture du maillage (reste en NumPy)
        element_dict, elements_tot = self.read_mesh_file()

        # creation de l'élément DKT
        self.DKT = DKT_element()


    def read_mesh_file(self) -> Tuple[Dict, np.ndarray]:
        """
        Lit le fichier de maillage et retourne les données de maillage
        
        Outputs:
        element_dict: dict avec clés "nodes" et "elements"
        elements_tot: array numpy de tous les éléments
        
        Note: Cette fonction reste en NumPy car elle fait de l'I/O
        """
        import gmsh
        gmsh.initialize()
        gmsh.open(self.mesh_file)
        
        # Lecture des nœuds
        nodes_tag, nodes_coord, par_coord = gmsh.model.mesh.getNodes() 
        n_nodes = int(len(nodes_tag))
        nodes_coord = nodes_coord.reshape((n_nodes, 3))
        nodes = np.zeros((n_nodes, 4))
        nodes[:, 0] = nodes_tag
        nodes[:, 1:] = nodes_coord
        self.nodes = nodes
        
        # Lecture des éléments
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements()
        n_types = len(elemTypes)
        n_elements = int(np.array([len(elemTags[i]) for i in range(n_types)]).sum())
        elements_tot = np.zeros((n_elements, 7))
        element_sets = {}
        
        physical_groups = np.array(gmsh.model.getPhysicalGroups())
        elem_index = 0
        surfaces_tri = np.zeros((n_elements, 1))
        
        # Boucle sur les groupes physiques
        for pg in physical_groups:
            entities = gmsh.model.getEntitiesForPhysicalGroup(pg[0], pg[1])
            name = gmsh.model.getPhysicalName(pg[0], pg[1])
            
            for i in range(len(entities)):
                elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(pg[0], entities[i])
                n_elem = len(elemTags[0]) 
                n_nodes_e = len(elemNodeTags[0])
                elements_tot[elem_index:elem_index+n_elem, 0] = elemTags[0]
                elements_tot[elem_index:elem_index+n_elem, 1] = elemTypes[0]
                elements_tot[elem_index:elem_index+n_elem, 2] = pg[1]
                
                if elemTypes[0] == 1:  # Beam
                    elements_tot[elem_index:elem_index+n_elem, 3:5] = elemNodeTags[0].reshape((n_elem, 2))
                elif elemTypes[0] == 2:  # Triangle
                    elements_tot[elem_index:elem_index+n_elem, 3:6] = elemNodeTags[0].reshape((n_elem, 3))
                    # Calcul des surfaces des triangles
                    coord_nodes_tri = self.nodes[elemNodeTags[0].reshape((n_elem, 3)).astype(int)-1]
                    coord_nodes_tri = coord_nodes_tri[:, :, 1:] 
                    M1 = np.ones((n_elem, 3, 3))
                    M2 = np.ones((n_elem, 3, 3))
                    M3 = np.ones((n_elem, 3, 3))
                    M1[:, 0, :] = coord_nodes_tri[:, :, 0]
                    M1[:, 1, :] = coord_nodes_tri[:, :, 1]
                    M2[:, 0, :] = coord_nodes_tri[:, :, 1]
                    M2[:, 1, :] = coord_nodes_tri[:, :, 2]
                    M3[:, 0, :] = coord_nodes_tri[:, :, 2]
                    M3[:, 1, :] = coord_nodes_tri[:, :, 0]
                    surfaces_tri[elem_index:elem_index+n_elem, 0] = 0.5 * np.sqrt(
                        np.linalg.det(M1)**2 + np.linalg.det(M2)**2 + np.linalg.det(M3)**2
                    )
                elif elemTypes[0] == 3:  # Quad
                    elements_tot[elem_index:elem_index+n_elem, 3:7] = elemNodeTags[0].reshape((n_elem, 4))
                
                elem_index = elem_index + n_elem
            
            # Remplissage du dictionnaire element_sets
            if elemTypes[0] == 1:
                elemType_name = 'beam'
            elif elemTypes[0] == 2:
                elemType_name = 'tri'
            elif elemTypes[0] == 3:
                elemType_name = 'quad'
            
            ind = elements_tot[:, 2] == pg[1]
            element_sets[pg[1]] = {
                "name": name,
                "elements": elements_tot[ind, 0],
                "element_type": elemType_name,
                "surfaces": surfaces_tri[ind, 0]
            }
        
        self.surfaces_tri = surfaces_tri
        
        # Organisation par type d'élément
        beam_elements_ind = elements_tot[:, 1] == 1
        beam_elements = elements_tot[beam_elements_ind, :]
        n_beam = beam_elements.shape[0]
        print(f"Number of beam elements = {n_beam}")
        
        tri_elements_ind = elements_tot[:, 1] == 2
        tri_elements = elements_tot[tri_elements_ind, :]
        n_tri = tri_elements.shape[0]
        print(f"Number of triangular elements = {n_tri}")
        
        quad_elements_ind = elements_tot[:, 1] == 3
        quad_elements = elements_tot[quad_elements_ind, :]
        n_quad = quad_elements.shape[0]
        print(f"Number of quadrilateral elements = {n_quad}")
        
        # Construction du dictionnaire d'éléments
        element_dict = {'nodes': nodes, 'elements': {}}
        elem_nodes = []
        
        if n_beam > 0:
            beam_elements_temp = np.zeros((n_beam, 3))
            beam_elements_temp[:, 0] = beam_elements[:, 0]
            beam_elements_temp[:, 1:] = beam_elements[:, 3:5]
            element_dict['elements']['beam'] = beam_elements_temp
            for elm in beam_elements_temp:
                for node_id in elm[1:3]:
                    if node_id not in elem_nodes:
                        elem_nodes.append(node_id)
        
        if n_tri > 0:
            tri_elements_temp = np.zeros((n_tri, 4))
            tri_elements_temp[:, 0] = tri_elements[:, 0]
            tri_elements_temp[:, 1:] = tri_elements[:, 3:6]
            element_dict['elements']['tri'] = tri_elements_temp
            for elm in tri_elements_temp:
                for node_id in elm[1:4]:
                    if node_id not in elem_nodes:
                        elem_nodes.append(node_id)
        
        if n_quad > 0:
            quad_elements_temp = np.zeros((n_quad, 5))
            quad_elements_temp[:, 0] = quad_elements[:, 0]
            quad_elements_temp[:, 1:] = quad_elements[:, 3:7]
            element_dict['elements']['quad'] = quad_elements_temp
            for elm in quad_elements_temp:
                for node_id in elm[1:5]:
                    if node_id not in elem_nodes:
                        elem_nodes.append(node_id)
        
        self.elem_nodes = elem_nodes
        element_dict["element_sets"] = element_sets
        
        # Détection des nœuds non utilisés
        self.fake_nodes = []
        for node in self.nodes:
            if node[0] not in elem_nodes:
                self.fake_nodes.append(node[0])
        
        if len(self.fake_nodes) > 0:
            print(f"Warning: {len(self.fake_nodes)} nodes not used detected")
        
        self.elements_tot = elements_tot
        self.element_dict = element_dict
        gmsh.model.remove()
        
        return element_dict, elements_tot
    
    
    def prepare_all_tri_element_coords(self,nodes_index: jnp.ndarray, nodes_coord: jnp.ndarray, 
                                         elements: jnp.ndarray) -> jnp.ndarray:
        """
        Prépare les coordonnées de tous les éléments pour un traitement batch
    
        Cette fonction extrait et organise les coordonnées nodales de tous les 
        éléments dans un seul array pour permettre un calcul vectorisé/parallèle.
        
        Args:
        nodes_index: (n_nodes,) array avec les identifiants des nœuds
        nodes_coord: (n_nodes, 3) array avec [x, y, z] pour chaque nœud
        elements: (n_elements, n_vertex+1) array avec [elem_id, node1, node2, ...]
        
        Returns:
        all_coords: array JAX de forme (n_elements, n_nodes_per_elem, 3)
                contenant les coordonnées (x,y,z) pour chaque élément
        """        

        
        n_nodes_per_elem = 3
    
        n_elements = elements.shape[0]

        # Pas de boucles du tout !
        element_node_ids = elements[:, 1:n_nodes_per_elem+1]
        
        # Broadcasting pour créer un masque
        mask = (element_node_ids[:, :, None] == nodes_index[None, None, :])
        
        # Extraire coordonnées avec le masque
        coords = jnp.sum(nodes_coord[None, None, :, :] * mask[:, :, :, None], axis=2)
        
        return coords


    def assembling_K_parametric(self,nodes_index,nodes_coord,element_properties: list, materials: list):
        """
        Assemble la matrice de rigidité globale en fonction de paramètres variables
        Version fonctionnelle pour la différentiation
        
        Args:
        nodes_index: (n_nodes,) array avec les identifiants des nœuds (static_argnums=0 pour JIT)
        nodes_coord: (n_nodes, 3) array avec [x, y, z] pour chaque nœud
        element_properties: dict des propriétés géométriques
        materials: dict des propriétés matériaux
        
        Returns:
        K: matrice de rigidité (array JAX)
        """
        nodes_index = np.array(nodes_index,dtype=int)
        
        K = jnp.zeros((nodes_index.shape[0]*6, nodes_index.shape[0]*6))
        # Version vectorisée 
        
        # on boucle sur les éléments de réference (pas sur les éléments du maillage), on se limite au element tri donc une seule boucle sur les sets d'éléments
        # en réalité on boucle sur les set d'élements 
        # pour eviter les dynamic shape on fait un seul set et on mask les résultats (calcul de matrice élementaire inutile mais évite la recompilation)
        elements_sets = self.elements_tot[:, [0,3,4,5]]
        all_coords = self.prepare_all_tri_element_coords(nodes_index,nodes_coord, elements_sets)
        self.assembler = FastAssembler(
                elements_sets,
                nodes_index,
                'tri'
            )
        for i in self.element_dict['element_sets'].keys():
            name = self.element_dict['element_sets'][i]['name']
            # creation du mask 
            mask = (self.elements_tot[:, 2] == i)
            compute_K_ref = self.DKT.compute_K_elem
            def compute_K_single_masked(coords, m, materials, properties):
                K = compute_K_ref(coords,materials,properties)
                return jnp.where(m, K, 0.0)
            materials_set = jnp.atleast_2d(materials[i-1]).repeat(mask.shape[0],axis=0)
            property_set = jnp.atleast_2d(element_properties[i-1]).repeat(mask.shape[0],axis=0)
            K_elem = vmap(compute_K_single_masked)(all_coords,mask,materials_set,property_set)

           
            # Assemblage ultra-rapide
            K = self.assembler.assemble(K_elem) + K
        self.K = K
        
        return K





    def prepare_all_element_coords(self,nodes: jnp.ndarray, 
                                         elements: jnp.ndarray,
                                         element_type: str) -> jnp.ndarray:
        """
        Prépare les coordonnées de tous les éléments pour un traitement batch
    
        Cette fonction extrait et organise les coordonnées nodales de tous les 
        éléments dans un seul array pour permettre un calcul vectorisé/parallèle.
        
        Args:
        nodes: (n_nodes, 4) array avec [node_id, x, y, z] pour chaque nœud
        elements: (n_elements, n_vertex+1) array avec [elem_id, node1, node2, ...]
        element_type: type d'élément ('beam', 'tri', 'quad')
        
        Returns:
        all_coords: array JAX de forme (n_elements, n_nodes_per_elem, 3)
                contenant les coordonnées (x,y,z) pour chaque élément
        """        

        if element_type == 'beam':
            n_nodes_per_elem = 2
        elif element_type == 'tri':
            n_nodes_per_elem = 3
        elif element_type == 'quad':
            n_nodes_per_elem = 4
        else:
            raise ValueError(f"Type d'élément inconnu: {element_type}")
    
        n_elements = elements.shape[0]

        # Pas de boucles du tout !
        element_node_ids = elements[:, 1:n_nodes_per_elem+1]
        
        # Broadcasting pour créer un masque
        mask = (element_node_ids[:, :, None] == nodes[None, None, :, 0])
        
        # Extraire coordonnées avec le masque
        coords = jnp.sum(nodes[None, None, :, 1:4] * mask[:, :, :, None], axis=2)
        
        return coords

    
    def boundary_conditions(self, nodes_set: List[List[int]], l_dof: List[List[int]]) -> None:
        """
        Applique les conditions aux limites
        
        Args:
        nodes_set: liste de listes de nœuds à contraindre
        l_dof: liste de listes de DDL à bloquer pour chaque ensemble
        """
        self.nodes_set = nodes_set
        self.l_dof = l_dof
        for i, nodes in enumerate(nodes_set):
            for node in nodes:
                ind_node = np.argwhere(self.nodes[:, 0] == node)[0, 0]
                
                if self.mode == 'full':
                    for dof in l_dof[i]:
                        self.K = self.K.at[:, 6*ind_node + dof].set(0.0)
                        self.K = self.K.at[6*ind_node + dof, :].set(0.0)
                        self.K = self.K.at[6*ind_node + dof, 6*ind_node + dof].set(1.0)
                        self.rhs = self.rhs.at[6*ind_node + dof].set(0.0)
                                    
                elif self.mode == 'sparse':                    
                    for dof in l_dof[i]:
                        ind_row = self.row == 6*ind_node + dof
                        self.data = self.data.at[ind_row].set(0.0)
                        ind_col = self.col == 6*ind_node + dof
                        self.data = self.data[ind_col].set(0.0)
                        ind_diag = ind_col & ind_row
                        self.data = self.data.at[ind_diag].set(1.0)
                        self.rhs = self.rhs.at[6*ind_node + dof].set(0.0)
                    
    
    def set_rhs(self, rhs: np.ndarray) -> None:
        """
        Définit le second membre
        
        Args:
        rhs: vecteur du second membre (doit être de dimension 6*n_nodes)
        """
        if rhs.shape[0] != 6 * len(self.nodes):
            raise ValueError(f'Dimension incorrecte du RHS! '
                           f'Attendu: {6*len(self.nodes)}, reçu: {rhs.shape[0]}')
        self.rhs = jnp.array(rhs)
    
    def solve(self) -> jnp.ndarray:
        """
        Résout le système KU = F
        
        Returns:
        U: vecteur des déplacements (array JAX)
        """
        # Résolution avec JAX
        U = jlinalg.solve(self.K, self.rhs)
        return U
    
    def compute_strain_and_stress(self,nodes_index, U: jnp.ndarray,nodes_coord,element_properties: list, materials: list) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Calcule les déformations et contraintes aux points de Gauss pour tous les éléments
        
        Args:
        U: vecteur des déplacements (array JAX) (n_nodes,6)
        
        Returns:
        strains: array JAX avec les déformations pour chaque élément exprimées dans la base locale de l'élément
        stresses: array JAX avec les contraintes pour chaque élément exprimées dans la base locale de l'élément
        points_gauss: array JAX avec les coordonnées des points de Gauss
        """
        # Cette fonction peut être implémentée en utilisant une approche similaire à compute_K_parametric
        # en calculant les déformations et contraintes élémentaires par batch, puis en les assemblant dans des arrays globaux.

        strains_list: List[jnp.ndarray]  = []
        stresses_list: List[jnp.ndarray] = []
        points_list: List[jnp.ndarray]   = []
        #comme pour compute_K_elem il est préfarable d'utiliser un mask
        elements_sets = self.elements_tot[:, [0,3,4,5]]
        all_coords = self.prepare_all_tri_element_coords(nodes_index,nodes_coord, elements_sets)
        elem_nodes = (elements_sets[:,1:]-1).astype(int)
        U_elem = U[elem_nodes]
        U_elem_flat = U_elem.reshape(U_elem.shape[0],-1)
        for i in self.element_dict['element_sets'].keys():
            name = self.element_dict['element_sets'][i]['name']
            mask = (self.elements_tot[:, 2] == i)
            compute_strain_stress_ref = self.DKT.compute_strain_and_stress
            def compute_strain_stress(coords,U,material,element_property):
                strain,stress,points = compute_strain_stress_ref(coords,U,material,element_property)
                return strain,stress,points

            materials_set = jnp.atleast_2d(materials[i-1]).repeat(mask.shape[0],axis=0)
            property_set = jnp.atleast_2d(element_properties[i-1]).repeat(mask.shape[0],axis=0)
            results = vmap(compute_strain_stress)(all_coords,U_elem_flat,materials_set,property_set)
            strains_list.append(results[0][mask,:,:])
            stresses_list.append(results[1][mask,:,:])
            points_list.append(results[2][mask,:,:])
        # concatène les résultats de tous les jeux d'éléments
        strains      = jnp.concatenate(strains_list, axis=0)
        stresses     = jnp.concatenate(stresses_list, axis=0)
        points_gauss = jnp.concatenate(points_list, axis=0)

        return strains, stresses, points_gauss    


    def compute_vonMises(self, stress):
        """Calcule la contrainte de Von Mises à partir des contraintes dans le repère de l'élément"""
        sigma_x = stress[0]
        sigma_y = stress[1]
        tau_xy = stress[2]
        
        von_mises = jnp.sqrt(sigma_x**2 - sigma_x*sigma_y + sigma_y**2 + 3*tau_xy**2)
        return von_mises
  
    def post_processing(self,U,file_name):
        """
        Displacement Post processing file for gmsh
        """  
       #copying mesh file
        with open(self.mesh_file) as f:
            with open(file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        f1 = open(file_name+".msh",'a')
        #vector displacement field at nodes
        f1.write("\n")
        f1.write("$NodeData\n")
        f1.write("1\n")
        f1.write('"Displacement"\n')
        f1.write("1\n")
        f1.write("0.0\n")
        f1.write("3\n")
        f1.write("0\n")
        f1.write("3\n")
        f1.write(str(int(len(self.nodes)))+"\n")
        for i in range(len(self.nodes)):
            f1.write(str(i+1)+" "+str(U[6*i])+" "+str(U[6*i+1])+" "+str(U[6*i+2])+"\n")
        f1.write("$EndNodeData")    
        return


class FastAssembler:
    """
    Classe qui pré-calcule les indices d'assemblage une seule fois
    Puis assemble très rapidement pour différentes matrices K
    
    Idéal pour:
    - Calculs de sensibilités (K change, connectivité reste la même)
    - Optimisation (nombreux assemblages)
    """
    
    def __init__(self, elements: np.ndarray, nodes: np.ndarray, 
                 element_type: str):
        """
        Pré-calcule les indices d'assemblage
        
        Args:
        elements: (n_elements, n_nodes_per_elem+1) connectivité
        nodes: (n_nodes,) array avec les identifiants des nœuds
        element_type: 'beam', 'tri', ou 'quad'
        """
        self.element_type = element_type
        
        if element_type == 'beam':
            self.n_nodes_per_elem = 2
        elif element_type == 'tri':
            self.n_nodes_per_elem = 3
        elif element_type == 'quad':
            self.n_nodes_per_elem = 4
        
        self.n_elements = elements.shape[0]
        self.dof_per_elem = self.n_nodes_per_elem * 6
        self.n_dof_global = nodes.shape[0] * 6
        
        # Mapping node_id -> node_index
        node_id_to_idx = {int(nodes[i]): i for i in range(nodes.shape[0])}
        
        # Convertir node_ids en indices
        node_ids = elements[:, 1:self.n_nodes_per_elem+1].astype(int)
        self.node_indices = np.zeros_like(node_ids)
        for i in range(self.n_elements):
            for j in range(self.n_nodes_per_elem):
                self.node_indices[i, j] = node_id_to_idx[node_ids[i, j]]
        
        self.node_indices = jnp.array(self.node_indices)
        
        # Pré-calculer les indices globaux pour scatter
        self._precompute_indices()
    
    def _precompute_indices(self):
        """Pré-calcule tous les indices d'assemblage"""
        
        # Indices locaux
        local_i = jnp.arange(self.dof_per_elem)
        local_j = jnp.arange(self.dof_per_elem)
        local_ii, local_jj = jnp.meshgrid(local_i, local_j, indexing='ij')
        
        node_idx_i = local_ii // 6
        node_idx_j = local_jj // 6
        dof_offset_i = local_ii % 6
        dof_offset_j = local_jj % 6
        
        # Indices globaux pour tous les éléments
        global_node_i = self.node_indices[:, node_idx_i]
        global_node_j = self.node_indices[:, node_idx_j]
        
        self.global_i = global_node_i * 6 + dof_offset_i[None, :, :]
        self.global_j = global_node_j * 6 + dof_offset_j[None, :, :]
        
        # Aplatir pour scatter
        self.flat_indices = jnp.stack([
            self.global_i.flatten(), 
            self.global_j.flatten()
        ], axis=1)
    
    @partial(jit, static_argnums=(0,))
    def assemble(self, K_elements: jnp.ndarray) -> jnp.ndarray:
        """
        Assemble les matrices élémentaires (ultra-rapide)
        
        Args:
        K_elements: (n_elements, dof_per_elem, dof_per_elem)
        
        Returns:
        K_global: matrice assemblée
        """
        values = K_elements.flatten()
        
        K_global = jnp.zeros((self.n_dof_global, self.n_dof_global))
        K_global = K_global.at[self.flat_indices[:, 0], self.flat_indices[:, 1]].add(values)
        
        return K_global
    
    def assemble_coo(self, K_elements: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Assemble en format COO (pour solveurs sparse)
        
        Returns:
        row, col, data: triplets COO
        """
        values = K_elements.flatten()
        return self.flat_indices[:, 0], self.flat_indices[:, 1], values





   