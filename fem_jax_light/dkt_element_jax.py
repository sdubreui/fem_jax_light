import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Dict, Tuple

"""
Implementation of DKT (Discrete Kirchhoff Triangle) finite element using JAX
Version optimisée avec @jit pour des performances maximales
"""

class DKT_element():
    def __init__(self):#, element_property: Dict, material: Dict):
        
        
        # Points d'intégration de Gauss (coordonnées barycentriques)
        self.int_nodes = jnp.array([[1.0/6.0, 1.0/6.0],
                                     [2.0/3.0, 1.0/6.0],
                                     [1.0/6.0, 2.0/3.0]])
        self.int_weights = jnp.ones(3) * 1.0/6.0
        
        # Pré-calculer les matrices constitutives (une seule fois) => Non car on veut rendre la fonction de calcul de K_elem paramétrique
        # self.element_property = element_property
        # self.material = material
        # E = material['E']
        # nu = material['nu']
        # h = element_property['h']
        
        # H_temp = jnp.eye(3)
        # H_temp = H_temp.at[0, 1].set(nu)
        # H_temp = H_temp.at[1, 0].set(nu)
        # H_temp = H_temp.at[2, 2].set((1.0 - nu) / 2.0)
        
        #self.H_m = E * h / (1.0 - nu**2) * H_temp
        #self.H_f = E * h**3 / (12.0 * (1.0 - nu**2)) * H_temp
        
        # Matrice de transformation pour beta
        self.transform_matrix = jnp.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
    

    @staticmethod
    @jit
    def compute_H(material, element_property):
        # E = material["E"]
        # nu = material["nu"]
        # h = element_property["h"]

        E = material[0]
        nu = material[1]
        h = element_property[0]

        H_temp = jnp.eye(3)
        H_temp = H_temp.at[0, 1].set(nu)
        H_temp = H_temp.at[1, 0].set(nu)
        H_temp = H_temp.at[2, 2].set((1.0 - nu) / 2.0)

        H_m = E * h / (1.0 - nu**2) * H_temp
        H_f = E * h**3 / (12.0 * (1.0 - nu**2)) * H_temp

        return H_m, H_f

    @partial(jit, static_argnums=(0,))
    def compute_K_elem(self, nodes: jnp.ndarray, material, element_property) -> jnp.ndarray:
        """
        Calcule la matrice de rigidité élémentaire (version JIT optimisée)
        
        Args:
        nodes: (3, 3) coordonnées des 3 nœuds du triangle
        
        Returns:
        K_e_g: (18, 18) matrice de rigidité dans le repère global
        """
        H_m, H_f = self.compute_H(material, element_property)
        # Appel de la fonction statique JIT-compilée
        return self._compute_K_static(
            nodes, 
            H_m, 
            H_f,
            self.int_nodes,
            self.int_weights,
            self.transform_matrix
        )
    
    @staticmethod
    @jit
    def _compute_K_static(nodes, H_m, H_f, int_nodes, int_weights, transform_matrix):
        """
        Fonction statique pour le calcul JIT-compilé
        Tout le calcul est ici pour maximiser les optimisations JIT
        """
        # === REPÈRE LOCAL ===
        u = nodes[1, :] - nodes[0, :]
        v_temp = nodes[2, :] - nodes[0, :]
        n = jnp.cross(u, v_temp)
        v = jnp.cross(n, u)
        
        u = u / jnp.linalg.norm(u)
        v = v / jnp.linalg.norm(v)
        n = n / jnp.linalg.norm(n)
        
        r_glob_to_loc = jnp.stack([u, v, n])
        coord_loc = jnp.dot(r_glob_to_loc, nodes.T).T
        
        # === GÉOMÉTRIE (vectorisé) ===
        diff = jnp.array([
            coord_loc[1, :2] - coord_loc[0, :2],
            coord_loc[2, :2] - coord_loc[1, :2],
            coord_loc[0, :2] - coord_loc[2, :2]
        ])
        lengths = jnp.linalg.norm(diff, axis=1)
        Cos = diff[:, 0] / lengths
        Sin = diff[:, 1] / lengths
        
        # === INTÉGRATION ===
        K_m = jnp.zeros((6, 6))
        K_f_temp = jnp.zeros((9, 9))
        
        # Boucle d'intégration (sera déroulée par JIT)
        for k in range(3):
            xi, eta = int_nodes[k]
            
            # Dérivées des fonctions de forme (inline pour JIT)
            d_N = jnp.array([
                [-1.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [4.0*(1.0 - 2.0*xi - eta), -4.0*xi],
                [4.0*eta, 4.0*xi],
                [-4.0*eta, 4.0*(1.0 - xi - 2.0*eta)]
            ])
            
            # Jacobienne (vectorisé)
            J = jnp.array([
                [jnp.dot(d_N[:3, 0], coord_loc[:, 0]), jnp.dot(d_N[:3, 0], coord_loc[:, 1])],
                [jnp.dot(d_N[:3, 1], coord_loc[:, 0]), jnp.dot(d_N[:3, 1], coord_loc[:, 1])]
            ])
            
            det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            inv_J = jnp.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det_J
            
            # === MATRICE B_m (vectorisée) ===
            dN_dx = inv_J[0, 0]*d_N[:3, 0] + inv_J[0, 1]*d_N[:3, 1]
            dN_dy = inv_J[1, 0]*d_N[:3, 0] + inv_J[1, 1]*d_N[:3, 1]
            
            B_m = jnp.zeros((3, 6))
            B_m = B_m.at[0, 0::2].set(dN_dx)
            B_m = B_m.at[1, 1::2].set(dN_dy)
            B_m = B_m.at[2, 0::2].set(dN_dy)
            B_m = B_m.at[2, 1::2].set(dN_dx)
            
            K_m = K_m + int_weights[k] * (B_m.T @ H_m @ B_m) * jnp.abs(det_J)
            
            # === MATRICE B_f ===
            d_P = d_N[3:, :]
            B_f = DKT_element._compute_B_f_static(inv_J, d_N, d_P, lengths, Cos, Sin)
            
            K_f_temp = K_f_temp + int_weights[k] * (B_f.T @ H_f @ B_f) * jnp.abs(det_J)
        
        # === TRANSFORMATION K_f (vectorisée avec lax.fori_loop ou déployée) ===
        K_f_temp_2 = jnp.zeros((9, 9))
        for i in range(3):
            K_f_temp_2 = K_f_temp_2.at[:, i*3].set(K_f_temp[:, i*3])
            K_f_temp_2 = K_f_temp_2.at[:, i*3+1].set(-K_f_temp[:, i*3+2])
            K_f_temp_2 = K_f_temp_2.at[:, i*3+2].set(K_f_temp[:, i*3+1])
        
        K_f = jnp.zeros((9, 9))
        for i in range(3):
            K_f = K_f.at[i*3, :].set(K_f_temp_2[i*3, :])
            K_f = K_f.at[i*3+1, :].set(-K_f_temp_2[i*3+2, :])
            K_f = K_f.at[i*3+2, :].set(K_f_temp_2[i*3+1, :])
        
        # Rigidité fictive pour theta_z
        diag = jnp.diagonal(K_f)
        min_val = jnp.min(jnp.where(diag > 1e-15, diag, jnp.inf))
        epsilon = 1e-5 * min_val
        
        # === ASSEMBLAGE (18×18) ===
        K_e = jnp.zeros((18, 18))
        
        # Assemblage avec boucles (déroulées par JIT)
        for i in range(3):
            # Membrane
            K_e = K_e.at[i*6:i*6+2, i*6:i*6+2].set(K_m[i*2:i*2+2, i*2:i*2+2])
            # Flexion
            K_e = K_e.at[i*6+2:i*6+5, i*6+2:i*6+5].set(K_f[i*3:i*3+3, i*3:i*3+3])
            # Rigidité fictive theta_z
            K_e = K_e.at[i*6+5, i*6+5].set(epsilon)
            
            # Termes hors-diagonale
            for j in range(2-i):
                idx_j = i + j + 1
                # Membrane
                K_e = K_e.at[idx_j*6:idx_j*6+2, i*6:i*6+2].set(
                    K_m[idx_j*2:idx_j*2+2, i*2:i*2+2])
                K_e = K_e.at[i*6:i*6+2, idx_j*6:idx_j*6+2].set(
                    K_m[i*2:i*2+2, idx_j*2:idx_j*2+2])
                # Flexion
                K_e = K_e.at[idx_j*6+2:idx_j*6+5, i*6+2:i*6+5].set(
                    K_f[idx_j*3:idx_j*3+3, i*3:i*3+3])
                K_e = K_e.at[i*6+2:i*6+5, idx_j*6+2:idx_j*6+5].set(
                    K_f[i*3:i*3+3, idx_j*3:idx_j*3+3])
        
        # === ROTATION VERS REPÈRE GLOBAL ===
        R = jnp.zeros((18, 18))
        for i in range(6):
            R = R.at[i*3:i*3+3, i*3:i*3+3].set(r_glob_to_loc)
        
        K_e_g = R.T @ K_e @ R
        
        return K_e_g
    
    @staticmethod
    @jit
    def _compute_B_f_static(inv_J, d_N, d_P, L, Cos, Sin):
        """
        Calcule la matrice B pour la déformation de flexion (version JIT)
        """
        # Composantes B_x_xi (vectorisées au maximum)
        B_x_xi = jnp.array([
            6.*d_P[0, 0]*Cos[0]/(4.*L[0]) - 6.*d_P[2, 0]*Cos[2]/(4.*L[2]),
            d_N[0, 0] - 3./4.*(d_P[0, 0]*Cos[0]**2 + d_P[2, 0]*Cos[2]**2),
            -3./4.*(d_P[0, 0]*Cos[0]*Sin[0] + d_P[2, 0]*Cos[2]*Sin[2]),
            6.*d_P[1, 0]*Cos[1]/(4.*L[1]) - 6.*d_P[0, 0]*Cos[0]/(4.*L[0]),
            d_N[1, 0] - 3./4.*(d_P[1, 0]*Cos[1]**2 + d_P[0, 0]*Cos[0]**2),
            -3./4.*(d_P[1, 0]*Cos[1]*Sin[1] + d_P[0, 0]*Cos[0]*Sin[0]),
            6.*d_P[2, 0]*Cos[2]/(4.*L[2]) - 6.*d_P[1, 0]*Cos[1]/(4.*L[1]),
            d_N[2, 0] - 3./4.*(d_P[2, 0]*Cos[2]**2 + d_P[1, 0]*Cos[1]**2),
            -3./4.*(d_P[2, 0]*Cos[2]*Sin[2] + d_P[1, 0]*Cos[1]*Sin[1])
        ])
        
        B_x_eta = jnp.array([
            6.*d_P[0, 1]*Cos[0]/(4.*L[0]) - 6.*d_P[2, 1]*Cos[2]/(4.*L[2]),
            d_N[0, 1] - 3./4.*(d_P[0, 1]*Cos[0]**2 + d_P[2, 1]*Cos[2]**2),
            -3./4.*(d_P[0, 1]*Cos[0]*Sin[0] + d_P[2, 1]*Cos[2]*Sin[2]),
            6.*d_P[1, 1]*Cos[1]/(4.*L[1]) - 6.*d_P[0, 1]*Cos[0]/(4.*L[0]),
            d_N[1, 1] - 3./4.*(d_P[1, 1]*Cos[1]**2 + d_P[0, 1]*Cos[0]**2),
            -3./4.*(d_P[1, 1]*Cos[1]*Sin[1] + d_P[0, 1]*Cos[0]*Sin[0]),
            6.*d_P[2, 1]*Cos[2]/(4.*L[2]) - 6.*d_P[1, 1]*Cos[1]/(4.*L[1]),
            d_N[2, 1] - 3./4.*(d_P[2, 1]*Cos[2]**2 + d_P[1, 1]*Cos[1]**2),
            -3./4.*(d_P[2, 1]*Cos[2]*Sin[2] + d_P[1, 1]*Cos[1]*Sin[1])
        ])
        
        B_y_xi = jnp.array([
            6.*d_P[0, 0]*Sin[0]/(4.*L[0]) - 6.*d_P[2, 0]*Sin[2]/(4.*L[2]),
            -3./4.*(d_P[0, 0]*Cos[0]*Sin[0] + d_P[2, 0]*Cos[2]*Sin[2]),
            d_N[0, 0] - 3./4.*(d_P[0, 0]*Sin[0]**2 + d_P[2, 0]*Sin[2]**2),
            6.*d_P[1, 0]*Sin[1]/(4.*L[1]) - 6.*d_P[0, 0]*Sin[0]/(4.*L[0]),
            -3./4.*(d_P[1, 0]*Cos[1]*Sin[1] + d_P[0, 0]*Cos[0]*Sin[0]),
            d_N[1, 0] - 3./4.*(d_P[1, 0]*Sin[1]**2 + d_P[0, 0]*Sin[0]**2),
            6.*d_P[2, 0]*Sin[2]/(4.*L[2]) - 6.*d_P[1, 0]*Sin[1]/(4.*L[1]),
            -3./4.*(d_P[2, 0]*Cos[2]*Sin[2] + d_P[1, 0]*Cos[1]*Sin[1]),
            d_N[2, 0] - 3./4.*(d_P[2, 0]*Sin[2]**2 + d_P[1, 0]*Sin[1]**2)
        ])
        
        B_y_eta = jnp.array([
            6.*d_P[0, 1]*Sin[0]/(4.*L[0]) - 6.*d_P[2, 1]*Sin[2]/(4.*L[2]),
            -3./4.*(d_P[0, 1]*Cos[0]*Sin[0] + d_P[2, 1]*Cos[2]*Sin[2]),
            d_N[0, 1] - 3./4.*(d_P[0, 1]*Sin[0]**2 + d_P[2, 1]*Sin[2]**2),
            6.*d_P[1, 1]*Sin[1]/(4.*L[1]) - 6.*d_P[0, 1]*Sin[0]/(4.*L[0]),
            -3./4.*(d_P[1, 1]*Cos[1]*Sin[1] + d_P[0, 1]*Cos[0]*Sin[0]),
            d_N[1, 1] - 3./4.*(d_P[1, 1]*Sin[1]**2 + d_P[0, 1]*Sin[0]**2),
            6.*d_P[2, 1]*Sin[2]/(4.*L[2]) - 6.*d_P[1, 1]*Sin[1]/(4.*L[1]),
            -3./4.*(d_P[2, 1]*Cos[2]*Sin[2] + d_P[1, 1]*Cos[1]*Sin[1]),
            d_N[2, 1] - 3./4.*(d_P[2, 1]*Sin[2]**2 + d_P[1, 1]*Sin[1]**2)
        ])
        
        # Assemblage B_f (vectorisé)
        B_f = jnp.stack([
            inv_J[0, 0]*B_x_xi + inv_J[0, 1]*B_x_eta,
            inv_J[1, 0]*B_y_xi + inv_J[1, 1]*B_y_eta,
            inv_J[0, 0]*B_y_xi + inv_J[0, 1]*B_y_eta + inv_J[1, 0]*B_x_xi + inv_J[1, 1]*B_x_eta
        ])
        
        return B_f
    
    @partial(jit, static_argnums=(0,))
    def compute_strain_and_stress(self, nodes: jnp.ndarray, 
                                  U_elm: jnp.ndarray, material, element_property
                                  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Calcule les déformations et contraintes aux points de Gauss (version JIT)
        
        Args:
        nodes: (3, 3) coordonnées des nœuds
        U_elm: (18,) vecteur des déplacements élémentaires
        z: position dans l'épaisseur (si None, utilise h/2)
        
        Returns:
        strain: dict des déformations aux points de Gauss
        stress: dict des contraintes aux points de Gauss
        """

        
        E = material[0]
        nu = material[1]
        
        h = element_property[0]
        z = h/2
        
        # Matrice constitutive
        C = E / (1.0 - nu**2) * jnp.array([[1., nu, 0.],
                                            [nu, 1., 0.],
                                            [0., 0., 0.5*(1. - nu)]])
        
        # Appel de la fonction statique JIT
        return self._compute_strain_stress_static(
            nodes, U_elm, z, C,
            self.int_nodes, self.int_weights
        )
    
    @staticmethod
    @jit
    def _compute_strain_stress_static(nodes, U_elm, z, C, int_nodes, int_weights):
        """Calcul JIT des déformations et contraintes"""
        
        # Repère local
        u = nodes[1, :] - nodes[0, :]
        v_temp = nodes[2, :] - nodes[0, :]
        n = jnp.cross(u, v_temp)
        v = jnp.cross(n, u)
        
        u = u / jnp.linalg.norm(u)
        v = v / jnp.linalg.norm(v)
        n = n / jnp.linalg.norm(n)
        
        r_glob_to_loc = jnp.stack([u, v, n])
        coord_loc = jnp.dot(r_glob_to_loc, nodes.T).T
        
        # Géométrie
        diff = jnp.array([
            coord_loc[1, :2] - coord_loc[0, :2],
            coord_loc[2, :2] - coord_loc[1, :2],
            coord_loc[0, :2] - coord_loc[2, :2]
        ])
        lengths = jnp.linalg.norm(diff, axis=1)
        Cos = diff[:, 0] / lengths
        Sin = diff[:, 1] / lengths
        
        # Rotation de U_elm
        R = jnp.zeros((18, 18))
        for i in range(6):
            R = R.at[i*3:i*3+3, i*3:i*3+3].set(r_glob_to_loc)
        
        U_elm_loc = jnp.dot(R, U_elm)
        
        # Transformation beta_x = theta_y, beta_y = -theta_x
        U_elm_loc_transformed = U_elm_loc
        for i in range(3):
            theta_x = U_elm_loc[6*i + 3]
            theta_y = U_elm_loc[6*i + 4]
            U_elm_loc_transformed = U_elm_loc_transformed.at[6*i + 3].set(theta_y)
            U_elm_loc_transformed = U_elm_loc_transformed.at[6*i + 4].set(-theta_x)
        
        # Calcul aux points de Gauss (utilisation de listes pour compatibilité avec return)
        strains_list = []
        stresses_list = []
        coords_list = []
        
        for k in range(3):
            xi, eta = int_nodes[k]
            
            # Dérivées des fonctions de forme
            d_N = jnp.array([
                [-1.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [4.0*(1.0 - 2.0*xi - eta), -4.0*xi],
                [4.0*eta, 4.0*xi],
                [-4.0*eta, 4.0*(1.0 - xi - 2.0*eta)]
            ])
            
            # Jacobienne
            J = jnp.array([
                [jnp.dot(d_N[:3, 0], coord_loc[:, 0]), jnp.dot(d_N[:3, 0], coord_loc[:, 1])],
                [jnp.dot(d_N[:3, 1], coord_loc[:, 0]), jnp.dot(d_N[:3, 1], coord_loc[:, 1])]
            ])
            
            det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            inv_J = jnp.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det_J
            
            # B_m
            dN_dx = inv_J[0, 0]*d_N[:3, 0] + inv_J[0, 1]*d_N[:3, 1]
            dN_dy = inv_J[1, 0]*d_N[:3, 0] + inv_J[1, 1]*d_N[:3, 1]
            
            B_m = jnp.zeros((3, 6))
            B_m = B_m.at[0, 0::2].set(dN_dx)
            B_m = B_m.at[1, 1::2].set(dN_dy)
            B_m = B_m.at[2, 0::2].set(dN_dy)
            B_m = B_m.at[2, 1::2].set(dN_dx)
            
            U_uv = jnp.array([U_elm_loc_transformed[0], U_elm_loc_transformed[1],
                             U_elm_loc_transformed[6], U_elm_loc_transformed[7],
                             U_elm_loc_transformed[12], U_elm_loc_transformed[13]])
            strain_m = jnp.dot(B_m, U_uv)
            
            # B_f
            d_P = d_N[3:, :]
            B_f = DKT_element._compute_B_f_static(inv_J, d_N, d_P, lengths, Cos, Sin)
            
            U_w_beta = jnp.array([U_elm_loc_transformed[2], U_elm_loc_transformed[3], 
                                 U_elm_loc_transformed[4], U_elm_loc_transformed[8],
                                 U_elm_loc_transformed[9], U_elm_loc_transformed[10],
                                 U_elm_loc_transformed[14], U_elm_loc_transformed[15],
                                 U_elm_loc_transformed[16]])
            strain_b = jnp.dot(B_f, U_w_beta)
            
            # Déformations totales
            strain_tot = strain_m + z * strain_b
            
            # Contraintes
            stress_m = jnp.dot(C, strain_m)
            stress_b = z * jnp.dot(C, strain_b)
            stress_tot = stress_m + stress_b
            
            # Coordonnées du point de Gauss
            N = jnp.array([1.0 - xi - eta, xi, eta])
            x = jnp.dot(N, coord_loc[:, 0])
            y = jnp.dot(N, coord_loc[:, 1])
            gauss_coord = jnp.array([x, y, jnp.mean(coord_loc[:, 2])])
            gauss_coord_global = jnp.dot(r_glob_to_loc.T, gauss_coord)
            
            # Stocker les résultats
            strains_list.append(jnp.concatenate([strain_m, strain_b, strain_tot]))
            stresses_list.append(jnp.concatenate([stress_m, stress_b, stress_tot]))
            coords_list.append(gauss_coord_global)
        
        # Convertir en arrays
        strains_array = jnp.stack(strains_list)
        stresses_array = jnp.stack(stresses_list)
        coords_array = jnp.stack(coords_list)
        
        # Note: On retourne des arrays au lieu de dicts pour compatibilité JIT
        # L'utilisateur devra reconstruire les dicts si nécessaire
        # strains_array shape: (3, 9) => [strain_m(3), strain_b(3), strain_tot(3)], strain_m(3) = [epsilon_x, epsilon_y, gamma_xy]

        return strains_array, stresses_array, coords_array


