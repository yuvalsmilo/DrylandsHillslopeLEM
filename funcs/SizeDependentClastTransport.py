# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 19:36:28 2022

@author: yuvalshm
"""

import numpy as np
from landlab import Component

class SizeDependentClastTransport(Component):

    _name = "MPMSelectiveTransport"
    _unit_agnostic = True

    _info = {"soil__depth":{
          "dtype":float,
          "intent":"inout",
          "optional":False,
          "units": "m",
          "mapping":"node",
          "doc":"Soil depth at node",
      },
        "shear_stress": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "Kg / (m*s^2)",
            "mapping": "link",
            "doc": "Shear stress exerted by the overland flow on bed",
        },
        "sediment_flux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "Kg / (dt*unit width)",
            "mapping": "link",
            "doc": 'Sediment flux at link',
        },
        "tau_star_c": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "link",
            "doc": "Dimensionless critical shear stress at link",
        },
        "surface_water__depth_at_link": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "link",
            "doc": "The water surface depth at link",
        },
        "water_surface__gradient": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m/m",
            "mapping": "link",
            "doc": "The gradient of surface water at link",
        },
        "grain__weight": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "Sediment weight for all size fractions",
        },
        "median__size_weight": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "The median grain size in each node based on grain size weight",
        },
        "fraction_sizes": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "The size of grain fractions",
        },
    }
    def __init__(
            self,
            grid,
            alpha = 0.045,  # 0.045 from Komar 1987
            beta  = -0.68,  # -0.68 From Komar 1987
            rho   = 1000.0,  # fluid density, kg/m3
            sigma = 2000.0,  # sediment density, kg/m3
            g     = 9.8,  # gravity constant
            critical_debris_cover = 0.1, # critical value for debris cover [m]
            no_trasported_weight_threshold = 100, # minimal weight in node which below there is no transport of material
            correct_to_debris_thick = False,

    ):

        super(SizeDependentClastTransport, self).__init__(grid)
        self._alpha = alpha
        self._beta = beta
        self._rho = rho
        self._sigma = sigma
        self._critical_debris_cover = critical_debris_cover
        self._correct_to_debris_thick_flag = correct_to_debris_thick
        self._g = g
        self._minimal_weight_in_kilograms = no_trasported_weight_threshold
        self._unitwt = self._rho * self._g
        self._nodes = np.shape(self._grid.nodes)[0] * np.shape(self._grid.nodes)[1]
        self._cell_area = max(self._grid.cell_area_at_node)
        self._zeros_at_link = self._grid.zeros(at="link")
        self._zeros_at_node = self._grid.zeros(at="node")
        self._zeros_at_link_for_fractions = np.zeros((np.shape( self._zeros_at_link)[0],np.shape(self._grid.at_node['grain__weight'])[1]))
        self._zeros_at_node_for_fractions = np.zeros((self._nodes ,np.shape(self._grid.at_node['grain__weight'])[1]))
        self._inactive_links = grid.status_at_link == grid.BC_LINK_IS_INACTIVE
        self._nodes_flatten = grid.nodes.flatten().astype('int')
        self._n_links = np.size(self._zeros_at_link)
        self._links_array  = np.arange(0,np.size(self._zeros_at_link)).tolist()
        b = self._grid.at_node['median__size_weight'].copy()
        b[b == 0] = 100000  # stupid large number for the divide by very small value.
                            # This will create a very large tau_star_c where there is no sediment
        self._tau_star_c_at_node = self._alpha * ((self._grid.at_node['fraction_sizes'] / b.reshape(-1, 1)) ** self._beta)
        self._mean_sizes_at_link = np.zeros_like(self._zeros_at_link_for_fractions)
        self._stress_star_at_link = np.zeros_like(self._zeros_at_link_for_fractions)
        self._excess_stress =  np.zeros_like(self._zeros_at_link_for_fractions)
        self._denominator_for_stress_star = np.zeros_like(self._zeros_at_link_for_fractions)
        self._sum_dzdt =  np.zeros_like(self._zeros_at_node)
        self._tau_star_c_links = np.zeros_like(self._zeros_at_link_for_fractions)
        self._sed_flux_at_link_class = np.zeros_like(self._zeros_at_link_for_fractions)
        self._tau_star_c_at_node = np.zeros_like(self._zeros_at_node_for_fractions)+1000
        self._g_total_link = np.zeros((self._n_links, 1))  # Total grain size mass at link
        self._g_state_link = np.zeros_like(self._zeros_at_link_for_fractions)  # Grain size mass for each size fraction
        self._sed_flux_star_at_link_classes = np.zeros_like(self._zeros_at_link_for_fractions)
        self._dzdt_all = self._zeros_at_node_for_fractions
        self._shear_stress_at_link_extended =  np.zeros((self._n_links, 1))
        self._median_grain_size_vec = np.zeros((self._nodes, 1))
        self._min_sizes_for_check = np.zeros_like(self._zeros_at_link_for_fractions)
        self._stress_norm_to_check = np.zeros_like(self._zeros_at_link)
        self._soil_depth_at_link = np.zeros_like(self._zeros_at_link)
        self._minslope = 0.0001

        # Create out fields
        grid.add_zeros('shear_stress', at='link')
        grid.add_zeros('sediment_flux', at='link')

    @property
    def sed_flux_size_fractions(self):
        """
        np.array with the sediment flux at LINK for all size-fractions
        """
        return self._sed_flux_at_link_class

    @property
    def sum_dzdt(self):
        """
        np.array with the total change in dz for all nodes
        """
        return self._sum_dzdt

    @property
    def tau_star_c_at_link(self):
        """
        np.array with the tau_star_criticlal at links, for all size-fraction
        """
        return self._tau_star_c_links

    @property
    def tau_star_c_at_node(self):
        """
        np.array with the tau_star_criticlal at links, for all size-fraction
        """
        return self._tau_star_c_at_node


    def calc_flux(self, dt=1):

            ## Pointers
            soil_depth = self._grid.at_node['soil__depth']
            sed_flux_at_link = self._grid.at_link['sediment_flux']
            depth_at_link = self._grid.at_link['surface_water__depth_at_link']
            wsgrad = self._grid.at_link["water_surface__gradient"]
            wsgrad[self._inactive_links] = 0
            grain_weight_node = self._grid.at_node['grain__weight']
            median_grain_size = self._grid.at_node['median__size_weight']
            median_grain_size_vec = self._median_grain_size_vec
            median_grain_size_vec[:,0] =  median_grain_size
            stress_at_link =  self._grid.at_link['shear_stress']
            fractions_sizes = self._grid.at_node['fraction_sizes']
            g_total_link = self._g_total_link  # Total grain size mass at link
            g_state_link = self._g_state_link
            sed_flux_star_at_link_classes = self._sed_flux_star_at_link_classes
            self._min_sizes_for_check[:] = self._mean_sizes_at_link[:]
            denominator_for_stress_star = self._denominator_for_stress_star
            mean_sizes_at_link = self._mean_sizes_at_link
            stress_star_at_link = self._stress_star_at_link
            excess_stress = self._excess_stress
            soil_depth_at_link = self._soil_depth_at_link

            self._sum_dzdt.fill(0.)
            self._sed_flux_at_link_class.fill(0.)
            sed_flux_at_link.fill(0.)
            stress_star_at_link.fill(0.)
            excess_stress.fill(0.)

            sum_grain_weight = np.sum(grain_weight_node,1)
            indices_nonzero_mgs = np.where(sum_grain_weight > self._minimal_weight_in_kilograms )[0]

            # Calculate shear stress and shear stress star at LINKS:  links x size fractions
            stress_at_link[:] = self._unitwt * depth_at_link * wsgrad
            self._shear_stress_at_link_extended[:,0]=  stress_at_link[:]


            # Map the upwind node to the link
            nonzero_upwind_node_ids = self._grid.map_value_at_max_node_to_link('water_surface__elevation',
                                                                              self._nodes_flatten).astype('int')
            ## Calc tau star c at node based of mapped upwind node
            self._tau_star_c_at_node[:] = np.inf
            self._tau_star_c_at_node[indices_nonzero_mgs, :] = self._alpha * ((np.divide(
                self._grid.at_node['fraction_sizes'][indices_nonzero_mgs, :],
                median_grain_size_vec[indices_nonzero_mgs])) ** self._beta
            )


            # Update the values in the min matrix for later
            self._min_sizes_for_check[self._links_array, :] = fractions_sizes[nonzero_upwind_node_ids, :]

            # Map grain weights from node to link
            g_total_link[self._links_array, 0] = sum_grain_weight[nonzero_upwind_node_ids]  # Total sediment mass for of each up-wind node mapped to link.
            g_state_link.fill(0.)
            g_state_link[self._links_array, :] = grain_weight_node[nonzero_upwind_node_ids, :]  # Fraction of sediment mass of each upwind node, for all size-fraction, mapped to link
            soil_depth_at_link[self._links_array] = soil_depth[nonzero_upwind_node_ids]

            # Find the smaller grain size in the upwind node
            self._min_sizes_for_check[g_state_link<=0] = np.inf
            min_size_class_index = np.argmin(self._min_sizes_for_check,1).tolist()
            min_size_at_link = self._min_sizes_for_check[self._links_array,min_size_class_index]

            # Map the tau_star_c from nodes to links
            self._tau_star_c_links[:] = np.inf
            self._tau_star_c_links[self._links_array, :] = self._tau_star_c_at_node[nonzero_upwind_node_ids, :]
            self._tau_star_c_links[g_state_link<=0] = np.inf

            stress_norm_to_check  = np.abs(np.divide(stress_at_link,
                      ((self._sigma - self._rho) * min_size_at_link  * self._g),
                      ))

            links_with_transport = np.where(stress_norm_to_check
                                 > self._tau_star_c_links[self._links_array, min_size_class_index]
                                 )[0].tolist()

            if np.any(links_with_transport) ==  False:
                # i.e., == Nothing to transport
                sed_flux_at_link.fill(0.)
                self._sed_flux_at_link_class.fill(0.)
                return

            # Map the mean size of all fractions from nodes to links
            mean_sizes_at_link[self._links_array, :] = fractions_sizes[nonzero_upwind_node_ids, :]

            ## Based on the topographic elevation, avoid delivring of materials if TOPOGRAPHIC slope is very low. This is even if the WATER surface slope is higher than this threshold
            gradient_of_downwind_link_at_node = self._grid.at_node['downwind__link_gradient']
            self._tau_star_c_links[gradient_of_downwind_link_at_node[nonzero_upwind_node_ids]<0.0001,:] = np.inf

            # Convert stress to stress_star at link
            denominator_for_stress_star[:] = np.inf  # large number -> high stress star where there are no sediment
            denominator_for_stress_star[links_with_transport,:] = (self._sigma - self._rho) * self._g * mean_sizes_at_link[links_with_transport,:]
            stress_star_at_link[links_with_transport,:] =np.divide(self._shear_stress_at_link_extended[links_with_transport,:],
                                                        denominator_for_stress_star[links_with_transport, :],
                                                        )

            # Excess stress at link
            excess_stress[links_with_transport,:] = np.abs(stress_star_at_link[links_with_transport,:]) - self._tau_star_c_links[links_with_transport,:]
            row, col = np.where(excess_stress > 0)
            if np.any(row) == True:

                g_fraction_link = np.divide(g_state_link, g_total_link, out=np.zeros_like(self._zeros_at_link_for_fractions), where = g_total_link > self._minimal_weight_in_kilograms)

                ## Flux calculation based on MPM equation
                sed_flux_star_at_link_classes.fill(0.)
                sed_flux_star_at_link_classes[row,col] = 8 * (excess_stress[row,col]) ** 1.5
                self._sed_flux_at_link_class[row,col] = np.multiply(
                    -np.sign(
                    stress_star_at_link[row,col]),
                    sed_flux_star_at_link_classes[row,col] * ((((( self._sigma - self._rho) / self._rho) * self._g * mean_sizes_at_link[row,col]) ** 0.5) * mean_sizes_at_link[row,col])
                )

                # Take care for flux per size fraction
                self._sed_flux_at_link_class[row,col] = np.multiply(
                    self._sed_flux_at_link_class[row,col],
                    g_fraction_link[row,col]
                )  # Multipe by the fraction of mass of all size fractions

                if self._correct_to_debris_thick_flag == True:
                    # Correct flux according to debris cover
                    self._sed_flux_at_link_class[row, col] *= (1 - np.exp((-soil_depth_at_link[row])/self._critical_debris_cover )) # Update flux according to debris cover

                sed_flux_at_link[:] = np.sum(self._sed_flux_at_link_class, axis=1)


            else:
                sed_flux_at_link.fill(0.)
                self._sed_flux_at_link_class.fill(0.)

    def calc_dt(self,flux = None):

        if flux == None:
            self.calc_flux()
        sed_flux_at_link = self._grid.at_link['sediment_flux']
        topo_gradient_downwind = self._grid.at_node['downwind__link_gradient']
        topo_gradient_upwind = self._grid.at_node['upwnwind__link_gradient']
        topo = self._grid.at_node['topographic__elevation']
        soil = self._grid.at_node['soil__depth']
        self._dzdt_all.fill(0.)
        self._sum_dzdt.fill(0.)
        grain_weight_node = self._grid.at_node['grain__weight']

        if np.any(sed_flux_at_link!=0):

            size_class = np.where(np.any(self._sed_flux_at_link_class, axis=0))[0].tolist()
            if np.size(size_class) > 1:
                result = map(self.calc_dzdt,
                             size_class,
                             np.ones_like(size_class))
                self._dzdt_all[:, size_class] = np.asarray(list(result)).T
            else:
                dzdt = np.asarray(-self._grid.calc_flux_div_at_node(self._sed_flux_at_link_class[:, size_class])).reshape((-1,1))
                self._dzdt_all[:, size_class] = dzdt[:]

            dzdt_temp = np.sum(self._dzdt_all, 1)
            self._sum_dzdt[:] = dzdt_temp[:]
            weight_all = self._dzdt_all * self._cell_area * self._sigma

            if np.any(weight_all<0):
                dt_min_mass = np.min(
                    np.abs(
                    np.divide(grain_weight_node[weight_all < 0],
                              weight_all[weight_all < 0]
                              )
                    )
                )
            else:
                dt_min_mass = np.inf

            positive_indices = np.where(dzdt_temp>self._minslope)[0].tolist()
            positive_indices = np.delete(positive_indices,self._grid.node_is_boundary(positive_indices))
            negative_indices = np.where(dzdt_temp<-self._minslope)[0].tolist()
            negative_indices  = np.delete(negative_indices ,self._grid.node_is_boundary(negative_indices ))

            if np.any(positive_indices):
                max_grad_positive = np.abs(topo_gradient_upwind[positive_indices])
                max_grad_positive /= 2
                max_grad_positive[max_grad_positive<self._minslope]= self._minslope
                positive_dts = np.min(np.abs(max_grad_positive / dzdt_temp[positive_indices]))
            else:
                positive_dts = np.inf

            if np.any(negative_indices):
                max_grad_negative = np.abs(topo_gradient_downwind[negative_indices])
                max_grad_negative /= 2
                max_grad_negative[max_grad_negative==0] = np.inf # this may happend if the waterlevel is VERY high.  TO DO: BETTER
                max_grad_negative[max_grad_negative<self._minslope]= self._minslope
                negative_dts = np.min(np.abs(max_grad_negative / dzdt_temp[negative_indices]))
            else:
                negative_dts = np.inf

            self._dt = np.min([negative_dts,positive_dts, dt_min_mass])
            if self._dt == 0:
                self._dt = np.inf
        else:
            self._dt = np.inf

    def run_one_step(self, dt=None, ):
        topo = self._grid.at_node["topographic__elevation"]
        bedrock = self._grid.at_node["bedrock__elevation"]
        soil = self._grid.at_node["soil__depth"]
        grain_weight_node = self._grid.at_node['grain__weight']

        if dt == None:
            self.calc_dt()
        else:
            self._dt = dt

        if np.any(self._sed_flux_at_link_class != 0):

            grain_weight_node[:,:] += (self._dzdt_all * self._dt)  * (self._cell_area  * self._sigma)
            grain_weight_node[grain_weight_node < 0] = 0
            sum_grain_weight_node = np.sum(grain_weight_node, 1)
            soil[self._grid.core_nodes] = sum_grain_weight_node[self._grid.core_nodes] / (self._cell_area * self._sigma)
            soil[soil < 0] = 0  # For saftey. Prevent numeric issues
            topo[:] = soil[:] + bedrock[:]


    def calc_dzdt(self,size_class, dt=1):
        dt = dt
        dzdt = -self._grid.calc_flux_div_at_node(self._sed_flux_at_link_class[:, size_class] * dt)
        return dzdt


