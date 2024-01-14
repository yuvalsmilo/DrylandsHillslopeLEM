# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:35:37 2022

@author: yuvalshm
"""
import matplotlib.pyplot as plt
import numpy as np
import copy
from landlab import RasterModelGrid
from landlab.grid.nodestatus import NodeStatus
import numpy as np
from landlab import Component
import time
from landlab.grid.nodestatus import NodeStatus


class Cliff(Component):

    _name = "Cliff"
    _unit_agnostic = True

    _info = {
        "soil__depth":{
          "dtype":float,
          "intent":"inout",
          "optional":False,
          "units": "m",
          "mapping":"node",
          "doc":"Soil depth at node",
      },
        'lithological_contact_height':{
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "The height of lithological contact at node",
        },
        "landslide_sediment_point_source": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Source of sediment",
        },
        "landslide_soilsediment_point_source": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Source of sediment",
        },
        "landslide_bedrocksediment_point_source": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Source of sediment",
        },
            'landslide__deposition': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Landslide deposition at node",
        },
        'landslide__deposition_soil': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Landslide deposition at node",
        },
        'landslide__deposition_bedrock': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Landslide deposition at node",
        },

        'cliff_nodes': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Boolean of cliff nodes",
        },
        "hydraulic_conductivity": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Ks values",
        },
        "erodibility__coefficient": {
              "dtype": float,
              "intent": "out",
              "optional": False,
              "units": "m",
              "mapping": "node",
              "doc": "Erodibility values",
          },
        'linear_diffusivity': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Diffusivity values",
        },
    }


    def __init__(self,
                 grid,
                 weathering_rate = 0.01,
                 cliff_br_ks = 10 ** -10,
                 sediment_ks = 10 ** -6,
                 lower_layer_ks = 10 ** -6.5,
                 cliff_br_ke = 10**-10,
                 lower_layer_ke=10 ** -5,
                 threshold_slope = 33,
                 critical_soil_depth = 0.5,
                 sediment_diffusivity = 0.05,
                 critical_height = 5,
                 clast_density = 2000,
                 slope_of_softer_layer = 15,
                 cliff_fracture_slope = 80,
                 phi = 0,
                 min_deposition_slope=0.001,
                 max_deposition_slope=2,
                 bedrocklandslide_grainsize_fractures = "None",
                 soillandslide_grainsize_fractures= "None",
                 **kwargs):

        super(Cliff, self).__init__(grid)

        self.weathering_rate = weathering_rate
        self.clast_density = clast_density
        self.dx = self._grid.dx
        self.cliff_br_ks = cliff_br_ks
        self._HIGH_KS = 100  # Large number preventing runoff generation over the cliff top
        self.neg_diffusivity = 10**-20 # VERY small diffusivity for bedrock
        self._min_soil_cover = 0.01
        self._min_soil_cover_weight = self._min_soil_cover * self._grid.dx**2 * self.clast_density
        self.sediment_ks = sediment_ks
        self.lower_layer_ks = lower_layer_ks
        self.lower_layer_ke = lower_layer_ke
        self.cliff_br_ke = cliff_br_ke
        self.cliff_fracture_slope = cliff_fracture_slope
        self.cliff_fracture_slope_dzdx = np.tan(np.deg2rad(cliff_fracture_slope))
        self.slope_softer_layer =  slope_of_softer_layer
        self.slope_softer_layer_dzdx = np.tan(np.deg2rad(slope_of_softer_layer))
        self.cell_area = self._grid.dx ** 2
        self._phi = phi
        self._min_deposition_slope = min_deposition_slope
        self._max_deposition_slope = max_deposition_slope
        self.rows = np.shape(self._grid.nodes)[0]
        self.cols = np.shape(self._grid.nodes)[1]
        self._threshold_slope = threshold_slope
        self._critical_height = critical_height
        self._threshold_slope_dzdx  = np.tan(np.deg2rad(self._threshold_slope))
        self._critical_soil_depth = critical_soil_depth
        self.sediment_diffusivity = sediment_diffusivity
        self._zeros_at_node = self._grid.zeros(at="node")
        self._zeros_at_link = self._grid.zeros(at="link")
        self._zeros_at_link_for_fractions = np.zeros((np.shape( self._zeros_at_link)[0],np.shape(self._grid.at_node['grain__weight'])[1]))
        self._lithological_contact_height =  grid.at_node['lithological_contact_height']
        self.cliff_state = np.ones(
            (int(np.shape(grid)[0]), int(np.shape(grid)[1])), dtype=bool) * False
        self.cliff_top = np.ones(
            (int(np.shape(grid)[0]), int(np.shape(grid)[1])), dtype=bool) * False
        self.cliff_base = np.ones(
            (int(np.shape(grid)[0]), int(np.shape(grid)[1])), dtype=bool) * False

        if type(bedrocklandslide_grainsize_fractures) is str:
            self._bedrockllandslide_grainsize_fractures = np.zeros(np.shape(self._grid.at_node['grain__weight'])[1]) + np.shape(self._grid.at_node['grain__weight'])[1] / \
                                                          (np.shape(self._grid.at_node['grain__weight'])[1]*30) * 10e2
        else:
            self._bedrockllandslide_grainsize_fractures = bedrocklandslide_grainsize_fractures
        if type(soillandslide_grainsize_fractures) is str:
            self._soillandslide_grainsize_fractures = np.copy(self._bedrockllandslide_grainsize_fractures)
        else:
            self._soillandslide_grainsize_fractures = soillandslide_grainsize_fractures
        self.initialize_output_fields()

    def update_diffusive_mass(self, flux):

        soil = self._grid.at_node['soil__depth']
        grain_weight_node = self._grid.at_node['grain__weight']
        sigma  = self.clast_density
        slope = np.max(self._grid.at_node["hill_topographic__steepest_slope"], axis=1)
        landslide_sed_in = self._grid.at_node["landslide_soilsediment_point_source"]
        landslide_sed_in.fill(0.0)
        g_total_dt_node = np.sum(grain_weight_node, 1).copy()  # Total grain size mass at node
        g_total_link = self._zeros_at_link.copy()  # Total grain size mass at link
        g_state_link = self._zeros_at_link_for_fractions.copy()  # Grain size mass for each size fraction
        sed_flux_at_link = self._grid.at_link['sediment_flux']
        self._sum_dzdt = self._zeros_at_node.copy()

        upwind_node_id_at_link  = self._grid.map_value_at_max_node_to_link('topographic__elevation', self._grid.nodes.flatten(),)
        nonzero_upwind_node_ids = np.uint(upwind_node_id_at_link[np.nonzero(upwind_node_id_at_link)])
        nonzero_downind_link_ids = np.nonzero(upwind_node_id_at_link)[0]


        g_total_link[nonzero_downind_link_ids] = g_total_dt_node[
            nonzero_upwind_node_ids]  # Total sediment mass for of each up-wind node mapped to link.
        g_state_link[nonzero_downind_link_ids, :] = grain_weight_node[nonzero_upwind_node_ids,
                                                    :]  # Fraction of sediment mass of each upwind node, for all size-fraction, mapped to link
        g_fraction_link = np.divide(g_state_link,
                                    g_total_link.reshape(-1, 1),
                                    out=np.zeros_like(g_state_link),
                                    where=g_total_link.reshape(-1, 1) > self._min_soil_cover_weight)


        self._sed_flux_at_link_class = np.multiply(flux.reshape([-1, 1]),
                                                   g_fraction_link)

        sed_flux_at_link[:] = np.sum(self._sed_flux_at_link_class , axis=1)
        dzdt_temp = -self._grid.calc_flux_div_at_node(sed_flux_at_link)

        # THIS LOOP IS NEEDED TO AVOID DELIVERING MORE SEDIMENT THAN EXISTS
        correct_flux_nodes = np.where(np.round((soil + dzdt_temp), 4) < 0)[0]
        while correct_flux_nodes.size > 0:
            all_links_in_node = self._grid.links_at_node[correct_flux_nodes]
            fluxes = self._grid.link_dirs_at_node[correct_flux_nodes] * sed_flux_at_link[all_links_in_node]

            sum_fluxes = np.sum(fluxes, 1)
            soil_depth_in_correct_nodes = soil[correct_flux_nodes]

            fluxes_in_vec = np.zeros_like(fluxes)
            fluxes_out_vec = np.zeros_like(fluxes)

            fluxes_in_vec[fluxes > 0] = fluxes[fluxes > 0]
            fluxes_out_vec[fluxes < 0] = fluxes[fluxes < 0]

            sum_fluxes_in = np.sum(np.abs(fluxes_in_vec), 1)
            sum_fluxes_out = np.sum(np.abs(fluxes_out_vec), 1)

            ratio = np.divide(
                soil_depth_in_correct_nodes + sum_fluxes_in,
                sum_fluxes_out
            )
            for i in range(np.shape(all_links_in_node)[0]):
                links = all_links_in_node[i, :]
                links_out = links[fluxes[i, :] < 0]
                self._sed_flux_at_link_class[links_out, :] *= np.abs(ratio[i])

            sed_flux_at_link[:] = np.sum(self._sed_flux_at_link_class , axis=1)
            dzdt_temp = -self._grid.calc_flux_div_at_node(sed_flux_at_link)
            correct_flux_nodes = np.where(np.round((soil + dzdt_temp), 4) < 0)[0]

        for size_class in range(np.shape(self._sed_flux_at_link_class)[1]):
            dzdt = -self._grid.calc_flux_div_at_node(self._sed_flux_at_link_class[:, size_class])
            ##
            self._sum_dzdt += dzdt  # sum the dzdt over all size fractions
            grain_weight_node[:, size_class] += (dzdt * self._grid.dx**2) * self.clast_density * (1-self._phi) # in kg


        grain_weight_node[grain_weight_node < 0] = 0  # For saftey
        sum_grain_weight_node = np.sum(grain_weight_node, 1)
        soil[self._grid.core_nodes] = sum_grain_weight_node[self._grid.core_nodes] / (self._grid.dx ** 2 * self.clast_density * (1 - self._phi))
        soil[soil < 0] = 0  # For saftey
        self._sed_flux_at_link_class[:] = 0
    def weathering_run_one_step(self):

        # Pointers
        topo = self._grid.at_node["topographic__elevation"]
        bed = self._grid.at_node["bedrock__elevation"]
        steepest_slope = np.max(self._grid.at_node["hill_topographic__steepest_slope"], axis=1)
        s = self._grid.at_node["soil__depth"]
        cliff_nodes = self._grid.at_node['cliff_nodes']
        landslide_sed_in = self._grid.at_node["landslide_bedrocksediment_point_source"]
        landslide_soilsed_in = self._grid.at_node["landslide_soilsediment_point_source"]
        grain_fractions = self.grid.at_node['grain__weight']
        weathering_depth = np.zeros_like(topo)
        grad_at_link = self._grid.calc_grad_at_link('topographic__elevation', out=None)
        slope_of_downwind_link  = self._grid.map_downwind_node_link_max_to_node(grad_at_link)
        soil_removal = self._grid.at_node["landslide__deposition_soil"]
        # Reset landslide sediment point source field
        landslide_sed_in.fill(0.0)
        landslide_soilsed_in.fill(0.0)
        soil_removal.fill(0.0)


        weathering_depth[self._grid.core_nodes] = self.weathering_rate *  steepest_slope[self._grid.core_nodes]  # Horizontal rate to vertical
        weathering_depth *= np.exp((-s[:]/self._critical_soil_depth)) # Adjust accoring to debris cover
        weathering_depth[cliff_nodes.flatten()==False] = 0 # Weathering apply only for caprock

        bed[self._grid.core_nodes] = bed[self._grid.core_nodes] - weathering_depth[self._grid.core_nodes]
        landslide_sed_in[self._grid.core_nodes] = weathering_depth[self._grid.core_nodes] * self._grid.dx**2 #

        return


    def failure_run_one_step(self,):

        litho_contact = self._grid.at_node['lithological_contact_height']
        ct = self.cliff_top
        cb = self.cliff_base
        topo = self._grid.at_node['topographic__elevation']
        bedrock_topo = self._grid.at_node['bedrock__elevation']
        soil_depth = self._grid.at_node['soil__depth']
        landslide_sed_in_bedrock = self._grid.at_node["landslide_bedrocksediment_point_source"]
        landslide_sed_in_bedrock.fill(0.)
        landslide_sed_in_soil = self._grid.at_node["landslide_soilsediment_point_source"]
        landslide_sed_in_soil.fill(0)
        soil_removal = self._grid.at_node["landslide__deposition_soil"]
        soil_removal.fill(0.)
        cliff_nodes = self._grid.at_node['cliff_nodes']
        d8_recivers = self._grid.at_node["hill_flow__receiver_node"]
        nodes = self._grid.nodes.flatten()
        grain_fractions = self.grid.at_node['grain__weight']

        cliffbase_nodes = np.where(cb.flatten() == True)[0]
        cliffbase_nodes = cliffbase_nodes[~self._grid.node_is_boundary(cliffbase_nodes)]
        before_topo = np.copy(topo)
        new_topo = np.copy(topo)
        before_soil_copy = np.copy(soil_depth)

        recivers_topo = topo[d8_recivers]
        recivers_topo[d8_recivers < 0] = np.inf
        min_reciver_topo = np.nanmin(recivers_topo, 1)

        indices = np.where((cliff_nodes.flatten()[nodes] == True) & (
            (min_reciver_topo[nodes] + self._critical_height) < litho_contact[nodes]))

        if np.size(indices)>0:
            up_nodes = nodes[indices]
            up_nodes = up_nodes[~self._grid.node_is_boundary(up_nodes)]
        else:
            up_nodes =[]

        for slope_node in up_nodes:
            dz_rockfall = 0
            neighbors = np.concatenate(
                (
                    self._grid.active_adjacent_nodes_at_node[
                        slope_node
                    ],
                    self._grid.diagonal_adjacent_nodes_at_node[
                        slope_node
                    ],
                )
            )

            below_cliff_nodes = neighbors[((topo[neighbors] + self._critical_height) < litho_contact[slope_node])]
            neighbors = neighbors[neighbors != -1]

            x_slope = self._grid.node_x[slope_node]
            y_slope = self._grid.node_y[slope_node]

            for below_node in below_cliff_nodes:

                slopenode_elv = topo[below_node]
                distance_to_crit_node = np.sqrt(
                    np.add(
                        np.square(
                            x_slope - self._grid.node_x[below_node]
                        ),
                        np.square(
                            y_slope - self._grid.node_y[below_node]
                        ),
                    )
                )

                new_upslope_elev = slopenode_elv + \
                                   self.slope_softer_layer_dzdx * distance_to_crit_node
                new_topo[slope_node] = new_upslope_elev
                x_cliff = self._grid.node_x[slope_node]
                y_cliff =  self._grid.node_y[slope_node]

                distance_to_up_node = np.sqrt(
                    np.add(
                        np.square(
                            x_cliff - self._grid.node_x
                        ),
                        np.square(
                            y_cliff - self._grid.node_y
                        ),
                    )
                )

                new_elev = (self.cliff_fracture_slope_dzdx * distance_to_up_node) + new_upslope_elev
                new_topo[new_elev<new_topo] = new_elev[new_elev<new_topo]

        topo_diff =   before_topo[:] -  new_topo[:]
        indices = np.argwhere((topo_diff > 0))
        if np.size(indices)>0:
            indices = np.delete(indices,
                                self._grid.node_is_boundary(indices).flatten())

        soil_erosion_depth = np.min((before_soil_copy[indices], topo_diff[indices]), axis=0)
        store_volume_sed = soil_erosion_depth * (1 - self._phi) * (self.grid.dx ** 2)
        store_volume_bed = (topo_diff[indices] - soil_erosion_depth) * (self.grid.dx ** 2)

        landslide_sed_in_soil[indices] = store_volume_sed
        landslide_sed_in_bedrock[indices] = store_volume_bed
        soil_removal[indices] -= store_volume_sed / (1-self._phi) /  (self.grid.dx ** 2)
        soil_depth[indices] -= soil_erosion_depth
        topo[:] = new_topo[:]
        bedrock_topo[:] =  topo[:] - soil_depth[:]

        a = np.sum(grain_fractions[indices, :], axis=0)
        b = np.sum(np.sum(grain_fractions[indices, :], axis=0))
        grain_fractions_failure = np.divide(a, b, where=b != 0)
        self._soillandslide_grainsize_fractures = b * grain_fractions_failure.flatten()
        self.update_mass()
        return

    def update_mass(self, ):

        sigma = self.clast_density
        grain_weight = self._grid.at_node['grain__weight']
        landslide_depo_soil = self._grid.at_node["landslide__deposition_soil"]
        landslide_depo_bedrock = self._grid.at_node["landslide__deposition_bedrock"]

        deposition_indices_bedrock = np.where(landslide_depo_bedrock != 0)[0]
        deposition_indices_soil = np.where(landslide_depo_soil != 0)[0]
        landslide_grain_sizes = self._soillandslide_grainsize_fractures

        if np.any(landslide_depo_bedrock):
            deposition_mass = (
                                      landslide_depo_bedrock[deposition_indices_bedrock] * self._grid.dx**2)\
                              * sigma * (1-self._phi) # in Kg

            for (index, mass) in zip(deposition_indices_bedrock, deposition_mass):
                # Ratio of every node added mass relative to the initial mass
                add_mass_vec_ratio = mass / np.sum(self._bedrockllandslide_grainsize_fractures)  # self._grid.g1.g_state_slide
                # Making the matrix of add mass per class size (row) and per node (column)
                grain_weight[index,
                :] += add_mass_vec_ratio * self._bedrockllandslide_grainsize_fractures  # self._grid.g1.g_state_slide

        if np.any(landslide_depo_soil):
            deposition_mass = (
                                      landslide_depo_soil[deposition_indices_soil] * self._grid.dx ** 2) \
                              * sigma * (1 - self._phi)  # in Kg

            for (index, mass) in zip(deposition_indices_soil, deposition_mass):

                if mass >0:
                    add_mass_vec_ratio = mass / np.sum(self._soillandslide_grainsize_fractures)

                # Making the matrix of add mass per class size (row) and per node (column)
                    grain_weight[index,
                    :] += add_mass_vec_ratio * self._soillandslide_grainsize_fractures  # self._grid.g1.g_state_slide
                else:
                    in_fractions = np.argwhere((grain_weight[index,:]>0)).tolist()
                    mass_by_fraction = mass * (grain_weight[index,in_fractions] / np.sum(grain_weight[index, in_fractions]))
                    grain_weight[index,
                    in_fractions] += mass_by_fraction

        landslide_depo_soil[:] = 0
        landslide_depo_bedrock[:] = 0
        self.update_cliff_state()

    def update_cliff_state(self, ):

        litho = self._grid.at_node['lithological_contact_height']
        bedrock_topo = self._grid.at_node['bedrock__elevation']
        soil = self._grid.at_node['soil__depth']
        topo = self._grid.at_node['topographic__elevation']
        hydraulic_conductivity = self._grid.at_node['hydraulic_conductivity']
        bedrock_topo2d = self._grid.node_vector_to_raster(bedrock_topo)
        linear_diffusive_coef = self._grid.at_node['linear_diffusivity']
        cliff_nodes = self._grid.at_node['cliff_nodes']
        ke_vals = self._grid.at_node['erodibility__coefficient']
        above_contact_logic = bedrock_topo >=   (litho + self.slope_softer_layer_dzdx/2)
        above_contant = np.where(above_contact_logic == True)[0].tolist()
        below_contant = np.where(above_contact_logic == False)[0].tolist()
        self.cliff_state[:] = self._grid.node_vector_to_raster(above_contact_logic)
        self.cliff_base[:]=False
        self.cliff_top[:] = False

        ks_effective_above = self.sediment_ks * (1 - np.exp(-self._critical_soil_depth * soil[above_contant])) + self.cliff_br_ks
        hydraulic_conductivity[above_contant] = ks_effective_above

        ks_effective_below = self.sediment_ks * (1 - np.exp(-self._critical_soil_depth * soil[below_contant])) + self.lower_layer_ks
        hydraulic_conductivity[below_contant ] = ks_effective_below

        ## Preven runoff generation over the hogback slope
        hydraulic_conductivity2d = self._grid.node_vector_to_raster(hydraulic_conductivity)
        for col in range(self._grid.shape[1]):
            cliff_base_index = np.argwhere(self.cliff_state[:,col]==True)[0]
            cliff_top_index = np.argmax(bedrock_topo2d[:,col])
            self.cliff_top[cliff_top_index,col] = True
            hydraulic_conductivity2d[cliff_top_index:,col] = self._HIGH_KS # Prevent runoff generation on the hogback slope
            self.cliff_base[cliff_base_index,col] = True

        cliff_nodes[:] = self.cliff_state.flatten()
        hydraulic_conductivity[:] = hydraulic_conductivity2d.flatten()
        linear_diffusive_coef[np.where(soil[:] > self._min_soil_cover)] = self.sediment_diffusivity
        linear_diffusive_coef[np.where(soil[:] <= self._min_soil_cover)] =  self.neg_diffusivity

        ke_vals[:]  = self.lower_layer_ke
        ke_vals[cliff_nodes==True] = self.cliff_br_ke


    def sediment_run_out(self):

        grad_at_link = self._grid.calc_grad_at_link('topographic__elevation', out=None)
        slope_of_downwind_link  = self._grid.map_downwind_node_link_max_to_node(grad_at_link)

        topo = self._grid.at_node["topographic__elevation"]
        bed = self._grid.at_node["bedrock__elevation"]
        soil_d = self._grid.at_node["soil__depth"]
        landslide_depo_soil = self._grid.at_node["landslide__deposition_soil"]
        landslide_depo_bedrock = self._grid.at_node["landslide__deposition_bedrock"]
        landslide_depo_soil.fill(0)
        landslide_depo_bedrock.fill(0)
        grain_weight = self._grid.at_node['grain__weight']
        stack_rev = np.flip(self.grid.at_node["flow__upstream_node_order"])
        threshold_slope_dzdx = self._threshold_slope_dzdx
        node_flatten = self._grid.nodes.flatten()
        receivers = self._grid.at_node["hill_flow__receiver_node"]
        fract_receivers = self._grid.at_node["hill_flow__receiver_proportions"]
        node_status = self._grid.status_at_node
        length_adjacent_cells = np.array([self._grid.dx, self._grid.dx, self._grid.dx, self._grid.dx,
                                          self._grid.dx * np.sqrt(2), self._grid.dx * np.sqrt(2),
                                          self._grid.dx * np.sqrt(2), self._grid.dx * np.sqrt(2)])


        total_grainsize = np.expand_dims(np.sum(grain_weight, axis=1), axis=1)
        grain_size_frac = np.divide(grain_weight, total_grainsize, where=total_grainsize != 0)

        Qin_hill_soil = self._grid.at_node["landslide_soilsediment_point_source"]  # m^3 for time step
        Qin_hill_bedrock = self._grid.at_node["landslide_bedrocksediment_point_source"]  # m^3 for time step
        Qout_hill_bedrock = np.zeros_like(Qin_hill_soil)
        Qout_hill_soil = np.zeros_like(Qin_hill_soil)

        dh_hill = np.zeros_like(Qin_hill_soil)  # deposition dz
        max_D = np.zeros_like(Qin_hill_soil)

        neighbors = np.concatenate(
            (self._grid.active_adjacent_nodes_at_node[self.grid.nodes.flatten()],
             self._grid.diagonal_adjacent_nodes_at_node[self.grid.nodes.flatten()],),axis=1)

        neighbors[neighbors<0] = self.grid.nodes[np.argmin(topo[self.grid.nodes]),0]
        max_D[:] = np.max(topo[neighbors],1) - self._min_deposition_slope*length_adjacent_cells[np.argmax(topo[neighbors],1)] - topo[self.grid.nodes.flatten()]
        max_D[max_D<0] = 0

        Qout_hill = np.zeros_like(Qin_hill_soil)
        Qin_hill = Qin_hill_soil + Qin_hill_bedrock

        if np.any(Qin_hill > 0):
            slope = self._grid.at_node["hill_topographic__steepest_slope"]
            slope_min = np.min(slope,axis=1)

            slope_copy = np.copy(slope)
            slope_copy[slope_copy<=0]=np.inf
            slope_copy_min = np.min(slope_copy,axis=1)
            slope_copy_min[slope_copy_min==np.inf]=0

            slope = np.max((slope_min, slope_copy_min), axis=0)
            slope[slope <= 0] = 0

            topo_copy = np.array(topo.copy())

            stack_rev_sel = stack_rev[node_status[stack_rev] == NodeStatus.CORE]
            L_Hill = np.where(
                slope < threshold_slope_dzdx,
                self._grid.dx / (1 - (slope / threshold_slope_dzdx) ** 2),
                1e6,
            )

            for i, donor in enumerate(stack_rev_sel):

                dH = max(
                    0,
                    min(((Qin_hill[donor] / self._grid.dx) / L_Hill[donor]) / (1 - self._phi), max_D[donor])
                )

                # Make sure you dont go over a maximal downslope topoggraphic slope
                donor_elev = topo_copy[donor]

                neighbors = np.concatenate(
                    (
                        self._grid.active_adjacent_nodes_at_node[
                            donor
                        ],
                        self._grid.diagonal_adjacent_nodes_at_node[
                            donor
                        ],
                    )
                )

                neighbors = neighbors[
                    ~self._grid.node_is_boundary(neighbors)]
                neighbors = neighbors[neighbors > 0]
                neibhors_elev = topo_copy[neighbors]

                downstream_neibhors = neighbors[neibhors_elev < donor_elev]
                downstream_neibhors_elev = topo_copy[downstream_neibhors]

                if np.size(downstream_neibhors_elev) == 0:
                    if np.size(neibhors_elev[neibhors_elev >donor_elev])>0:
                        max_diff_downstream = np.min(neibhors_elev[neibhors_elev>donor_elev]) - donor_elev
                    else:
                        max_diff_downstream=0
                else:
                    elev_diff_downstream = (donor_elev - downstream_neibhors_elev)
                    max_diff_downstream_indx = np.argmax((elev_diff_downstream))
                    max_diff_downstream_node = downstream_neibhors[max_diff_downstream_indx]

                    dist_to_downstream_neibhor = np.sqrt(
                        (self.grid.node_x[max_diff_downstream_node] - self.grid.node_x[donor]) ** 2 + (
                                    self.grid.node_y[max_diff_downstream_node] - self.grid.node_y[donor]) ** 2)

                    max_diff_downstream = np.max((0,(topo_copy[max_diff_downstream_node ] + self._max_deposition_slope * dist_to_downstream_neibhor) -  topo_copy[donor ]))

                dH = np.min((dH,max_diff_downstream))
                dH_volume = (dH * self._grid.dx ** 2) * (1 - self._phi)
                Qin_ratio_soil = np.divide(Qin_hill_soil[donor], Qin_hill[donor],
                                           where=Qin_hill[donor] > 0,
                                           out = np.zeros_like(Qin_hill[donor]))
                deposited_soil_flux = np.min((Qin_ratio_soil * dH_volume, Qin_hill_soil[donor]), axis=0)
                dH_volume -= deposited_soil_flux

                Qin_hill_soil[donor] -= deposited_soil_flux
                Qout_hill_soil[donor] += Qin_hill_soil[donor]

                Qin_hill_bedrock[donor] -= dH_volume
                Qout_hill_bedrock[donor] += Qin_hill_bedrock[donor]

                landslide_depo_soil[donor] += (deposited_soil_flux / self._grid.dx ** 2) / (1 - self._phi)
                landslide_depo_bedrock[donor] += (dH_volume / self._grid.dx ** 2) / (1 - self._phi)

                Qin_hill[donor] -= (dH_volume + deposited_soil_flux)
                Qout_hill[donor] += Qin_hill[donor]

                dh_hill[donor] += dH
                topo_copy[donor] += dH

                for r in range(receivers.shape[1]):
                    rcvr = receivers[donor, r]

                    max_D_angle = topo_copy[donor] - self._min_deposition_slope * length_adjacent_cells[r] - topo_copy[
                        rcvr]
                    max_D[rcvr] = min(max(max_D[rcvr], topo_copy[donor] - topo_copy[rcvr]), max_D_angle)

                    proportion = fract_receivers[donor, r]
                    if proportion > 0. and donor != rcvr:
                        Qin_hill[rcvr] += Qout_hill[donor] * proportion
                        Qin_hill_soil[rcvr] += Qout_hill_soil[donor] * proportion
                        Qin_hill_bedrock[rcvr] += Qout_hill_bedrock[donor] * proportion

                        Qin_hill[donor] -= Qout_hill[donor] * proportion
                        Qin_hill_soil[donor] -= Qout_hill_soil[donor] * proportion
                        Qin_hill_bedrock[donor] -= Qout_hill_bedrock[donor] * proportion



            soil_d[self._grid.core_nodes] += dh_hill[self._grid.core_nodes]
            topo[self._grid.core_nodes] = bed[self._grid.core_nodes] + soil_d[self._grid.core_nodes]

            dh_hill[:] = 0
            Qin_hill[:] = 0
            Qin_hill_soil[:] = 0
            Qin_hill_bedrock[:] = 0



