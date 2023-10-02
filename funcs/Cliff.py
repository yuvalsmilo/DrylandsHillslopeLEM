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

            'landslide__deposition': {
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
                 slope_of_softer_layer = 30,
                 cliff_fracture_slope = 80,
                 **kwargs):
        super(Cliff, self).__init__(grid)

        self.weathering_rate = weathering_rate
        self.clast_density = clast_density
        self.dx = self._grid.dx
        self.cliff_br_ks = cliff_br_ks
        self._HIGH_KS = 100  # Large number preventing runoff generation over the cliff top
        self.neg_diffusivity = 10**-20 # VERY small diffusivity for bedrock
        self._min_soil_cover = 0.01
        self.sediment_ks = sediment_ks
        self.lower_layer_ks = lower_layer_ks
        self.lower_layer_ke = lower_layer_ke
        self.cliff_br_ke = cliff_br_ke
        self.cliff_fracture_slope = cliff_fracture_slope
        self.cliff_fracture_slope_dzdx = np.tan(np.deg2rad(cliff_fracture_slope))
        self.slope_softer_layer =  slope_of_softer_layer
        self.slope_softer_layer_dzdx = np.tan(np.deg2rad(slope_of_softer_layer))
        self.cell_area = self._grid.dx ** 2
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

        grid.add_zeros("landslide_sediment_point_source", at="node")
        grid.add_zeros('landslide__deposition', at="node")
        grid.add_zeros('cliff_nodes', at="node")
        grid.add_zeros('erodibility__coefficient', at='node')
        grid.add_ones("hydraulic_conductivity", at="node")
        grid.add_zeros('linear_diffusivity', at="node")

    def update_diffusive_mass(self, flux):

        soil = self._grid.at_node['soil__depth']
        grain_weight_node = self._grid.at_node['grain__weight']
        sigma  = self.clast_density
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
                                    where=g_total_link.reshape(-1, 1) != 0)


        self._sed_flux_at_link_class = np.multiply(flux.reshape([-1, 1]),
                                                   g_fraction_link)

        sed_flux_at_link[:] = np.sum(self._sed_flux_at_link_class , axis=1)
        dzdt_temp = -self._grid.calc_flux_div_at_node(sed_flux_at_link)


        correct_flux_nodes = np.where(np.round((soil + dzdt_temp), 4) < 0)[0]
        while correct_flux_nodes.size > 0:
            # THIS LOOP IS NEEDED TO AVOID DELIVERING MORE SEDIMENT THAN EXISTS
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
            self._sum_dzdt += dzdt  # sum the dzdt over all size fractions
            grain_weight_node[:, size_class] += (dzdt * max(self._grid.cell_area_at_node)) * self.clast_density # in kg


        grain_weight_node[grain_weight_node < 0] = 0  # For saftey
        sum_grain_weight_node = np.sum(grain_weight_node, 1)
        soil[self._grid.core_nodes] = sum_grain_weight_node[self._grid.core_nodes] / (max(self._grid.cell_area_at_node) * self.clast_density)
        soil[soil < 0] = 0  # For saftey

    def weathering_run_one_step(self):

        # Pointers
        topo = self._grid.at_node["topographic__elevation"]
        bed = self._grid.at_node["bedrock__elevation"]
        steepest_slope = np.max(self._grid.at_node["hill_topographic__steepest_slope"], axis=1)
        s = self._grid.at_node["soil__depth"]
        cliff_nodes = self._grid.at_node['cliff_nodes']
        landslide_sed_in = self._grid.at_node["landslide_sediment_point_source"]
        weathering_depth = np.zeros_like(topo)
        grad_at_link = self._grid.calc_grad_at_link('topographic__elevation', out=None)

        upwind_links_at_node = self._grid.upwind_links_at_node(grad_at_link)
        upwind_node_id_at_link  = self._grid.map_value_at_max_node_to_link('water_surface__elevation', self._grid.nodes.flatten(),)

        # Reset landslide sediment point source field
        landslide_sed_in.fill(0.0)

        weathering_depth[self._grid.core_nodes] = self.weathering_rate *  steepest_slope[self._grid.core_nodes]  # Horizontal rate to vertical
        weathering_depth *= np.exp((-s[:]/self._critical_soil_depth)) # Adjust accoring to debris cover
        weathering_depth[cliff_nodes.flatten()==False] = 0 # Weathering apply only for caprock
        bed[self._grid.core_nodes] = bed[self._grid.core_nodes] - weathering_depth[self._grid.core_nodes]
        landslide_sed_in[self._grid.core_nodes] = weathering_depth[self._grid.core_nodes]
        return


    def failure_run_one_step(self,):

        litho_contact = self._grid.at_node['lithological_contact_height']
        ct = self.cliff_top
        cb = self.cliff_base
        topo = self._grid.at_node['topographic__elevation']
        bedrock_topo = self._grid.at_node['bedrock__elevation']
        soil_depth = self._grid.at_node['soil__depth']
        landslide_sed_in = self._grid.at_node["landslide_sediment_point_source"]
        cliff_nodes = self._grid.at_node['cliff_nodes']
        d8_recivers = self._grid.at_node["hill_flow__receiver_node"]
        nodes = self._grid.nodes.flatten()

        cliffbase_nodes = np.where(cb.flatten() == True)[0]
        cliffbase_nodes = cliffbase_nodes[~self._grid.node_is_boundary(cliffbase_nodes)]

        new_bedrock_topo = np.copy(bedrock_topo)
        recivers_topo = topo[d8_recivers]
        recivers_topo[d8_recivers < 0] = np.inf
        min_reciver_topo = np.nanmin(recivers_topo, 1)

        indices = np.where((cliff_nodes.flatten()[nodes] == True) & (
                     min_reciver_topo[nodes] + self._critical_height < litho_contact[nodes]))

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
                new_bedrock_topo[slope_node] = new_upslope_elev
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

                new_cliff_elev = (self.cliff_fracture_slope_dzdx * distance_to_up_node) + new_upslope_elev
                new_bedrock_topo[new_cliff_elev<new_bedrock_topo] = new_cliff_elev[new_cliff_elev<new_bedrock_topo]

        landslide_sed_in[:] = bedrock_topo - new_bedrock_topo
        landslide_sed_in[landslide_sed_in < 0] = 0
        bedrock_topo[:] = new_bedrock_topo
        topo[:] = new_bedrock_topo + soil_depth
        return

    def update_deposited_mass(self, ):

        sigma = self.clast_density
        deposition = self._grid.at_node['landslide__deposition']
        deposition_indices = np.where(deposition > 0)[0]
        grain_weight = self._grid.at_node['grain__weight']

        if np.any(deposition_indices):
            deposition_mass = (
                                      deposition[deposition_indices] * max(
                                  self._grid.cell_area_at_node))\
                              * sigma  # in Kg

            for (index, mass) in zip(deposition_indices, deposition_mass):
                # Ratio of every node added mass relative to the initial mass
                add_mass_vec_ratio = mass / np.sum(self._grid.g1.g_state_slide)  # self._grid.g1.g_state_slide
                # Making the matrix of add mass per class size (row) and per node (column)
                grain_weight[index,
                :] += add_mass_vec_ratio * self._grid.g1.g_state_slide  # self._grid.g1.g_state_slide

        deposition[:] = 0
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

        ks_effective_above = self.sediment_ks * (1 - np.exp(-3 * soil[above_contant])) + self.cliff_br_ks
        hydraulic_conductivity[above_contant] = ks_effective_above

        ks_effective_below = self.sediment_ks * (1 - np.exp(-3 * soil[below_contant])) + self.lower_layer_ks
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

            # Sediment run_out algorithm is based on Hyland component
            topo = self._grid.at_node["topographic__elevation"]
            bed = self._grid.at_node["bedrock__elevation"]
            soil_d = self._grid.at_node["soil__depth"]
            landslide_depo = self._grid.at_node["landslide__deposition"]
            dh_hill = self._grid.at_node["landslide_sediment_point_source"]
            threshold_slope_dzdx = self._threshold_slope_dzdx
            node_flatten = self._grid.nodes.flatten()

            if np.any(dh_hill >0):

                slope = np.max(self._grid.at_node["hill_topographic__steepest_slope"], axis=1)
                slope[slope < 0] = 0.0
                topo_copy = np.array(topo.copy())
                length_adjacent_cells = np.array([self._grid.dx, self._grid.dx, self._grid.dx, self._grid.dx,
                                                  self._grid.dx * np.sqrt(2), self._grid.dx * np.sqrt(2),
                                                  self._grid.dx * np.sqrt(2), self._grid.dx * np.sqrt(2)])

                topo_runout = topo[node_flatten]
                sorted_indices_in_topo_runout = np.flip(np.argsort(topo_runout,0))
                stack_rev_sel= node_flatten[sorted_indices_in_topo_runout]
                stack_rev_sel = np.delete(stack_rev_sel,self._grid.node_is_boundary(stack_rev_sel))
                L_Hill = np.where(
                    slope < threshold_slope_dzdx,
                    self._grid.dx / (1 - (slope / threshold_slope_dzdx) ** 2),
                    1e6,
                )
                for i,donor in enumerate(stack_rev_sel):

                    if dh_hill[donor] == 0:
                        continue

                    dH = max(
                        0,(dh_hill[donor] / self.dx) / L_Hill[donor])
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
                    neighbors = neighbors[neighbors>0]
                    neibhors_elev = topo_copy[neighbors]

                    upstream_neibhors = neighbors[neibhors_elev>=donor_elev]
                    downstream_neibhors = neighbors[neibhors_elev<donor_elev]

                    upstream_neibhors_elev = topo_copy[upstream_neibhors]
                    downstream_neibhors_elev = topo_copy[downstream_neibhors]


                    if np.size(upstream_neibhors_elev) == 0:
                        upstream_neibhors_elev = donor_elev

                    if np.size(downstream_neibhors_elev) == 0:
                        downstream_neibhors_elev = donor_elev


                    elev_diff_downstream = donor_elev - downstream_neibhors_elev
                    max_diff_downstream = np.max((np.min(threshold_slope_dzdx - elev_diff_downstream),0)) # zero in case above criticala ngel

                    elev_diff_upstream = upstream_neibhors_elev - donor_elev
                    max_diff_upstream = np.max((elev_diff_upstream))/2

                    depo = np.min((dH,max_diff_upstream))
                    if depo >0:
                        actual_deposition = np.min((dh_hill[donor],depo))
                    else:
                        actual_deposition = 0

                    downstream_elev_diffs = topo_copy[donor] -  downstream_neibhors_elev
                    sum_diff = np.sum(downstream_elev_diffs)
                    if sum_diff == 0:
                        proportions = np.zeros_like(downstream_elev_diffs)
                    else:
                        proportions = np.divide(downstream_elev_diffs, sum_diff)

                    dz_out = dh_hill[donor] - actual_deposition
                    dh_hill[donor] = actual_deposition
                    topo_copy[donor] += actual_deposition
                    dh_hill[downstream_neibhors] += dz_out * proportions

                soil_d[self._grid.core_nodes] += dh_hill[self._grid.core_nodes]
                topo[self._grid.core_nodes]  = bed[self._grid.core_nodes]+soil_d[self._grid.core_nodes]
                landslide_depo[self._grid.core_nodes] = dh_hill[self._grid.core_nodes]
                dh_hill[:] = 0





