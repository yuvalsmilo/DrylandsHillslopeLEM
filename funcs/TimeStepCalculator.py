import numpy as np
from landlab import Component
import time
from landlab.grid.nodestatus import NodeStatus

class TimeStepCalculator(Component):
    _name = "TimeStepCalculator"
    _unit_agnostic = True

    _info = {
        "soil__depth": {
        "dtype": float,
        "intent": "in",
        "optional": False,
        "units": "m",
        "mapping": "node",
        "doc": "Soil depth at node",
        },

        'topographic__elevation': {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Topographic elevation at node",
        },

        "surface_water__depth": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Surface water depth at node",
        },

        'downwind__link_gradient': {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Gradient to downling link at node",
        },

        "water_surface__slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": 'Water surface slope at node',
        },

    }

    def __init__(
            self,
            grid,
            alpha_flow=0.7,
            alpha=0.045,  # 0.045 from Komar 1987
            beta=-0.68,  # -0.68 From Komar 1987
            rho=1000.0,  # fluid density, kg/m3
            sigma=2000.0,  # sediment density, kg/m3
            g=9.80, # gravity constant
            k_e = 10**-4, # Bedrock erodibility
            alpha_br = 1, # alpha in br
            tau_crit = 1 # tau crit


    ):
        super(TimeStepCalculator, self).__init__(grid)
        self._alpha = alpha
        self._beta = beta
        self._rho = rho
        self._sigma = sigma
        self._g = g
        self._grid = grid
        self._alpha_flow =  alpha_flow
        self._nodes = np.shape(self.grid.nodes)[0] * np.shape(self.grid.nodes)[1]
        self._zeros_at_link = self.grid.zeros(at="link")
        self._n_links = np.size(self._zeros_at_link)
        self._links_array  = np.arange(0,np.size(self._zeros_at_link)).tolist()
        self._zeros_at_link_for_fractions = np.zeros(
            (np.shape(self._zeros_at_link)[0], np.shape(self.grid.at_node['grain__weight'])[1]))
        self._zeros_at_node_for_fractions = np.zeros((self._nodes, np.shape(self.grid.at_node['grain__weight'])[1]))
        self._inactive_links = grid.status_at_link == grid.BC_LINK_IS_INACTIVE
        self._unitwt = self._rho * self._g
        self._E = self._grid.zeros(at="node")
        self.node_flatten = self._grid.nodes.flatten()
        self._k_e = k_e
        self._a = alpha_br
        self._tau_crit = tau_crit


    def calc_dt(self,
                dt_sediment = np.inf,
                dt_flow = np.inf,
                dt_bedrock_incision = np.inf):

        dt_sediment = dt_sediment
        dt_flow = dt_flow
        dt_bedrock_incision = dt_bedrock_incision

        self._dt = np.nanmin((dt_sediment , dt_flow, dt_bedrock_incision))
        return self._dt


        #
        # # Dt - clast transport
        # if np.any(row_water) ==  False:
        #     self._erosion_dt = np.inf
        # else:
        #
        #     self._tau_star_c_at_node = np.zeros_like(self._zeros_at_node_for_fractions)+10000
        #     indices_nonzero_mgs= np.where(median_grain_size>0)[0]
        #     self._tau_star_c_at_node[indices_nonzero_mgs,:] = self._alpha * (
        #                 (self.grid.at_node['fraction_sizes'][indices_nonzero_mgs,:] / median_grain_size_vec[indices_nonzero_mgs]) ** self._beta)
        #
        #
        #     # Map the upwind node to the link
        #     upwind_node_id_at_link = self.grid.map_value_at_max_node_to_link('water_surface__elevation',
        #                                                                      self.node_flatten, )
        #     nonzero_upwind_node_ids = upwind_node_id_at_link.astype('int')
        #     nonzero_downind_link_ids = self._links_array
        #     # Map the tau_star_c from nodes to links
        #     self._tau_star_c_links[nonzero_downind_link_ids, :] = self._tau_star_c_at_node[nonzero_upwind_node_ids, :]
        #
        #     # Calculate shear stress and shear stress star at LINKS:  links x size fractions
        #     stress_at_link[:] = 0
        #     stress_at_link[row_water] = self._unitwt * depth_at_link[row_water ] * wsgrad[row_water]
        #     stress_at_link[self._inactive_links] = 0
        #
        #
        #     # Map the mean size of all fractions from nodes to links
        #     mean_sizes_at_link[nonzero_downind_link_ids, :] = fractions_sizes[nonzero_upwind_node_ids, :]
        #
        #     # Convert stress to stress_star at link
        #     denominator_for_stress_star[:] += 10 ** 4
        #     denominator_for_stress_star[row_water,:] = (self._sigma - self._rho) * self._g * mean_sizes_at_link[row_water,:]
        #
        #     stress_star_at_link[row_water,:] =np.divide(
        #         stress_at_link[row_water].reshape([-1, 1]),
        #         denominator_for_stress_star[row_water, :])
        #
        #     # Excess stress at link
        #
        #     excess_stress =  np.zeros_like(self._zeros_at_link_for_fractions)
        #     excess_stress[row_water] = np.abs(stress_star_at_link[row_water,:]) - self._tau_star_c_links[row_water,:]
        #
        #     if np.any(excess_stress > 0) == True:
        #         row, col = np.where(excess_stress>0)
        #
        #         g_total_dt_node = np.sum(grain_weight_node, 1)  # Total grain size mass at node
        #
        #         # Initilize vectors for link
        #         g_total_link = np.zeros((self._n_links,1)) # Total grain size mass at link
        #         g_state_link = np.zeros_like(self._zeros_at_link_for_fractions) # Grain size mass for each size fraction
        #
        #         g_total_link[nonzero_downind_link_ids,0] = g_total_dt_node[nonzero_upwind_node_ids] # Total sediment mass for of each up-wind node mapped to link.  # Total sediment mass for of each up-wind node mapped to link.
        #         g_state_link[nonzero_downind_link_ids, :] = grain_weight_node[nonzero_upwind_node_ids,
        #                                                     :]  # Fraction of sediment mass of each upwind node, for all size-fraction, mapped to link
        #         g_fraction_link = np.divide(g_state_link, g_total_link, out=np.zeros_like(g_state_link),
        #                                     where=g_total_link != 0)
        #
        #         ## MPM equation
        #         sed_flux_star_at_link_classes = np.zeros_like(self._zeros_at_link_for_fractions)
        #         sed_flux_star_at_link_classes[row,col] = 8 * (excess_stress[row,col]) ** 1.5
        #
        #         #sed_flux_star_at_link_classes = 8 * (excess_stress) ** 1.5
        #         self._sed_flux_at_link_class[row,col] = np.multiply(-np.sign(
        #             stress_star_at_link[row,col]), sed_flux_star_at_link_classes[row,col] * ((((( self._sigma - self._rho) / self._rho) * self._g * mean_sizes_at_link[row,col]) ** 0.5) * mean_sizes_at_link[row,col]))
        #
        #         # Take care for flux per size fraction
        #         self._sed_flux_at_link_class[row,col] = np.multiply(self._sed_flux_at_link_class[row,col], g_fraction_link[row,col])  # Multipe by the fraction of mass of all size fractions
        #         self._sed_flux_at_link_class = np.sum(self._sed_flux_at_link_class,1)
        #
        #     if np.any(np.abs(self._sed_flux_at_link_class) > 0):
        #         max_elev_change = (topo_gradient * self._grid.dx) / 2  # m
        #         dz_by_flux = ((self._sed_flux_at_link_class) / self._grid.dx)  # m/s
        #         dz_by_flux[dz_by_flux == 0] = -np.inf
        #         # maximal_possible_dz = (g_total_link / self._rho) / (self._grid.dx *self._grid.dy)
        #         # maximal_possible_dz[maximal_possible_dz==0] = np.nan
        #         max_dt_elev_change = np.abs(max_elev_change / dz_by_flux)
        #         max_dt_elev_change[max_dt_elev_change ==0] = np.nan
        #         #max_dt_elev_change_weight = np.abs(maximal_possible_dz / dz_by_flux)
        #         self._erosion_dt = np.nanmin(max_dt_elev_change)
        #         if self._erosion_dt == 0:
        #             self._erosion_dt = np.inf
        #     else:
        #         self._erosion_dt = np.inf




