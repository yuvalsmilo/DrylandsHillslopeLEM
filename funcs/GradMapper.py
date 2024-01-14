import numpy as np
from landlab import Component

class GradMapper(Component):
    _name = "GradMapper"
    _unit_agnostic = True
    _info = {
        'topographic__gradient': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/m",
            "mapping": "link",
            "doc": "Topographic gradient at link",
        },
        'downwind__link_gradient': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/m",
            "mapping": "node",
            "doc": "Gradient of link to downwind node at node",
        },
        'downwind__link_gradient': {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/m",
            "mapping": "node",
            "doc": "Gradient of link to upwind node at node",
        },
    }

    def __init__(
            self,
            grid,
            minslope = 0.1,
    ):
        super(GradMapper, self).__init__(grid)

        grid.add_zeros('topographic__gradient', at="link")
        grid.add_zeros('downwind__link_gradient', at="node")
        grid.add_zeros('upwnwind__link_gradient', at="node")
        self._minslope = minslope

    def run_one_step(self,):
        # positive link direction is INCOMING
        gradient_of_downwind_link_at_node = self._grid.at_node['downwind__link_gradient']
        gradient_of_upwind_link_at_node = self._grid.at_node['upwnwind__link_gradient']
        cliff_nodes = self._grid.at_node['cliff_nodes']
        topographic_gradient_at_link = self._grid.at_link['topographic__gradient']
        gradients_vals = self._grid.at_node['water_surface__slope']
        topographic_gradient_at_link[:] = self._grid.calc_grad_at_link('topographic__elevation')


        gradient_of_upwind_link_at_node[:] = -self._grid.map_upwind_node_link_max_to_node(topographic_gradient_at_link) # Map the largest magnitude of the links bringing flux into the node to the node.
        gradient_of_upwind_link_at_node[gradient_of_upwind_link_at_node>0] = 0 # POSTIVE HERE ARE OUTFLUX
        gradient_of_upwind_link_at_node = np.abs(gradient_of_upwind_link_at_node)


        values_at_links = topographic_gradient_at_link[self._grid.links_at_node] * self._grid.link_dirs_at_node         # this procedure makes incoming links NEGATIVE
        steepest_links_at_node = np.amax(values_at_links, axis=1) # take the maximm (positive means out link)

        gradient_of_downwind_link_at_node[:] = 0 # set all to zero
        gradient_of_downwind_link_at_node[:] = np.fmax(steepest_links_at_node,
                                                       gradient_of_downwind_link_at_node) # if maximal link is negative, it will be zero. meaning, no outflux
        gradient_of_downwind_link_at_node[gradient_of_downwind_link_at_node <= self._minslope] = 0



        ### CALC WATER SURFACE GRADIENT AT !NODE!
        water_gradient_at_link = self._grid.calc_grad_at_link('water_surface__elevation')

        values_at_links = water_gradient_at_link[
                              self._grid.links_at_node] * self._grid.link_dirs_at_node  # this procedure makes incoming links NEGATIVE
        steepest_links_at_node = np.amax(values_at_links, axis=1)  # take the maximm (positive means out link)

        gradients_vals[:]=0
        watergradient_of_downwind_link_at_node = np.fmax(steepest_links_at_node,
                                                       gradients_vals)  # if maximal link is negative, it will be zero. meaning, no outflux
        watergradient_of_downwind_link_at_node[watergradient_of_downwind_link_at_node <= self._minslope] = 0
        gradients_vals[:] = watergradient_of_downwind_link_at_node[:]

        return