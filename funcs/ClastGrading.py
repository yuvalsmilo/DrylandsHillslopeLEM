
import numpy as np
from landlab import Component
import time
from landlab.grid.nodestatus import NodeStatus


class ClastGrading(Component):

    _name = 'ClastGrading'
    _unit_agnostic = True
    _info = {"soil__depth":{
          "dtype":float,
          "intent":"out",
          "optional":False,
          "units": "m",
          "mapping":"node",
          "doc":"Soil depth at node",
      },
        "grain__weight": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "Sediment weight for all size fractions",
        },
        "median__size_weight": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "The median grain size in each node based on grain size weight",
        },
        "fraction_sizes": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "The size of grain fractions",
        },
    }

    def __init__(self,
                 grid,
                 grading_name =  'p2-0-100',  # Fragmentation pattern (string)
                 n_size_classes = 10,         # Number of size classes
                 alpha = 1,
                 clast_density = 2000,        # Particles density [kg/m3]
                 **kwargs,
    ):
        super(ClastGrading, self).__init__(grid)

        self.grading_name = grading_name
        self.kwargs = kwargs
        self.alpha = alpha
        self.n_sizes = n_size_classes
        self.N = int(self.grading_name.split('-')[0][1:])
        self.clast_density = clast_density

        # Create out fields
        grid.add_zeros('median__size_weight', at='node')
        grid.add_zeros('soil__depth', at='node')
        grid.add_field("fraction_sizes", np.ones((grid.shape[0], grid.shape[1], self.n_sizes )), at="node", dtype=float)
        grid.add_field("grain__weight", np.ones((grid.shape[0], grid.shape[1], self.n_sizes)), at="node", dtype=float)

    def create_transion_mat(self, n_fragments=2):

        self.A = np.zeros((self.n_sizes, self.n_sizes))
        precents = np.array([float(s) for s in self.grading_name.split('-') if s.replace('.', '', 1).isdigit()])
        if 'spread' in self.grading_name:
            volume_precent_in_spread = 1 / n_fragments
            precents_to_add = np.ones((1, n_fragments)) * volume_precent_in_spread
            precents = np.append(precents, precents_to_add)
        alphas_fractios = precents / 100
        self.A_factor = np.ones_like(self.A)

        for i in range(self.n_sizes):

            if i == 0:
                self.A[i, i] = 0
            elif i == self.n_sizes:  ## the last cell in the matrix
                self.A[i, i] = -(self.alpha - (
                            self.alpha * alphas_fractios[0])
                                 )  ## this is the volume fraction that weathered
            else:
                self.A[i, i] = -(self.alpha - (
                        self.alpha * alphas_fractios[0])
                                 )
                cnti = i - 1
                cnt = 1
                while cnti >= 0 and cnt <= (len(alphas_fractios) - 1):
                    self.A[cnti, i] = (self.alpha * alphas_fractios[cnt])
                    if cnti == 0 and cnt <= (len(alphas_fractios) - 1):
                        self.A[cnti, i] = (1 - alphas_fractios[0]) - np.sum(alphas_fractios[1:cnt])
                        cnt += 1
                        cnti -= 1
                    cnt += 1
                    cnti -= 1

    def set_grading_classes(self, maxsize, power_of = 1 / 3):


        def lower_limit_of(maxsize):
            lower_limit = maxsize * (1 / self.N) ** (power_of)
            return lower_limit

        upperlimits = []
        lowerlimits = []
        num_of_size_classes_plusone = self.n_sizes + 1

        for _ in range(num_of_size_classes_plusone):
            upperlimits.append(maxsize)
            maxsize = lower_limit_of(maxsize)
            lowerlimits.append(maxsize)

        self.upperlims = upperlimits
        self.lowerlims = lowerlimits
        self.meansizes = np.flip(np.array(lowerlimits)[0:-1] + np.abs(np.diff(np.array(upperlimits)) / 2))
        self.clast_volumes = (self.meansizes / 2) ** 3 * np.pi * (4 / 3)  # in m^3
        self.grid.at_node["fraction_sizes"] *= self.meansizes
        self.update_sizes()

    def create_dist(self,
                    median_size = 0.05,
                    std = None,
                    num_of_clasts = 10000,
                    grading_name = 'g_state0',
                    init_val_flag = False):

        n_classes = self.n_sizes
        median_size = median_size
        num_of_clasts = num_of_clasts
        if std == None:
            std = median_size * 0.8
        b = np.random.normal(median_size, std, num_of_clasts)

        locals()[grading_name] = np.histogram(b, np.insert(np.sort(self.upperlims), 0, 0))[0][:-1]
        if init_val_flag:
            self.g_state = np.ones(
                (int(np.shape(self.grid)[0]), int(np.shape(self.grid)[1]), int(len(locals()[grading_name])))) * locals()[grading_name]
            self.g_state0 = locals()[grading_name]
            self.grid.at_node['grain__weight'] *= locals()[grading_name]
            layer_depth = np.sum(self.g_state0) / (self.clast_density * self.grid.dx *self.grid.dx )  # meter
            self.grid.at_node['soil__depth'] +=  layer_depth
        else:
            self.g_state_slide = locals()[grading_name]

    def update_sizes(self):

        grain_number = self.grid.at_node['grain__weight'] / (self.clast_volumes * self.clast_density)
        grain_volume = grain_number * (self.clast_volumes)  # Volume
        grain_area = grain_volume / (self.meansizes)  # Area

        # Median size based on weight
        cumsum_gs = np.cumsum(self.grid.at_node['grain__weight'], axis=1)
        sum_gs = np.sum(self.grid.at_node['grain__weight'], axis=1)
        self.grid.at_node['median__size_weight'][sum_gs <= 0] = 0
        sum_gs_exp = np.expand_dims(sum_gs, -1)
        median_val_indx = np.argmin(
            np.abs(
                np.divide(
                    cumsum_gs,
                    sum_gs_exp,
                    out=np.zeros_like(cumsum_gs),
                    where=sum_gs_exp != 0) - 0.5),
            axis=1
        )
        self.grid.at_node['median__size_weight'] = self.meansizes[median_val_indx[:]]

    def run_one_step(self, A_factor=None):

        if np.any(A_factor == None):
            A_factor = self.A_factor
        temp_g_weight = np.moveaxis(np.dot(self.A * A_factor, np.swapaxes(
            np.reshape(self.grid.at_node['grain__weight'], (self.grid.shape[0], self.grid.shape[1], self.n_sizes)), 1,
            2)), 0, -1)
        self.grid.at_node['grain__weight'] += np.reshape(temp_g_weight,
                                                         (self.grid.shape[0] * self.grid.shape[1],
                                                          self.n_sizes))
        self.update_sizes()