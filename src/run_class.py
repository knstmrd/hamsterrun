from typing import Dict, Callable, List
import copy
import pandas as pd


class Runs():
    def __init__(self, params: Dict, param_to_path_map: Callable[..., str],
                 load_and_process_data: Callable[str, pd.DataFrame],
                 n_ensemble=1, verbose=0):
        self._params = copy.deepcopy(params)
        self._param_to_path_map = param_to_path_map
        self._data = {}
        self._n_ensemble = n_ensemble

    def iterate_over_parameter(parameter: str, fixed_param_values: Dict,
                               output_values: List, ensembling_function=None):
        pass
