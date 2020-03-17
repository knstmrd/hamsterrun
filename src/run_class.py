from typing import Dict, Callable, List
import copy
import pandas as pd
import numpy as np


class Runs():
    def __init__(self, params: List,
                 load_and_process_data: Callable[[str], pd.DataFrame],
                 n_ensemble=1, verbose=0):
        self._params = copy.deepcopy(params)
        self._load_and_process_data = load_and_process_data
        self._data = pd.DataFrame({})  # data is n_ensemble x n_data rows x possible combinations of parameter values
        self._n_ensemble = n_ensemble
        self._n_data = 1
        self._verbose = verbose

        self._iterate_recursive(self._params, [])
        self._data.reset_index(inplace=True, drop=True)
        # for ensemble in n_ensemble:

    def _iterate_recursive(self, remaining_params: List,
                           params: List):
        if not remaining_params:
            params_to_dict = {x[0]: x[1] for x in params}
            key = '_'.join(['='.join(x) for x in params])

            tmp_dfs = []
            for ensemble in range(self._n_ensemble):
                if self._verbose > len(self._params):
                    print(f"{ensemble + 1}/{self._n_ensemble}")
                tmp_dfs.append(self._load_and_process_data(**params_to_dict,
                                                           ensemble=ensemble))
                # for columns in tmp_df.columns:
            tmp_df = pd.concat(tmp_dfs)
            for column in tmp_df.columns:
                self._data[key + '/' + column] = tmp_df[column].copy()

        else:
            for param_and_values in remaining_params:
                for value in param_and_values[1]:
                    if self._verbose > len(params):
                        print(f"{param_and_values[0]}={value}")
                    this_params = copy.deepcopy(params)
                    this_params.append((param_and_values[0], value))
                    self._iterate_recursive(remaining_params[1:], this_params)

    def iterate_over_parameter(self, parameter: str, fixed_param_values: Dict,
                               output_value_names: List,
                               ensembling_function=None):
        if ensembling_function is None:
            n_ensemble_mult = self._n_ensemble
        else:
            n_ensemble_mult = 1
        output_arr = {x: np.zeros(len(n_ensemble_mult * self._n_data))
                      for x in output_value_names}

        return pd.DataFrame(output_arr)
