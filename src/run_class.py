from typing import Dict, Callable, List, Any, Tuple
import copy
import pandas as pd
import numpy as np
import os


class RunsLoader():
    def __init__(self, params: List,
                 load_and_process_data: Callable[[str], pd.DataFrame],
                 n_ensemble=1, verbose=0):
        self._params = copy.deepcopy(params)
        self._load_and_process_data = load_and_process_data
        self._n_ensemble = n_ensemble
        self._n_data = 1
        self._verbose = verbose

        # n_ensemble x data_length x [data_fields] x [params]
        self._data = None
        self._param_names = [x[0] for x in self._params]
        self._param_sizes = [len(x[1]) for x in self._params]
        self._data_columns = []

        self._iterate_recursive(self._params, [])

    def _iterate_recursive(self, remaining_params: List,
                           params: List, rec_index=0):
        if not remaining_params:
            params_to_dict = {x[0]: x[1] for x in params}
            # key = '_'.join(['='.join(x) for x in params])
            tmp_dfs = []
            for ensemble in range(self._n_ensemble):
                if self._verbose > len(self._params):
                    print(f"{ensemble + 1}/{self._n_ensemble}")
                tmp_dfs.append(self._load_and_process_data(**params_to_dict,
                                                           ensemble=ensemble))

            self._dimension_names = ['ensemble', 'data_length']
            self._data_columns = list(tmp_dfs[0].columns)
            self._dimension_names += self._data_columns
            self._dimension_names += self._param_names

            dimensions = [self._n_ensemble, len(tmp_dfs[0]),
                          len(tmp_dfs[0].columns)]
            dimensions += self._param_sizes
            self._data = np.zeros(dimensions)

            array_index = []
            for i, param in enumerate(params):
                array_index += [self._params[i][1].index(param[1])]

            for tmp_df, ensemble in zip(tmp_dfs, range(self._n_ensemble)):
                this_array_index = tuple([ensemble, ...] + array_index)
                self._data[this_array_index] = tmp_df[self._data_columns].values.copy()
        else:
            param_and_value = remaining_params[0]
            for value in param_and_value[1]:
                this_params = copy.deepcopy(params)
                this_params.append((param_and_value[0], value))

                self._iterate_recursive(remaining_params[1:],
                                        this_params, rec_index+1)


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

    def save(self, path_to_hdf5: str):
        pass


class RunsCreater():
    def __init__(self, path_to_bash_file: str,
                 output_prefix: str,
                 executable_str: str,
                 output_files: List[str],  # list of files to be copied
                 params: List,
                 sed_param_replacer: Callable[[str], List[str]],
                 n_ensemble=1, verbose=0):
        self._path_to_bash_file = path_to_bash_file
        self._output_prefix = output_prefix
        self._executable_str = executable_str
        self._output_files = copy.deepcopy(output_files)
        self._params = copy.deepcopy(params)
        self._sed_param_replacer = sed_param_replacer
        self._n_ensemble = n_ensemble
        self._n_data = 1
        self._verbose = verbose

        self._create_bash_file()


    def _create_bash_file(self):
        with open(self._path_to_bash_file, 'w') as f:
            for param in self._params:
                param_str = ' '.join([f'"{x}"' for x in param[1]])
                f.write(f'declare -a {param[0]}_arr=({param_str})\n')
            f.write('\n')

            for indent, param in enumerate(self._params):
                indent_str = '\t' * indent
                f.write(f'{indent_str}for {param[0]} in "${{{param[0]}_arr[@]}}"\n{indent_str}do\n\n')
                if self._verbose > indent:
                    indent_str2 = indent_str + '\t'
                    f.write(f'{indent_str2}echo ${param[0]}\n')

                replacer_str = self._sed_param_replacer(param[0])
                indent_str2 = indent_str + '\t'
                f.write(f"{indent_str2}sed -i 's/{replacer_str[0]}/{replacer_str[1]}/g'\n")

            indent_str = '\t' * len(self._params)

            param_dirs = [f'${{{param[0]}}}' for param in self._params]
            path_to_create = os.path.join(self._output_prefix, *param_dirs)

            if self._n_ensemble > 1:
                path_to_create = os.path.join(path_to_create, "${ensemblenumber}")
                f.write(f"{indent_str}for ensemblenumber in {{1..{self._n_ensemble}}}\n{indent_str}do\n")
                replacer_str = self._sed_param_replacer("ensemblenumber")
                indent_str += '\t'
                f.write(f"{indent_str}sed -i 's/{replacer_str[0]}/{replacer_str[1]}/g'\n")

            f.write(f'{indent_str}mkdir -p {path_to_create}\n')
            f.write(f'{indent_str}{self._executable_str}\n')

            for filename in self._output_files:
                f.write(f'{indent_str}mv {filename} {path_to_create}\n')

            if self._n_ensemble > 1:
                indent_str = indent_str[:-1]
                f.write(f'{indent_str}done\n')

            for indent, param in enumerate(self._params):
                indent_str = '\t' * (len(self._params) - 1 - indent)
                f.write(f'{indent_str}done\n')
