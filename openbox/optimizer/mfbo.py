# License: MIT

import time
import math
from typing import List
from tqdm import tqdm
import numpy as np
from openbox import logger
from openbox.optimizer.base import BOBase
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.utils.util_funcs import parse_result, deprecate_kwarg, check_random_state
from openbox.utils.history import Observation, History
from openbox.visualization import build_visualizer


class MFBO(BOBase):
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            objective_function: callable,
            config_space,
            num_objectives=1,
            num_constraints=0,
            sample_strategy: str = 'bo',
            max_runs=200,
            runtime_limit=None,
            time_limit_per_trial=180,
            advisor_type='default',
            surrogate_type='auto',
            acq_type='auto',
            acq_optimizer_type='auto',
            initial_runs=3,
            init_strategy='random_explore_first',
            initial_configurations=None,
            ref_point=None,
            transfer_learning_history: List[History] = None,
            logging_dir='logs',
            task_id='OpenBox',
            visualization='none',
            auto_open_html=False,
            random_state=None,
            logger_kwargs: dict = None,
            advisor_kwargs: dict = None,
    ):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')
        

        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.FAILED_PERF = [np.inf] * num_objectives
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         runtime_limit=runtime_limit, sample_strategy=sample_strategy,
                         time_limit_per_trial=time_limit_per_trial, transfer_learning_history=transfer_learning_history,
                         logger_kwargs=logger_kwargs)
        

        self.rng = check_random_state(random_state)
        self.config_space = config_space
        # init history
        self.history = History(
            task_id=task_id, num_objectives=num_objectives, num_constraints=num_constraints, config_space=config_space,
            ref_point=ref_point, meta_info=None,  # todo: add meta info
        )
        self.surrogate_model1 = None
        self.surrogate_model2 = None
        self.acquisition_function = None
        self.optimizer = None
        

        

    def run(self) -> History:
        for _ in tqdm(range(self.iteration_id, self.max_iterations)):
            # get configuration suggestion from advisor
            config = self.config_advisor.get_suggestion()

            try:
                # evaluate configuration on objective_function within time_limit_per_trial
                args, kwargs = (config,), dict()
                timeout_status, _result = time_limit(self.objective_function,
                                                    _time_limit_per_trial,
                                                    args=args, kwargs=kwargs)

            except Exception as e:
                logger.warning(f'Exception when calling objective function: {e}\nconfig: {config}')
                trial_state = FAILED
                objectives = self.FAILED_PERF
                constraints = None
                extra_info = None
                inner_config = None
            
            # update observation to advisor
            observation = Observation(
                config=config, objectives=objectives, constraints=constraints,
                trial_state=trial_state, elapsed_time=elapsed_time, extra_info=extra_info, inner_config=inner_config,
            )

            self.config_advisor.update_observation(observation)

            self.iteration_id += 1
            # Logging
            if self.num_constraints > 0:
                logger.info('Iter %d, objectives: %s. constraints: %s.'
                                % (self.iteration_id, objectives, constraints))
            else:
                logger.info('Iter %d, objectives: %s.' % (self.iteration_id, objectives))

            
        return self.get_history()
    
    def get_suggestion(self, history: History = None):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """
        if history is None:
            history = self.history

        num_config_evaluated = len(history)
        num_config_successful = history.get_success_count()

        if self.rng.random() < self.rand_prob:
            logger.info('Sample random config. rand_prob=%f.' % self.rand_prob)
            res = self.sample_random_configs(1, history)[0]
            return [res] if return_list else res

        X = history.get_config_array(transform='scale')
        Y = history.get_objectives(transform='infeasible')

        if num_config_successful < max(self.init_num, 1):
            logger.warning('No enough successful initial trials! Sample random configuration.')
            res = self.sample_random_configs(1, history)[0]
            return [res] if return_list else res

        # train surrogate model
        if self.num_objectives == 1:
            self.surrogate_model.train(X, Y[:, 0])
        elif self.acq_type == 'parego':
            weights = self.rng.random_sample(self.num_objectives)
            weights = weights / np.sum(weights)
            scalarized_obj = get_chebyshev_scalarization(weights, Y)
            self.surrogate_model.train(X, scalarized_obj(Y))
        else:  # multi-objectives
            for i in range(self.num_objectives):
                self.surrogate_model[i].train(X, Y[:, i])

        # train constraint model
        for i in range(self.num_constraints):
            self.constraint_models[i].train(X, cY[:, i])

        # update acquisition function
        if self.num_objectives == 1:
            incumbent_value = history.get_incumbent_value()
            self.acquisition_function.update(model=self.surrogate_model,
                                                constraint_models=self.constraint_models,
                                                eta=incumbent_value,
                                                num_data=num_config_evaluated)
        else:  # multi-objectives
            mo_incumbent_values = history.get_mo_incumbent_values()
            if self.acq_type == 'parego':
                self.acquisition_function.update(model=self.surrogate_model,
                                                    constraint_models=self.constraint_models,
                                                    eta=scalarized_obj(np.atleast_2d(mo_incumbent_values)),
                                                    num_data=num_config_evaluated)
            elif self.acq_type.startswith('ehvi'):
                partitioning = NondominatedPartitioning(self.num_objectives, Y)
                cell_bounds = partitioning.get_hypercell_bounds(ref_point=self.ref_point)
                self.acquisition_function.update(model=self.surrogate_model,
                                                    constraint_models=self.constraint_models,
                                                    cell_lower_bounds=cell_bounds[0],
                                                    cell_upper_bounds=cell_bounds[1])
            else:
                self.acquisition_function.update(model=self.surrogate_model,
                                                    constraint_models=self.constraint_models,
                                                    constraint_perfs=cY,  # for MESMOC
                                                    eta=mo_incumbent_values,
                                                    num_data=num_config_evaluated,
                                                    X=X, Y=Y)

        # optimize acquisition function
        challengers = self.optimizer.maximize(runhistory=history,
                                                num_points=50000)
        if return_list:
            # Caution: return_list doesn't contain random configs sampled according to rand_prob
            return challengers.challengers

        for config in challengers.challengers:
            if config not in history.configurations:
                return config
        logger.warning('Cannot get non duplicate configuration from BO candidates (len=%d). '
                            'Sample random config.' % (len(challengers.challengers), ))
        return self.sample_random_configs(1, history)[0]


    def sample_random_configs(self, num_configs=1, history=None, excluded_configs=None):
        """
        Sample a batch of random configurations.
        Parameters
        ----------
        num_configs

        history

        Returns
        -------

        """
        if history is None:
            history = self.history
        if excluded_configs is None:
            excluded_configs = set()

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = self.config_space.sample_configuration()
            sample_cnt += 1
            if config not in (history.configurations + configs) and config not in excluded_configs:
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs