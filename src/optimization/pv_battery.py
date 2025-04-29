import cvxpy as cp
import numpy as np
import src.tensors.tensorisation as tensor
from src.models.lstm import LSTM
from src.models.lstmopt import LSTMOPT
import torch
import copy


def _torch_py(torch_tensor):
    return torch_tensor.cpu().detach().numpy().flatten()


def _rescale(values, scaler):
    rescaled_values = values * (scaler[1] - scaler[0]) + scaler[0]
    return rescaled_values


class PV_battery:

    def __init__(self,
                 house,
                 house_nr,
                 capacity,
                 max_charge,
                 max_discharge,
                 self_consumption=True,
                 ):

        self.house = house
        self.house_nr = house_nr
        self.capacity = capacity
        self.max_charge = max_charge
        self.max_discharge = max_discharge
        self.self_consumption = self_consumption
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_optimization_problem(self, T):

        # Define Variables
        imp = cp.Variable(T)
        exp = cp.Variable(T)
        bat_energy = cp.Variable(T + 1)
        mode = cp.Variable(T)
        bat_charge = cp.Variable(T)
        bat_discharge = cp.Variable(T)
        variables = [imp, exp, bat_energy, mode, bat_charge, bat_discharge]

        # Define Parameters
        load = cp.Parameter(T)
        off = cp.Parameter(T)
        inj = cp.Parameter(T)
        pv = cp.Parameter(T)
        initial_battery_energy = cp.Parameter()

        parameters = [pv, load, off, inj, initial_battery_energy]

        # Objective Function
        objective = cp.Minimize(cp.sum(imp @ off - exp @ inj))

        # Constraints

        if self.self_consumption:
            constraints = [
                pv + imp + bat_discharge == exp + load + bat_charge,
                exp >= 0,
                imp >= 0,
                bat_charge >= 0,
                bat_discharge >= 0,
                bat_charge <= self.max_charge * (1 - mode),
                bat_charge <= pv - exp,
                bat_discharge <= self.max_discharge * mode,
                bat_energy[0] == initial_battery_energy,
                bat_energy[-1] == self.capacity * 0.5,
                bat_energy >= self.capacity * 0.2,
                bat_energy <= self.capacity * 0.8,
                mode >= 0,
                mode <= 1
            ]
        else:
            constraints = [
                pv + imp + bat_discharge == exp + load + bat_charge,
                exp >= 0,
                imp >= 0,
                bat_charge >= 0,
                bat_discharge >= 0,
                bat_charge <= self.max_charge * (1 - mode),
                bat_discharge <= self.max_discharge * mode,
                bat_energy[0] == initial_battery_energy,
                bat_energy[-1] == self.capacity * 0.5,
                bat_energy >= self.capacity * 0.2,
                bat_energy <= self.capacity * 0.8,
                mode >= 0,
                mode <= 1
            ]

        for t in range(1, T + 1):
            constraints += [
                bat_energy[t] == bat_energy[t - 1] + bat_charge[t - 1] - bat_discharge[t - 1]
            ]

        # Optimization problem
        problem = cp.Problem(objective, constraints)

        return problem, variables, parameters

    def create_post_forecast_optimization_problem(self, T):

        # Define Variables
        imp = cp.Variable(T)
        exp = cp.Variable(T)
        bat_energy = cp.Variable(T + 1)
        variables = [imp, exp, bat_energy]

        # Define Parameters
        pv = cp.Parameter(T)
        load = cp.Parameter(T)
        off = cp.Parameter(T)
        inj = cp.Parameter(T)
        initial_battery_energy = cp.Parameter()
        bat_charge = cp.Parameter(T)
        bat_discharge = cp.Parameter(T)

        parameters = [pv, load, off, inj, initial_battery_energy, bat_charge, bat_discharge]

        # Objective Function
        objective = cp.Minimize(cp.sum(imp @ off - exp @ inj))

        # Constraints
        constraints = [
            pv + imp + bat_discharge == exp + load + bat_charge,
            exp >= 0,
            imp >= 0,
            bat_energy[0] == initial_battery_energy,
            bat_energy[-1] == self.capacity * 0.5,
            bat_energy >= self.capacity * 0.05,
            bat_energy <= self.capacity * 0.95
        ]

        for t in range(1, T + 1):
            constraints += [
                bat_energy[t] == bat_energy[t - 1] + bat_charge[t - 1] - bat_discharge[t - 1]
            ]

        # Create and return the problem
        problem = cp.Problem(objective, constraints)

        return problem, variables, parameters

    def execute_optimization(self,
                             T,
                             min_T,
                             model,
                             neurons,
                             layers,
                             past_features,
                             future_features,
                             domain_min,
                             domain_max,
                             train_test_split=0.8,
                             noise:float=0):

        # These values get updated each loop
        lags = 24  # The number of lags for this first timeslot: the 24 previous hours
        forecast_gap = 0  # The gap after every forecast which is 0 when we need to forecast 24 hours
        dictionary_list = []

        local_domain_min = copy.deepcopy(domain_min)
        local_domain_max = copy.deepcopy(domain_max)

        # loop over each of the problems
        for t in range(T, min_T, -1):

            pvb_dictionary = {'imp': [],
                              'exp': [],
                              'energy': [],
                              'charge': [],
                              'discharge': [],
                              'pv': [],
                              'load': [],
                              'offtake': [],
                              'injection': []}

            print('Setting up optimization for ' + str(forecast_gap) + ':00')

            # Get the optimization problem for the current problem
            problem, variables, parameters = self.create_optimization_problem(t)
            # Get what actually happens based on the (dis)charging scheme to obtain the initial battery values
            problem_post, variables_post, parameters_post = self.create_post_forecast_optimization_problem(t)

            # Tensors for training an E2E network
            tensors_opt = tensor.Tensors(self.house, 'solar_energy', past_features, future_features, lags,
                                         t, forecast_gap=forecast_gap, train_test_split=train_test_split,
                                         domain_min=local_domain_min, domain_max=local_domain_max)

            # We don't need the Y values as they are identical to the ones from the base forecaster
            _, X_test_opt, _, y_test, scalers_opt = tensors_opt.create_tensor()

            # We have to assign initial battery values to the current optimization, first we create an empty tensor
            initial_bat_tensor_test = torch.zeros([X_test_opt.shape[0], lags, 1])

            # If this is the first optimization done at midnight, the initial battery is set at 50% of the capacity,
            # as we also make sure that the end state of the battery from the previous day is 50%
            if forecast_gap == 0:
                initial_bat_tensor_test[:, -1, :] = self.capacity * 0.5

            # If it is not the first optimization, we obtain the initial battery values from the list of battery values
            else:
                initial_bat_tensor_test[:, -1, :] = torch.tensor(
                    dictionary_list[forecast_gap - 1]['energy'][:, 1]).unsqueeze(-1)

            # We add this tensor to our X tensors for the E2E network
            X_test_opt = torch.concat([X_test_opt, initial_bat_tensor_test], dim=-1).to(self.device)

            # Create the models for PV forecasts

            # Features excludes the parameters, except for PV (so parameters - 1) and the perfect load (- 1)
            features = X_test_opt.shape[-1] - (len(parameters) - 1) - 1

            if model == 'Perfect':
                pv_test = y_test
            elif model == 'Naive':
                pv_test = X_test_opt[:, forecast_gap:24, 0]
            elif model == 'LSTM' or model == 'LSTM_Weather':
                lstm = LSTM(features, neurons, layers, t, 0.5).to(self.device)
                lstm.load_state_dict(
                    torch.load('../models/' + model + '/building' + str(self.house_nr) + '_' + str(t) + 'h.pth'))
                lstm.eval()
                pv_test = lstm(X_test_opt[:, :, 0:lstm.input_size])
            else:
                cvx = LSTMOPT(features, neurons, layers, t, 0.5, problem, parameters, variables, scalers_opt[0]).to(
                    self.device)
                cvx.load_state_dict(
                    torch.load('../models/' + model + '/building' + str(self.house_nr) + '_' + str(t) + 'h_' + str(
                        self.capacity) + 'kwh_' + str(noise) + 'noise.pth'))
                cvx.eval()
                pv_test, _ = cvx(X_test_opt[:, :, 0:cvx.input_size],
                                 X_test_opt[:, -t:, -4],
                                 X_test_opt[:, -t:, -3],
                                 X_test_opt[:, -t:, -2],
                                 X_test_opt[:, -1, -1])

            # Loop over the days, first use the forecast of PV, next plug in the real PV, charge and discharge schedules

            for j in range(len(X_test_opt)):
                if model == "Perfect":
                    parameters[0].value = _torch_py(y_test[j])

                    parameters[1].value = _torch_py(X_test_opt[j, -t:, -5])
                else:
                    parameters[0].value = _torch_py(_rescale(pv_test[j], scalers_opt[0]))
                    parameters[1].value = _torch_py(X_test_opt[j, -t:, -4])
                parameters[2].value = _torch_py(X_test_opt[j, -t:, -3])
                parameters[3].value = _torch_py(X_test_opt[j, -t:, -2])
                parameters[4].value = round(_torch_py(X_test_opt[j, -1:, -1].double())[0],4)
                problem.solve()
                if model == "Perfect":
                    pvb_dictionary['imp'].append(variables[0].value)
                    pvb_dictionary['exp'].append(variables[1].value)
                    pvb_dictionary['energy'].append(variables[2].value)
                    pvb_dictionary['charge'].append(variables[-2].value)
                    pvb_dictionary['discharge'].append(variables[-1].value)
                    pvb_dictionary['pv'].append(parameters[0].value)
                    pvb_dictionary['load'].append(parameters[1].value)
                    pvb_dictionary['offtake'].append(parameters[2].value)
                    pvb_dictionary['injection'].append(parameters[3].value)
                else:
                    parameters_post[0].value = _torch_py(y_test[j])
                    parameters_post[1].value = _torch_py(X_test_opt[j, -t:, -5])
                    parameters_post[2].value = _torch_py(X_test_opt[j, -t:, -3])
                    parameters_post[3].value = _torch_py(X_test_opt[j, -t:, -2])
                    parameters_post[4].value = round(_torch_py(X_test_opt[j, -1:, -1].double())[0],4)
                    parameters_post[5].value = variables[-2].value
                    parameters_post[6].value = variables[-1].value
                    problem_post.solve()
                    pvb_dictionary['imp'].append(variables_post[0].value)
                    pvb_dictionary['exp'].append(variables_post[1].value)
                    pvb_dictionary['energy'].append(variables_post[2].value)
                    pvb_dictionary['pv'].append(parameters[0].value)
                    pvb_dictionary['load'].append(parameters_post[1].value)
                    pvb_dictionary['offtake'].append(parameters_post[2].value)
                    pvb_dictionary['injection'].append(parameters_post[3].value)
                    pvb_dictionary['charge'].append(parameters_post[5].value)
                    pvb_dictionary['discharge'].append(parameters_post[6].value)


                    # Add the initial battery values to our list

            pvb_dictionary['pv'] = np.array(pvb_dictionary['pv'])
            pvb_dictionary['load'] = np.array(pvb_dictionary['load'])
            pvb_dictionary['imp'] = np.array(pvb_dictionary['imp'])
            pvb_dictionary['exp'] = np.array(pvb_dictionary['exp'])
            pvb_dictionary['energy'] = np.array(pvb_dictionary['energy'])
            pvb_dictionary['charge'] = np.array(pvb_dictionary['charge'])
            pvb_dictionary['discharge'] = np.array(pvb_dictionary['discharge'])
            pvb_dictionary['offtake'] = np.array(pvb_dictionary['offtake'])
            pvb_dictionary['injection'] = np.array(pvb_dictionary['injection'])

            dictionary_list.append(pvb_dictionary)

            # Update the lags we can use for the forecast
            lags += 1
            # Add to the gap between forecasts (f.e. the gap is 1 if we only have to forecast 23 hours)
            forecast_gap += 1

        return dictionary_list
