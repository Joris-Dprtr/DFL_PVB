import cvxpy as cp


class PV_battery:

    def __init__(self,
                 house,
                 capacity,
                 max_charge,
                 max_discharge,
                 ):

        self.house = house
        self.capacity = capacity
        self.max_charge = max_charge
        self.max_discharge = max_discharge

    def create_optimization_problem(self, T):

        # Define Variables
        imp = cp.Variable(T)
        exp = cp.Variable(T)
        bat_energy = cp.Variable(T+1)
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
        objective = cp.Minimize(cp.sum(imp @  off - exp @ inj))

        # Constraints
        constraints = [
            pv + imp + bat_discharge == exp + load + bat_charge,
            exp >= 0,
            exp <= pv + bat_discharge,
            imp >= 0,
            bat_charge >= 0,
            bat_discharge >= 0,
            bat_charge <= self.max_charge * (1-mode),
            bat_discharge <= self.max_discharge * mode,
            bat_energy[0] == initial_battery_energy,
            bat_energy[-1] == self.capacity * 0.5,
            bat_energy >= self.capacity * 0.2,
            bat_energy <= self.capacity * 0.8,
            mode >= 0,
            mode <= 1
        ]

        for t in range(1, T+1):
            constraints += [
                bat_energy[t] == bat_energy[t-1] + bat_charge[t-1] - bat_discharge[t-1]
            ]

        # Optimization problem
        problem = cp.Problem(objective, constraints)

        return problem, variables, parameters

    def create_post_forecast_optimization_problem(self, T):

        # Define Variables
        imp = cp.Variable(T)
        exp = cp.Variable(T)
        bat_energy = cp.Variable(T+1)
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
        objective = cp.Minimize(cp.sum(imp @  off - exp @ inj))

        # Constraints
        constraints = [
            pv + imp + bat_discharge == exp + load + bat_charge,
            exp >= 0,
            exp <= pv + bat_discharge,
            imp >= 0,
            bat_energy[0] == initial_battery_energy,
            bat_energy[-1] == self.capacity * 0.5,
            bat_energy >= self.capacity * 0.05,
            bat_energy <= self.capacity * 0.95
        ]

        for t in range(1, T+1):
            constraints += [
                bat_energy[t] == bat_energy[t-1] + bat_charge[t-1] - bat_discharge[t-1]
            ]

        # Create and return the problem
        problem = cp.Problem(objective, constraints)

        return problem, variables, parameters
