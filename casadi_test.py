import casadi as ca
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


class Casadi(object):
    def eval(self):
        # Declare simulation constants
        T = 3 # planning horizon
        N = 30 # Number of control intervals
        h = T / N

        # system dimensions
        Dim_state = 2
        Dim_ctrl  = 1

        # Declare model variables
        x = ca.MX.sym('x', (Dim_state, N + 1))
        u = ca.MX.sym('u', (Dim_ctrl, N))

        # Continuous dynamics model
        x_model = ca.MX.sym('xm', (Dim_state, 1))
        u_model = ca.MX.sym('um', (Dim_ctrl, 1))

        xdot = ca.vertcat(x_model[1],
                        u_model[0])

        # system dynamics in discrete time
        Fun_dynmaics_dt = ca.Function('f', [x_model, u_model], [xdot * h + x_model])

        # cost
        L = x_model[0]**2 + x_model[1]**2 + 0.01 * u_model[0]**2
        P = (x_model[0]**2 + x_model[1]**2) * 10
        Fun_cost_terminal = ca.Function('P', [x_model], [P])
        Fun_cost_running = ca.Function('Q', [x_model, u_model], [L])

        # state and control constraints
        state_ub = np.array([ 1e4,  1e4])
        state_lb = np.array([-1e4, -1e4])
        ctrl_ub = np.array([ 4])
        ctrl_lb = np.array([-4])

        # initial condition
        x_init = [2, 2]

        # upper bound and lower bound
        ub_x = np.matlib.repmat(state_ub, N + 1, 1)
        lb_x = np.matlib.repmat(state_lb, N + 1, 1)

        ub_x[0, :] = x_init
        lb_x[0, :] = x_init

        ub_u = np.matlib.repmat(ctrl_ub, N, 1)
        lb_u = np.matlib.repmat(ctrl_lb, N, 1)

        ub_var = np.concatenate((ub_x.reshape((Dim_state * (N+1), 1)), ub_u.reshape((Dim_ctrl * N, 1))))
        lb_var = np.concatenate((lb_x.reshape((Dim_state * (N+1), 1)), lb_u.reshape((Dim_ctrl * N, 1))))

        # dynamics constraints: x[k+1] = x[k] + f(x[k], u[k]) * dt
        cons_dynamics = []
        ub_dynamics = np.zeros((N * Dim_state, 1))
        lb_dynamics = np.zeros((N * Dim_state, 1))
        for k in range(N):
            Fx = Fun_dynmaics_dt(x[:, k], u[:, k])
            for j in range(Dim_state):
                cons_dynamics.append(x[j, k+1] -  Fx[j])

        # state constraints: G(x) <= 0
        cons_state = []
        ub_state_cons = np.zeros((N, 1))
        lb_state_cons = np.zeros((N, 1)) - 1e5
        for k in range(N):
            cons_state.append(x[1, k+1]**2 + x[0, k+1]**2 - 400.0)

        # cost function: 
        J = Fun_cost_terminal(x[:, -1])
        for k in range(N):
            J = J + Fun_cost_running(x[:, k], u[:, k])
        

        # Define variables for NLP solver
        vars_NLP = ca.vertcat(x.reshape((Dim_state * (N+1), 1)), u.reshape((Dim_ctrl * N, 1)))
        cons_NLP = cons_dynamics + cons_state
        cons_NLP = ca.vertcat(*cons_NLP)
        lb_cons = np.concatenate((lb_dynamics, lb_state_cons))
        ub_cons = np.concatenate((ub_dynamics, ub_state_cons))
        # print(vars_NLP)

        # Create an NLP solver
        prob = {'f': J, 'x': vars_NLP, 'g':cons_NLP}
        # opts = {'ipopt.print_level': 0, 'print_time': 0} # , 'ipopt.sb': 'yes'}
        solver = ca.nlpsol('solver', 'ipopt', prob) # , opts)

        x0_nlp = np.random.randn(vars_NLP.shape[0], vars_NLP.shape[1])

        sol = solver(x0=x0_nlp, lbx=lb_var, ubx=ub_var, lbg=lb_cons, ubg=ub_cons)


        sol_num = sol['x'].full()
        sol_x = sol_num[0:(N + 1) * Dim_state]
        sol_u = sol_num[(N + 1) * Dim_state:]
        print("sol_u: ", sol_u)
        print("sol_u.shape: ", sol_u.shape)
        print("sol_u.type: ", type(sol_u))
        return sol_u
        

def main():
    print("Welcome to the world of CASADI!!!")
    casa.eval()
    print("Goodbye!")

if __name__ == "__main__":
    casa = Casadi()
    main()