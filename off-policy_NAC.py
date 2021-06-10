import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
import matplotlib
from scipy.special import softmax

class MarkovDP:
    def __init__(self, s, a, gamma, rho_bar, c_bar, alpha, beta):
        self.num_state = s
        self.num_action = a
        self.mu = np.ones((1, self.num_state)) / self.num_state
        self.gamma = gamma
        self.rho_bar = rho_bar
        self.c_bar = c_bar
        self.alpha = alpha
        self.beta = beta
        self.states = np.array(range(0, s))
        self.actions = np.array(range(0, a))
        self.transitions = np.zeros((a, s, s))  # self.transitions (a,s,s') = P(s'|s,a)
        self.rewards = np.zeros((s, a, s))  # seld.rewards(s,a,s') = r(s,a,s')
        self.initialize_mdp()

    # The function below initializes transition probability matrix and rewards at a random value

    def initialize_mdp(self):
        self.transitions = np.random.rand(self.num_action, self.num_state, self.num_state)
        self.transitions = self.transitions / np.sum(self.transitions, axis=2, keepdims=True)
        self.rewards = np.random.rand(self.num_state, self.num_action, self.num_state)
        self.optimum_policy_finder()

    # The function below initializes transition probability matrix and rewards with the particular values

    def circle_MDP_initialization(self):
        self.transitions[0, :, :] = np.array(
            [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]])
        self.transitions[1, :, :] = np.array(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
        self.transitions[2, :, :] = np.array(
            [[0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])

        for a in range(self.num_action):
            for s in range(self.num_state):
                for sp in range(self.num_state):
                    self.rewards[s, a, sp] = 1 - 0.5 * a

    def initialize_opNAC(self, behaviour_policy, n, K):
        self.pi_t = np.ones((self.num_action, self.num_state)) / self.num_action  # pi(a|s) = self.pi_t(a,s)
        self.theta_t = np.zeros((self.num_action, self.num_state))
        self.Q_t = np.zeros((self.num_state, self.num_action))  # Q_t: output of Q-trace at iteration t
        self.behaviour_policy = behaviour_policy
        self.n = n  # number of steps of the TD
        self.K = K
        self.S0 = random.randrange(0, self.num_state)
        self.A0 = np.random.choice(self.actions, p=self.behaviour_policy[:, self.S0])
        self.latest_state = self.S0
        self.latest_action = self.A0
        self.batch_data_generator()
        self.evaluate_policy()

    # Policy Iteration:

    def optimum_policy_finder(self):
        # V = np.zeros((self.num_state, 1))  # Initialize Value function vector : [0,0,0...0]
        pi = np.ones((self.num_action, self.num_state)) / self.num_action  # Policy Initialization
        policy_stable = False
        while not policy_stable:
            V = self.evaluate_policy_2(pi)  # Policy Evaluation step
            pi, policy_stable = self.improve_policy(V, pi)  # Policy Iteration step
        self.optimal_policy = pi
        self.V_optimal = np.dot(self.mu, V)[0][0]

    def evaluate_policy_2(self, pi):
        P_pi = np.zeros((self.num_state, self.num_state))
        R_pi = np.zeros((self.num_state, 1))
        for s in self.states:
            for a in self.actions:
                for sp in self.states:
                    P_pi[s, sp] = P_pi[s, sp] + self.transitions[a, s, sp] * pi[a, s]
                    R_pi[s] = R_pi[s] + self.rewards[s, a, sp] * self.transitions[a, s, sp] * pi[a, s]
        return np.matmul(np.linalg.inv(np.identity(self.num_state) - self.gamma * P_pi), R_pi)

    def improve_policy(self, V, pi):
        policy_stable = True  # If policy_stable == True : Policy need not be updated anymore
        for s in self.states:
            old = pi[:, s].copy()
            self.choose_best_action(V, pi, s)
            if not np.array_equal(pi[:, s], old):
                policy_stable = False
        return pi, policy_stable

    def choose_best_action(self, V, pi, s):
        q = np.empty((self.num_action, 1), dtype=float)
        for a in self.actions:
            pi[a][s] = 0
            transitions = np.reshape(self.transitions[a][s][:], (-1, 1))
            rewards = np.reshape(self.rewards[s][a][:], (-1, 1))
            q[a] = np.sum(np.multiply(transitions, rewards) + self.gamma * np.multiply(transitions, V))
        action = np.argmax(q)  # Choose greedy action
        pi[action][s] = 1  # Update Policy

    # Off-policy NAC

    def batch_data_generator(self):
        Batch_Data = np.zeros((self.K + self.n, 2), dtype=int)  # Batch_Data = [S_t, A_t; S_{t+1}, A_{t+1}, ...]
        S = self.latest_state
        A = self.latest_action

        for b in range(self.K + self.n):
            Sp = int(np.random.choice(self.states, p=self.transitions[A, S, :]))
            Ap = int(np.random.choice(self.actions, p=self.behaviour_policy[:, Sp]))
            Batch_Data[b, :] = np.array([Sp, Ap])
            S = Sp
            A = Ap

        self.latest_state = Sp
        self.latest_action = Ap
        self.batch_data = Batch_Data

    def Delta_k_calculator(self, k):
        Delta_k = np.zeros(self.n)
        for i in range(k, k + self.n):
            S_i = self.batch_data[i, 0]
            A_i = self.batch_data[i, 1]
            S_i1 = self.batch_data[i + 1, 0]
            A_i1 = self.batch_data[i + 1, 1]
            rho = np.min([self.rho_bar, self.pi_t[A_i1, S_i1] / self.behaviour_policy[A_i1, S_i1]])
            Delta_k[i - k] = self.rewards[S_i, A_i, S_i1] + self.gamma * rho * self.Q_t[S_i1, A_i1] - self.Q_t[S_i, A_i]
        return Delta_k

    def evaluate_policy(self):
        P_pi = np.zeros((self.num_state, self.num_state))
        R_pi = np.zeros((self.num_state, 1))
        for s in self.states:
            for a in self.actions:
                for sp in self.states:
                    P_pi[s, sp] = P_pi[s, sp] + self.transitions[a, s, sp] * self.pi_t[a, s]
                    R_pi[s] = R_pi[s] + self.rewards[s, a, sp] * self.transitions[a, s, sp] * self.pi_t[a, s]
        V = np.matmul(np.linalg.inv(np.identity(self.num_state) - self.gamma * P_pi), R_pi)

        self.V_pi_t = np.dot(self.mu, V)[0][0]
        self.Q_pi_t = np.zeros((self.num_state, self.num_action))

        for s in self.states:
            for a in self.actions:
                for sp in self.states:
                    self.Q_pi_t[s, a] = self.Q_pi_t[s, a] + self.transitions[a, s, sp] * (
                                self.rewards[s, a, sp] + self.gamma * V[sp])

    def critic(self, K):
        for k in range(K):
            Delta_k = self.Delta_k_calculator(k)
            Delta_Q = 0.0
            for i in range(k, k + self.n):
                c_pi_mult = 1.0
                for j in range(k + 1, i + 1):
                    S_j = self.batch_data[j, 0]
                    A_j = self.batch_data[j, 1]
                    c_pi_mult = c_pi_mult * np.min([self.c_bar, self.pi_t[A_j, S_j] / self.behaviour_policy[A_j, S_j]])
                Delta_Q = Delta_Q + (self.gamma ** (i - k)) * c_pi_mult * Delta_k[i - k]

            S_k = self.batch_data[k, 0]
            A_k = self.batch_data[k, 1]

            self.Q_t[S_k, A_k] = self.Q_t[S_k, A_k] + self.alpha * Delta_Q

    def actor(self):
        self.theta_t = self.theta_t + self.beta * self.Q_t.transpose()
        self.pi_t = softmax(self.theta_t, axis=0)

    def ploter(self):
        fig, ax1 = plt.subplots()
        plt.rcParams.update({"text.usetex": True, "font.serif": ["Computer Modern Roman"], 'font.size': 22})
        ax1.set_ylabel(r'$V^{\pi^*}(\nu) - V^{\pi_t}(\nu)$', rotation=90)
        ax1.set_xlabel('Number of iterations')
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(36)

        for loop in range(self.num_loops):
            color = np.random.random(3)
            ax1.plot(self.evaluation_points, self.convergence_Data[loop, :], '--' , label='Sample path: %i' % (loop + 1),
                     linewidth=2,
                     color=color)

        color = np.random.random(3)
        ax1.plot(self.evaluation_points, np.mean(self.convergence_Data, axis=0), label='Average', linewidth=4,
                 color=color)
        plt.show()

    def variable_saver(self, file_name):
        f = open(file_name, 'wb')
        pickle.dump([self.convergence_Data, self.error_Data], f)
        f.close()

    def off_policy_NAC(self, num_TD_step, K, T, num_loops, evaluation_points):
        self.convergence_Data = np.empty((num_loops, len(evaluation_points)), dtype=float)
        self.error_Data = np.empty((num_loops, len(evaluation_points)), dtype=float)
        self.iterations = T
        self.num_loops = num_loops
        self.evaluation_points = evaluation_points
        for loop in range(num_loops):
            self.initialize_opNAC(np.ones((self.num_action, self.num_state)) / self.num_action, num_TD_step, K)
            print(loop)
            d=0

            for t in range(T):
                self.critic(K)
                if t in self.evaluation_points:
                    self.error_Data[loop, d] = max(np.abs((self.Q_t - self.Q_pi_t).min()), np.abs((self.Q_t - self.Q_pi_t).max()))
                self.actor()
                if t in self.evaluation_points:
                    self.evaluate_policy()
                    self.convergence_Data[loop, d] = self.V_optimal - self.V_pi_t
                    d=d+1
                self.batch_data_generator()
            print("Loop = ", loop, "pi = ", self.pi_t)

        print("optimal policy = ", self.optimal_policy)

        self.ploter()
        self.variable_saver('divergence_data.pckl')


MDP = MarkovDP(5, 3, 0.9, 3, 1, 0.05, 0.1)

MDP.initialize_opNAC(np.ones((MDP.num_action, MDP.num_state)) / MDP.num_action, 6,
                     1000)
MDP.circle_MDP_initialization()
MDP.optimum_policy_finder()
print("optimal policy = ", MDP.optimal_policy)

evaluation_points = list(range(100))

MDP.off_policy_NAC(6, 1000, 100, 6, evaluation_points)
