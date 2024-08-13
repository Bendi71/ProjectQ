import numpy as np
from itertools import product


class CRR(object):
    def __init__(self, asset_price, risk_free_rate, volatility, time_to_maturity, steps):
        self.S0 = asset_price
        self.r = risk_free_rate
        self.sigma = volatility
        self.T = time_to_maturity
        self.N = steps
        self.dt = self.T / self.N
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.q = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.dt)

    def _build_tree(self):
        self.St = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.N + 1):
            for j in range(i + 1):
                self.St[j, i] = self.S0 * (self.u ** (i - j)) * (self.d ** j)
        return self.St

    def _build_tree_big(self):
        eps_map = {
            'u': self.u,
            'd': self.d,
        }

        for combo in product(['u', 'd'], repeat=self.N):
            S_route = [self.S0]
            for move in combo:
                S_route.append(S_route[-1] * eps_map[move])
            yield S_route

    def _build_tree_fast(self):
        indices = np.arange(2 ** self.N)
        ups = np.array([(indices >> i) & 1 for i in range(self.N)]).T
        S_routes = np.cumprod(np.where(ups, self.u, self.d), axis=1) * self.S0
        S_routes = np.hstack([np.full((2 ** self.N, 1), self.S0), S_routes])
        return S_routes

    def _build_tree_last(self):
        ST = np.zeros((self.N + 1))
        ST[0] = self.S0 * self.d ** self.N
        for i in range(1, self.N + 1):
            ST[i] = ST[i - 1] * self.u / self.d
        return ST

    def _calculate_probability(self):
        q_prob = np.zeros(2 ** self.N)
        for i in range(2 ** self.N):
            binary_rep = np.array([(i >> j) & 1 for j in range(self.N)])
            q_prob[i] = (self.q ** np.sum(binary_rep) * (1 - self.q) ** (self.N - np.sum(binary_rep)))
        return q_prob

    def calculate_option_price(self, payoff):
        raise NotImplementedError

    def call(self):
        raise NotImplementedError

    def put(self):
        raise NotImplementedError

    def visualize_tree(self):
        import matplotlib.pyplot as plt
        self._build_tree()
        fig, ax = plt.subplots()
        for i in range(self.N + 1):
            for j in range(i + 1):
                ax.text(i, j - i / 2, f'{self.St[j, i]:.2f}', ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='black'))
                if i < self.N:
                    ax.plot([i, i + 1], [j - i / 2, j - (i + 1) / 2], 'k-', lw=1)
                    ax.plot([i, i + 1], [j - i / 2, j + 1 - (i + 1) / 2], 'k-', lw=1)
        ax.set_xlim(-0.5, self.N + 0.5)
        ax.set_ylim(-self.N / 2 - 0.5, self.N / 2 + 0.5)
        ax.set_xticks(range(self.N + 1))
        ax.set_yticks(range(-self.N // 2, self.N // 2 + 1))
        ax.set_xlabel('Steps')
        ax.set_ylabel('Nodes')
        plt.gca().invert_yaxis()
        plt.show()
