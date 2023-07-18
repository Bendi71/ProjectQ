import numpy as np
class Barrier_option:
    def __init__(self, asset_price, strike_price, upper_strike, lower_strike, riskfreerate, volatility,
                 time_toexpiariton, steps):
        self.S = asset_price
        self.K = strike_price
        self.Kup = upper_strike
        self.Kdo = lower_strike
        self.r = riskfreerate
        self.sig = volatility
        self.T = time_toexpiariton
        self.N = steps
        self.dt = self.T / self.N
        self.up_prob = np.exp(self.sig * np.sqrt(self.dt))
        self.do_prob = 1 / self.up_prob
        self.q_prob = (np.exp(self.r * self.dt) - self.do_prob) / (self.up_prob - self.do_prob)

    def upout_call(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(ST - self.K, 0)
        g[ST >= self.Kup] = 0
        for i in np.arange(self.N - 1, -1, -1):
            g[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * g[1:i + 2] + (1 - self.q_prob) * g[:i + 1])
            g = g[:-1]
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            g[ST >= self.Kup] = 0
        return g

    def upout_put(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(self.K - ST, 0)
        g[ST >= self.Kup] = 0
        for i in np.arange(self.N - 1, -1, -1):
            g[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * g[1:i + 2] + (1 - self.q_prob) * g[:i + 1])
            g = g[:-1]
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            g[ST >= self.Kup] = 0
        return g

    def downout_call(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(ST - self.K, 0)
        g[ST <= self.Kdo] = 0
        for i in np.arange(self.N - 1, -1, -1):
            g[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * g[1:i + 2] + (1 - self.q_prob) * g[:i + 1])
            g = g[:-1]
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            g[ST <= self.Kdo] = 0
        return g

    def downout_put(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(self.K - ST, 0)
        g[ST <= self.Kdo] = 0
        for i in np.arange(self.N - 1, -1, -1):
            g[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * g[1:i + 2] + (1 - self.q_prob) * g[:i + 1])
            g = g[:-1]
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            g[ST <= self.Kdo] = 0
        return g

    def upin_call(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(ST - self.K, 0)
        C = g.copy()
        g[ST < self.Kup] = 0
        for i in np.arange(self.N - 1, -1, -1):
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            for j in range(0, i + 1):
                if ST[j] >= self.Kup:
                    g[j] = C[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]

    def upin_put(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(self.K - ST, 0)
        C = g.copy()
        g[ST < self.Kup] = 0
        for i in np.arange(self.N - 1, -1, -1):
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            for j in range(0, i + 1):
                if ST[j] >= self.Kup:
                    g[j] = C[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]

    def downin_call(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(ST - self.K, 0)
        C = g.copy()
        g[ST > self.Kdo] = 0
        for i in np.arange(self.N - 1, -1, -1):
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            for j in range(0, i + 1):
                if ST[j] <= self.Kdo:
                    g[j] = C[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]

    def downin_put(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g = np.maximum(self.K - ST, 0)
        C = g.copy()
        g[ST > self.Kdo] = 0
        for i in np.arange(self.N - 1, -1, -1):
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            for j in range(0, i + 1):
                if ST[j] <= self.Kdo:
                    g[j] = C[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]
class American_option:
    def __init__(self, asset_price, strike_price, riskfreerate, volatility,
                 time_toexpiariton, steps):
        self.S = asset_price
        self.K = strike_price
        self.r = riskfreerate
        self.sig = volatility
        self.T = time_toexpiariton
        self.N = steps
        self.dt = self.T / self.N
        self.up_prob = np.exp(self.sig * np.sqrt(self.dt))
        self.do_prob = 1 / self.up_prob
        self.q_prob = (np.exp(self.r * self.dt) - self.do_prob) / (self.up_prob - self.do_prob)

    def am_call(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g=np.maximum(ST-self.K,0)
        C=g.copy()
        for i in np.arange(self.N-1,-1,-1):
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            intrin=np.maximum(ST-self.K,0)
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            for j in range(0,i+1):
                if intrin[j]>C[j]:
                    g[j]=intrin[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]

    def am_put(self):
        ST = self.S * self.do_prob ** (np.arange(self.N, -1, -1)) * self.up_prob ** (np.arange(0, self.N + 1, 1))
        g=np.maximum(self.K-ST,0)
        C=g.copy()
        for i in np.arange(self.N-1,-1,-1):
            ST = self.S * self.do_prob ** (np.arange(i, -1, -1)) * self.up_prob ** (np.arange(0, i + 1, 1))
            intrin=np.maximum(self.K-ST,0)
            C[:i + 1] = np.exp(-self.r * self.dt) * (self.q_prob * C[1:i + 2] + (1 - self.q_prob) * C[:i + 1])
            for j in range(0,i+1):
                if intrin[j]>C[j]:
                    g[j]=intrin[j]
                else:
                    g[j] = np.exp(-self.r * self.dt) * (self.q_prob * g[j + 1] + (1 - self.q_prob) * g[j])
        return g[0]

print(American_option(100,120,.05,.2,1,20).am_call())