import numpy as np
class Company:
    def __init__(self, market_share, quality):
        self.MarketShare = market_share
        self.Quality = quality
        self.History = {'MarketShare': [self.MarketShare], 'Quality': [self.Quality]}
        self.entry_step = 0
        self.exit_step = None

    def MarketShare_process(self,Av_quality,Total_market, a):
        if Total_market < 1:
            self.MarketShare = (1 + a * (self.Quality - Av_quality) / Av_quality) * self.MarketShare
        else:
            self.MarketShare = (1 + a * (self.Quality - Av_quality) / Av_quality) * self.MarketShare / Total_market
        self.History['MarketShare'].append(self.MarketShare)

    def Quality_process(self, Min, Max):
        self.Quality = max(0,np.random.uniform(Min, Max) + self.Quality)
        self.History['Quality'].append(self.Quality)
