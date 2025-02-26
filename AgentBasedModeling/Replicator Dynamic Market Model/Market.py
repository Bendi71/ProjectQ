import numpy as np

from Company import Company


class Market:
    def __init__(self) -> None:
        self.Companies = []
        self.Av_Quality = 0
        self.Total_Market = 0
        self.Num_Companies = 0
        self.sim_data = []
        self.sim_all_companies = []

    def set_parameters(self, a, min, max, ExitThreshold, HetQuality, ProbEntry, InitShare, InitQuality,
                       NewShare=None) -> None:
        self.Selection_Pressure = a
        self.MinQuality = min
        self.MaxQuality = max
        self.ExitThreshold = ExitThreshold
        self.HetQuality = HetQuality
        self.ProbEntry = ProbEntry
        self.InitShare = InitShare
        self.NewShare = NewShare
        self.InitQuality = InitQuality

    def compile(self, steps: int, num_simulations: int, num_companies: int) -> None:
        self.steps = steps
        self.num_simulations = num_simulations
        self.num_companies = num_companies

    def setup_companies(self) -> None:
        self.Companies = [Company(self.InitShare, self.InitQuality) for _ in range(self.num_companies)]
        for comp in self.Companies:
            comp.History = {'MarketShare': [], 'Quality': []}
            comp.entry_step = 0

    def new_entry(self, step) -> None:
        if np.random.uniform() < self.ProbEntry:
            rnd = np.random.uniform(-self.HetQuality, self.HetQuality)
            new_quality = (1 + rnd) * self.Av_Quality
            new_company = Company(self.NewShare, new_quality)
            new_company.History = {'MarketShare': [], 'Quality': []}
            new_company.entry_step = step
            self.Companies.append(new_company)
            self.all_companies.append(new_company)

    def dropout(self, step) -> None:
        remaining_companies = []
        for comp in self.Companies:
            if comp.MarketShare >= self.ExitThreshold:
                remaining_companies.append(comp)
            else:
                comp.History['MarketShare'].append(comp.MarketShare)
                comp.History['Quality'].append(comp.Quality)
                comp.exit_step = step
        self.Companies = remaining_companies

    def step(self) -> None:
        for sim in range(self.num_simulations):
            self.setup_companies()
            step_data = []
            self.all_companies = self.Companies.copy()
            for step in range(self.steps):
                self.Total_Market = np.sum([comp.MarketShare for comp in self.Companies])
                self.Av_Quality = np.sum(
                    [comp.MarketShare * comp.Quality for comp in self.Companies]) / self.Total_Market
                self.Num_Companies = len(self.Companies)
                step_data.append(self.save_data(step))
                for comp in self.Companies:
                    comp.MarketShare_process(self.Av_Quality, self.Total_Market, self.Selection_Pressure)
                    comp.Quality_process(self.MinQuality, self.MaxQuality)
                self.new_entry(step)
                self.dropout(step)
            self.sim_data.append(step_data)
            self.sim_all_companies.append(self.all_companies)

    def save_data(self, step) -> dict:
        market_data = {
            'Av_Quality': self.Av_Quality,
            'Total_Market': self.Total_Market,
            'Num_Companies': self.Num_Companies,
            'Step': step
        }
        return market_data

    def reshape_MS_data(self, all_companies: list, steps: int) -> np.ndarray:
        # Initialize the data structure with np.nan
        reshaped_data = np.full((len(all_companies), steps), np.nan)

        for idx, company in enumerate(all_companies):
            # Determine the range of steps for which the company has data
            start_step = company.entry_step
            end_step = company.exit_step if company.exit_step is not None else steps

            # Populate the known values for each company
            for step in range(start_step, end_step):
                history_index = step - start_step
                if history_index < len(company.History['MarketShare']):
                    reshaped_data[idx, step] = company.History['MarketShare'][history_index]
                # No need to fill np.nan as it's already the default value

        return reshaped_data

    def reshape_Q_data(self, all_companies: list, steps: int) -> np.ndarray:
        # Initialize the data structure with np.nan
        reshaped_data = np.full((len(all_companies), steps), np.nan)

        for idx, company in enumerate(all_companies):
            # Determine the range of steps for which the company has data
            start_step = company.entry_step
            end_step = company.exit_step if company.exit_step is not None else steps

            # Populate the known values for each company
            for step in range(start_step, end_step):
                history_index = step - start_step
                if history_index < len(company.History['Quality']):
                    reshaped_data[idx, step] = company.History['Quality'][history_index]
                # No need to fill np.nan as it's already the default value

        return reshaped_data
