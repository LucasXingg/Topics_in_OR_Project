import csv
import re

import matplotlib.pyplot as plt
import numpy as np

class ResultData:
    """
    A class to store and manage results data.
    """

    def __init__(self, model=None, filepath=None, df=None, utils=None):
        """
        Initialize the ResultData object.

        :param model: Optional Gurobi model to extract data from.
        :param filepath: Optional file path to load data from.
        :param df: DataFrame containing the data (required).
        :param utils: Utilities object for data manipulation (required).
        """
        if df is None or utils is None:
            raise ValueError("Both 'df' and 'utils' are required.")

        self.df = df
        self.utils = utils

        if filepath is not None:
            self.load(filepath)
        else:
            self.fromModel(model) if model is not None else {}

    def fromModel(self, model):
        """
        Extract results from a Gurobi model object.

        :param model: The Gurobi model object.
        """
        self.data = {}
        for var in model.getVars():
            self.data[var.varName] = var.X

    def save(self, filename):
        """
        Save the results data to a CSV file.

        :param filename: The name of the file to save the data to.
        """

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Variable', 'Value'])
            for var_name, value in self.data.items():
                writer.writerow([var_name, value])
    
    def load(self, filename):
        """
        Load results data from a CSV file.

        :param filename: The name of the file to load the data from.
        """
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            self.data = {rows[0]: float(rows[1]) for rows in reader}

    def print(self, ignore_0=True):
        """Prints all decision variables and their values."""

        # Iterate through all decision variables in the model
        print("\nDecision Variables:")
        for var_name, value in self.data.items():
            if ignore_0:
                if value > 0.5:
                    print(f"{var_name} = {value}")
            else:
                print(f"{var_name} = {value}")


    def plot(self, title=None, save=None):
        """
        Display a chart comparing current and permuted census based on the results data.
        """
        df_rearrange = self.df.copy()
        m = self.df.shape[0]

        rearrangements = []
        new_bed = np.zeros(m, dtype='int')

        for var_name, value in self.data.items():
            if var_name.startswith("x_p") and value > 0.5:  # Binary, so check if it's 1
                b1, b2 = re.findall(r'\([^\(\)]*\)', var_name)
                rearrangements.append([b1, b2])

        # Moving beds value from original dataset to new dataset according to resulting moves.
        for bs in rearrangements:
            b1 = bs[0]
            b2 = bs[1]
            bed, week_num = self.utils.casesWeek(b1)
            for i in range(4):
                line_idx = self.df[(self.df['Block'] == b2) & (self.df['Week Number'] == week_num[i])].index
                new_bed[line_idx[0]] = bed[i]

        df_rearrange['Beds'] = new_bed

        # Calculate day count for each week
        weekday_daycount_mat = np.zeros((4, 7))
        weekday_daycount_re_mat = np.zeros((4, 7))

        for week in range(1, 5):
            weekn = self.df['Week Number'] == week
            df_weekn = self.df[weekn]

            weekday_daycount = []
            for d in range(1, 8):
                day_count = df_weekn[df_weekn['Weekday'] == d]['Beds'].sum()
                weekday_daycount.append(day_count)

            weekday_daycount_mat[week-1, :] = weekday_daycount

            weekn_re = df_rearrange['Week Number'] == week
            df_weekn_re = df_rearrange[weekn_re]

            weekday_daycount_re = []
            for d in range(1, 8):
                day_count = df_weekn_re[df_weekn_re['Weekday'] == d]['Beds'].sum()
                weekday_daycount_re.append(day_count)

            weekday_daycount_re_mat[week-1, :] = weekday_daycount_re

        weekday_daycount_avg = np.zeros(7)
        weekday_daycount_re_avg = np.zeros(7)

        for weekday in range(7):
            weekday_daycount_avg[weekday] = weekday_daycount_mat[:, weekday].mean()
            weekday_daycount_re_avg[weekday] = weekday_daycount_re_mat[:, weekday].mean()

        weekday_daycount_avg_tmp = weekday_daycount_avg.copy()
        weekday_daycount_re_avg_tmp = weekday_daycount_re_avg.copy()

        weekday_daycount_avg = np.concatenate(([weekday_daycount_avg_tmp[-1]], weekday_daycount_avg_tmp[:-1]))
        weekday_daycount_re_avg = np.concatenate(([weekday_daycount_re_avg_tmp[-1]], weekday_daycount_re_avg_tmp[:-1]))

        # Define labels for better readability
        day_labels = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        # Plot using Matplotlib
        plt.figure(figsize=(8, 8))
        plt.plot(range(7), weekday_daycount_avg, label='Current census')
        plt.plot(range(7), weekday_daycount_re_avg, label='Permuted census')

        # Set labels
        plt.xticks(ticks=range(7), labels=day_labels, rotation=45)
        plt.xlabel("Day of the Week")
        plt.ylabel("Beds")
        plt.ylim(250, 350)
        plt.legend()
        plt.tight_layout()

        if title:
            plt.title(title)

        if save:
            plt.savefig(save, facecolor='white', dpi=300)

        # Show plot
        plt.show()


