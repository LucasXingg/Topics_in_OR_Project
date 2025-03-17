import pandas as pd

class utils:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def getRoom(self, b):
        room = b.split('o')[1][0:-1]
        return int(room)

    def getDay(self, b):
        day = b.split('d')[1][0]
        return int(day)


    def cases(self, b):
        line = self.df[(self.df['Block'] == b)]
        return list(line['Beds'])


    def alloc(self, b, v):
        line = self.df[(self.df['Block'] == b) & (self.df['Service ID'] == v)]
        return len(line)


    def Di(self, b, w, s):
        line = self.df[(self.df['Block'] == b) & (self.df['Week Number'] == w) & (self.df['Surgeon ID'] == s)]
        return not line.empty  # Returns True if at least one row exists


    def getBs(self, p):
        b1, b2 = p
        return [b1, b2]