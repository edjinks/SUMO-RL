import pandas as pd
import itertools
from main import stringHelper

rows = set([x for x in itertools.combinations(["O", "1", "X"]*4, 4)])
rows = [stringHelper(r) for r in rows]

columns = set([x for x in itertools.combinations(["O", "1"]*4, 4)])
columns = [stringHelper(r) for r in columns]

columns = set([i for i in itertools.permutations(["0","1","2","3","4"]*3, 3)])



df = pd.DataFrame(0, columns=columns, index=rows)

s = df['OO1O']

ret = s.sample().index

print(ret[0])