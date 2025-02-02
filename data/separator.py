import pandas as pd
data = pd.read_csv('creditcard.csv')
part = data.sample(frac=0.1)
part.to_csv('creditcard_part.csv', index=False)