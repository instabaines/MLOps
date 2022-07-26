import pandas as pd
from datetime import datetime
import sys
sys.path.append('..')

from Homework.batch import preprare_data,predict

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)
data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
]

columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
df = pd.DataFrame(data, columns=columns)
input_file = './test.parquet'
test = preprare_data(df,columns[:2])
test.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
)
dicts = test[columns[:2]].to_dict(orient='records')
    
y_pred=predict(dicts)
print (sum(y_pred))
assert len(test) == 2