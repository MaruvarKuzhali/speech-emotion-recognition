import sys

import turicreate as tc
load = tc.load_model("model_name")
data="audiotrack"
y = load.predict(data)
print(y)
