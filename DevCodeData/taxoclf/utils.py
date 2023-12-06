# usr/bin/env python
import pandas as pd
from descriptastorus.descriptors import rdNormalizedDescriptors
import numpy as np


# Calculate Molecular descriptor   
def smile_to_md(smile):
    result = []
    gen = rdNormalizedDescriptors.RDKit2DNormalized()
    md = gen.process(smile)
    t = md[1:]
    result = np.asarray(t)
    nan_index = np.argwhere(np.isnan(result))
    for i in nan_index:
        result[i[0]] = 0  
    return result