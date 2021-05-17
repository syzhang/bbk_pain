"""
feature engineering and transform
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from connectivity_mat import load_connectivity



qs_id = True
idp_id = True
conn_id = True

df = load_connectivity(task_name='paintype_all', conn_type='fullcorr_100',add_questionnaire=qs_id, add_idp=idp_id, add_conn=conn_id)

print(df.shape)