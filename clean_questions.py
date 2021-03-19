"""
clean questionnaire data
"""
import os
import sys
import numpy as np
import pandas as pd

def extract_qs(df):
    """extract questionnaire set out of 5 possible"""
    return df_qs