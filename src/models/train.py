import pandas as pd
import numpy as np
import logging 
import joblib
from pathlib import Path
from sklearn.ensemble import RandmForeestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from src.common.config import load_config
from src.common.io_utils import read_parquet

logger = logging.getLogger(__name__)

