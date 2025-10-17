"""Configuration module for the project."""

import yaml
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ProjectConfig:
    project_name: str
    description: str
    years: List[int]
    year: int
    train_fraction: float
    data_paths: Dict[str,str]
    output_paths: Dict[str,str]
    districts: List[str]
    model: Dict
    xgboost_params: Dict
    features: Dict
    data_quality: Dict
    logging: Dict
    deployment: Dict

def load_config(path='config/base.yaml') 
  """Load configuration from a YAML file."""
  with open(path, 'r') as f:
     cfg = yaml.safe_lad(f)
  return ProjectConfig(**cfg)
