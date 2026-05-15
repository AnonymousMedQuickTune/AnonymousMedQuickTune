"""
Portfolio preprocessing utilities for QuickTune conditional search spaces.

This module provides functions to enhance portfolio DataFrames with active flags
and handle inactive categorical parameters without modifying the portfolio format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import yaml

from src.utils.neps_conditional_patch import extract_conditions_from_yaml

logger = logging.getLogger(__name__)


def get_conditional_parameters(yaml_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Get mapping of conditional parameters to their conditions.
    
    Args:
        yaml_path: Path to pipeline space YAML file
        
    Returns:
        Dictionary mapping conditional parameter names to their condition info:
        {
            "densenet_type": {
                "parent": "model",
                "parent_value": "densenet"
            },
            ...
        }
    """
    try:
        conditions = extract_conditions_from_yaml(yaml_path)
        conditional_params = {}
        for cond in conditions:
            child = cond["child_param"]
            conditional_params[child] = {
                "parent": cond["parent_param"],
                "parent_value": cond["parent_value"]
            }
        return conditional_params
    except Exception as e:
        logger.warning(f"Could not extract conditions from {yaml_path}: {e}")
        return {}


def get_categorical_conditional_parameters(yaml_path: str) -> set:
    """
    Get set of categorical conditional parameter names.
    
    Args:
        yaml_path: Path to pipeline space YAML file
        
    Returns:
        Set of parameter names that are both categorical and conditional
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            pipeline_space = yaml.safe_load(f)
        
        categorical_conditional = set()
        for param_name, param_config in pipeline_space.items():
            if (isinstance(param_config, dict) and
                param_config.get("type") == "categorical" and
                "condition" in param_config):
                categorical_conditional.add(param_name)
        
        return categorical_conditional
    except Exception as e:
        logger.warning(f"Could not load pipeline space from {yaml_path}: {e}")
        return set()


def add_active_flags_to_dataframe(
    df: pd.DataFrame,
    conditional_params: Dict[str, Dict[str, Any]],
    parent_param: Optional[str] = None
) -> pd.DataFrame:
    """
    Add active flags for conditional parameters to DataFrame.
    
    For each conditional parameter, adds a binary flag indicating whether
    the parameter is active (1) or inactive (0) based on the parent parameter value.
    
    Args:
        df: DataFrame with portfolio configurations
        conditional_params: Dictionary mapping conditional parameter names to condition info
        parent_param: Optional parent parameter name to check (if None, uses condition info)
        
    Returns:
        DataFrame with additional active flag columns (e.g., "densenet_type_is_active")
    """
    df = df.copy()  # Don't modify original
    
    for cond_param, cond_info in conditional_params.items():
        if cond_param not in df.columns:
            # Parameter doesn't exist in this portfolio (might be from different search space)
            continue
        
        flag_name = f"{cond_param}_is_active"
        if flag_name in df.columns:
            # Flag already exists, skip
            continue
        
        # Get parent parameter and expected value
        parent = cond_info["parent"]
        parent_value = cond_info["parent_value"]
        
        if parent not in df.columns:
            logger.warning(f"Parent parameter '{parent}' not found in DataFrame for conditional parameter '{cond_param}'")
            continue
        
        # Check if parameter is active: parent must equal parent_value
        is_active = (df[parent] == parent_value).astype(int)
        df[flag_name] = is_active
        
        logger.debug(f"Added active flag '{flag_name}' for conditional parameter '{cond_param}'")
    
    return df


def handle_inactive_categorical_parameters(
    df: pd.DataFrame,
    categorical_conditional: set,
    inactive_value: str = "__inactive__"
) -> pd.DataFrame:
    """
    Replace None/NaN values in categorical conditional parameters with inactive marker.
    
    This function ensures type consistency: if a column contains booleans, all values
    (including the inactive marker) are converted to strings to avoid mixed types.
    
    Args:
        df: DataFrame with portfolio configurations
        categorical_conditional: Set of categorical conditional parameter names
        inactive_value: String value to use for inactive parameters (default: "__inactive__")
        
    Returns:
        DataFrame with inactive categorical parameters replaced
    """
    df = df.copy()  # Don't modify original
    
    for param_name in categorical_conditional:
        if param_name not in df.columns:
            continue
        
        # Check if column has mixed types (e.g., bool + str) or contains booleans
        col = df[param_name]
        non_null_values = col.dropna()
        
        if len(non_null_values) == 0:
            # All values are NaN, skip
            continue
        
        # Convert column to object dtype BEFORE assigning inactive values
        # This is necessary to avoid type conflicts when assigning strings to numeric columns
        if df[param_name].dtype != 'object':
            df[param_name] = df[param_name].astype('object')
        
        # Convert ALL values to strings to ensure type consistency for OneHotEncoder
        # This is necessary because sklearn's Encoder requires uniform types (all strings or all numbers)
        # Since we're adding '__inactive__' as a string, we need all values to be strings
        df[param_name] = df[param_name].apply(lambda x: str(x) if pd.notna(x) else x)
        logger.debug(f"Converted all values in '{param_name}' to strings for type consistency")
        
        # Replace None/NaN with inactive marker
        mask = df[param_name].isna()
        if mask.any():
            df.loc[mask, param_name] = inactive_value
            logger.debug(f"Replaced {mask.sum()} inactive values in '{param_name}' with '{inactive_value}'")
    
    return df


def preprocess_portfolio_for_quicktune(
    df: pd.DataFrame,
    pipeline_space_path: Optional[str] = None,
    add_active_flags: bool = True,
    handle_inactive_categorical: bool = True,
    inactive_categorical_value: str = "__inactive__"
) -> pd.DataFrame:
    """
    Preprocess portfolio DataFrame for QuickTune predictor training.
    
    This function enhances the portfolio DataFrame with:
    1. Active flags for conditional parameters (optional)
    2. Inactive categorical parameter handling (optional)
    
    The original portfolio format remains unchanged - enhancements are added dynamically.
    
    Args:
        df: Portfolio DataFrame (from portfolio.pipeline_df)
        pipeline_space_path: Path to pipeline space YAML file (required for conditional handling)
        add_active_flags: Whether to add active flags for conditional parameters
        handle_inactive_categorical: Whether to replace None/NaN in categorical conditional params
        inactive_categorical_value: Value to use for inactive categorical parameters
        
    Returns:
        Enhanced DataFrame ready for QuickTune predictor training
    """
    df = df.copy()
    
    if pipeline_space_path is None:
        logger.warning("No pipeline_space_path provided - skipping conditional parameter enhancements")
        return df
    
    pipeline_space_path = Path(pipeline_space_path)
    if not pipeline_space_path.exists():
        logger.warning(f"Pipeline space file not found: {pipeline_space_path} - skipping enhancements")
        return df
    
    # Add active flags for conditional parameters
    if add_active_flags:
        conditional_params = get_conditional_parameters(str(pipeline_space_path))
        if conditional_params:
            df = add_active_flags_to_dataframe(df, conditional_params)
            logger.info(f"Added active flags for {len(conditional_params)} conditional parameters")
        else:
            logger.debug("No conditional parameters found in pipeline space")
    
    # Handle inactive categorical parameters
    if handle_inactive_categorical:
        categorical_conditional = get_categorical_conditional_parameters(str(pipeline_space_path))
        if categorical_conditional:
            df = handle_inactive_categorical_parameters(
                df, 
                categorical_conditional,
                inactive_value=inactive_categorical_value
            )
            logger.info(f"Handled inactive values for {len(categorical_conditional)} categorical conditional parameters")
        else:
            logger.debug("No categorical conditional parameters found")
    
    return df
