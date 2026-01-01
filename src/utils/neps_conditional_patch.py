"""
Monkey-patch for NePS to support conditional hyperparameters.

This module patches NePS's optimizer functions (random_search, BO, IFBO) to
respect conditional hyperparameters by validating and resampling invalid
configurations.

Usage:
    from src.utils.neps_conditional_patch import patch_neps_for_conditionals
    
    # Before calling neps.run()
    patch_neps_for_conditionals(pipeline_space, yaml_path)
    
    # Then call neps.run() as usual
    neps.run(pipeline_space=pipeline_space, ...)
"""

import yaml
from typing import Dict, Any, List, Callable, Optional
from ConfigSpace import ConfigurationSpace, EqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter
)

# Store original NePS functions before patching
_original_functions = {}
_conditional_configspace: Optional[ConfigurationSpace] = None


def extract_conditions_from_yaml(yaml_path: str) -> List[Dict[str, Any]]:
    """
    Extract conditional hyperparameter information from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        List of condition dictionaries with keys: child_param, parent_param, parent_value
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    conditions = []
    for param_name, param_config in config.items():
        if "condition" in param_config:
            condition_config = param_config["condition"]
            parent_param_name = condition_config.get("parent")
            parent_value = condition_config.get("value")
            
            if parent_param_name is None or parent_value is None:
                continue
            
            conditions.append({
                "child_param": param_name,
                "parent_param": parent_param_name,
                "parent_value": parent_value
            })
    
    return conditions


def create_configspace_with_conditions(
    pipeline_space: Dict[str, Any],
    conditions: List[Dict[str, Any]]
) -> ConfigurationSpace:
    """
    Create a ConfigurationSpace with all hyperparameters and conditions.
    
    Args:
        pipeline_space: NePS pipeline space dictionary
        conditions: List of condition dictionaries
        
    Returns:
        ConfigurationSpace object with conditions applied
    """
    import neps
    
    cs = ConfigurationSpace()
    cs_hyperparams = {}
    
    # First pass: Create all ConfigSpace hyperparameters from NePS parameters
    for key, neps_param in pipeline_space.items():
        if isinstance(neps_param, neps.Categorical):
            cs_param = CategoricalHyperparameter(
                name=key,
                choices=neps_param.choices
            )
        elif isinstance(neps_param, neps.Integer):
            cs_param = UniformIntegerHyperparameter(
                name=key,
                lower=neps_param.lower,
                upper=neps_param.upper,
                log=neps_param.log if hasattr(neps_param, 'log') else False
            )
        elif isinstance(neps_param, neps.Float):
            cs_param = UniformFloatHyperparameter(
                name=key,
                lower=neps_param.lower,
                upper=neps_param.upper,
                log=neps_param.log if hasattr(neps_param, 'log') else False
            )
        else:
            continue
        
        cs_hyperparams[key] = cs_param
        cs.add_hyperparameter(cs_param)
    
    # Second pass: Add all conditions to ConfigurationSpace
    for condition_info in conditions:
        child_name = condition_info["child_param"]
        parent_name = condition_info["parent_param"]
        parent_value = condition_info["parent_value"]
        
        if child_name not in cs_hyperparams or parent_name not in cs_hyperparams:
            continue
        
        child_cs_param = cs_hyperparams[child_name]
        parent_cs_param = cs_hyperparams[parent_name]
        
        condition = EqualsCondition(child_cs_param, parent_cs_param, parent_value)
        cs.add_condition(condition)
    
    return cs


def validate_config_against_conditions(config: Dict[str, Any], conditions: List[Dict[str, Any]]) -> bool:
    """
    Validate that a configuration respects all conditions.
    
    Args:
        config: Configuration dictionary to validate
        conditions: List of condition dictionaries
        
    Returns:
        True if configuration is valid, False otherwise
    """
    for condition in conditions:
        parent_param = condition["parent_param"]
        parent_value = condition["parent_value"]
        child_param = condition["child_param"]
        
        # Check if parent has the required value
        if parent_param in config:
            if config[parent_param] == parent_value:
                # Parent condition is met, child should be present
                if child_param not in config:
                    return False
            else:
                # Parent condition is not met, child should NOT be present
                if child_param in config:
                    return False
    
    return True


def filter_config_by_conditions(config: Dict[str, Any], conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filter a configuration to remove parameters that don't match their conditions.
    
    Args:
        config: Configuration dictionary to filter
        conditions: List of condition dictionaries
        
    Returns:
        Filtered configuration dictionary
    """
    filtered_config = config.copy()
    
    for condition in conditions:
        parent_param = condition["parent_param"]
        parent_value = condition["parent_value"]
        child_param = condition["child_param"]
        
        # If parent condition is not met, remove child parameter
        if parent_param in filtered_config:
            if filtered_config[parent_param] != parent_value:
                # Parent condition not met, remove child
                filtered_config.pop(child_param, None)
    
    # Convert numpy types to native Python types
    return _convert_to_native_types(filtered_config)


def patch_neps_for_conditionals(
    pipeline_space: Dict[str, Any],
    yaml_path: str
) -> None:
    """
    Patch NePS to respect conditional hyperparameters.
    
    This function monkey-patches NePS's optimizer functions (random_search,
    BO, IFBO) to validate and resample configurations that don't respect
    conditions. This ensures that all optimizers only use valid configurations.
    
    Args:
        pipeline_space: NePS pipeline space dictionary
        yaml_path: Path to YAML file with conditional hyperparameters
        
    Note:
        This is a monkey-patch that modifies NePS's internal behavior.
        It should be called BEFORE neps.run().
    """
    global _conditional_configspace
    
    # Extract conditions from YAML
    conditions = extract_conditions_from_yaml(yaml_path)
    
    if len(conditions) == 0:
        print("[NePS Conditional Patch] No conditions found in YAML, skipping patch.")
        return
    
    print(f"\n[NePS Conditional Patch] Found {len(conditions)} conditions to apply.")
    
    # Create ConfigurationSpace with conditions for validation
    _conditional_configspace = create_configspace_with_conditions(pipeline_space, conditions)
    print(f"[NePS Conditional Patch] Created ConfigurationSpace with {len(_conditional_configspace.get_hyperparameters())} hyperparameters and {len(_conditional_configspace.get_conditions())} conditions.")
    
    # Store conditions globally for use in patches
    _original_functions['conditions'] = conditions
    _original_functions['configspace'] = _conditional_configspace
    
    # Patch optimizer functions
    try:
        import neps.optimizers.algorithms as algo
        
        # CRITICAL: NePS uses PredefinedOptimizers dict to get optimizer functions
        # We need to patch both the function AND the dict entry
        optimizers_to_patch = ['random_search', 'bayesian_optimization', 'ifbo']
        
        for opt_name in optimizers_to_patch:
            if hasattr(algo, opt_name):
                original_func = getattr(algo, opt_name)
                _original_functions[opt_name] = original_func
                
                # Create wrapped function
                wrapped_func = _create_conditional_wrapper(
                    original_func,
                    conditions,
                    opt_name
                )
                
                # Patch the function directly
                setattr(algo, opt_name, wrapped_func)
                
                # CRITICAL: Also patch PredefinedOptimizers dict
                # This is where NePS actually gets the optimizer from
                if hasattr(algo, 'PredefinedOptimizers'):
                    if opt_name in algo.PredefinedOptimizers:
                        algo.PredefinedOptimizers[opt_name] = wrapped_func
                        print(f"[NePS Conditional Patch] Patched {opt_name} optimizer (function + PredefinedOptimizers dict).")
                    else:
                        print(f"[NePS Conditional Patch] Warning: {opt_name} not found in PredefinedOptimizers dict.")
                else:
                    print(f"[NePS Conditional Patch] Patched {opt_name} optimizer (function only).")
        
        print("[NePS Conditional Patch] All optimizers patched successfully.\n")
        
    except Exception as e:
        print(f"[NePS Conditional Patch] Error patching optimizers: {e}")
        import traceback
        traceback.print_exc()
        raise


def _create_conditional_wrapper(
    original_func: Callable,
    conditions: List[Dict[str, Any]],
    optimizer_name: str
) -> Callable:
    """
    Create a wrapper function that validates configurations against conditions.
    
    Args:
        original_func: Original optimizer function to wrap
        conditions: List of condition dictionaries
        optimizer_name: Name of the optimizer (for logging)
        
    Returns:
        Wrapped function that validates configurations
    """
    def wrapped_optimizer(pipeline_space, **kwargs):
        print(f"[Conditional Patch] Wrapper called for {optimizer_name}!")
        # Call original optimizer to get optimizer instance
        optimizer = original_func(pipeline_space, **kwargs)
        
        # Store original __call__ method
        original_call = optimizer.__call__
        
        def conditional_call(trials, budget_info=None, n=None):
            """
            Wrapped __call__ that validates and resamples invalid configurations.
            """
            print(f"[Conditional Patch] __call__ method called for {optimizer_name}!")
            # Call original __call__ to get sampled configs
            result = original_call(trials, budget_info, n)
            print(f"[Conditional Patch] Got result type: {type(result)}")
            
            # Handle both single config and list of configs
            if isinstance(result, list):
                validated_configs = []
                for config_obj in result:
                    config = config_obj.config.copy()  # Make a copy to avoid modifying original
                    
                    # Debug: Check if config is invalid
                    is_valid = validate_config_against_conditions(config, conditions)
                    if not is_valid:
                        print(f"[Conditional Patch] Invalid config detected in {optimizer_name}: {config}")
                    
                    # Validate and resample if needed
                    validated_config = _validate_and_resample_config(
                        config, conditions, optimizer_name, max_retries=50
                    )
                    if validated_config:
                        # Update config object with validated config
                        config_obj.config = validated_config
                        if not is_valid:
                            print(f"[Conditional Patch] Resampled to valid config: {validated_config}")
                        validated_configs.append(config_obj)
                    else:
                        # If we can't get a valid config after retries, filter it
                        # This ensures we at least have a valid config structure
                        filtered_config = filter_config_by_conditions(config, conditions)
                        print(f"[Conditional Patch] Could not resample, filtering config: {filtered_config}")
                        config_obj.config = filtered_config
                        validated_configs.append(config_obj)
                return validated_configs
            else:
                # Single config (SampledConfig object)
                print(f"[Conditional Patch] Result object type: {type(result)}")
                if hasattr(result, '__dict__'):
                    print(f"[Conditional Patch] Result object attributes: {list(result.__dict__.keys())}")
                if not hasattr(result, 'config'):
                    print(f"[Conditional Patch] ERROR: Result object has no 'config' attribute!")
                    print(f"[Conditional Patch] Result object: {result}")
                    return result
                config = result.config.copy()  # Make a copy
                
                # Debug: Check if config is invalid
                is_valid = validate_config_against_conditions(config, conditions)
                if not is_valid:
                    print(f"[Conditional Patch] Invalid config detected in {optimizer_name}: {config}")
                
                validated_config = _validate_and_resample_config(
                    config, conditions, optimizer_name, max_retries=50
                )
                if validated_config:
                    result.config = validated_config
                    if not is_valid:
                        print(f"[Conditional Patch] Resampled to valid config: {validated_config}")
                else:
                    # Filter invalid config
                    filtered_config = filter_config_by_conditions(config, conditions)
                    print(f"[Conditional Patch] Could not resample, filtering config: {filtered_config}")
                    result.config = filtered_config
                return result
        
        # CRITICAL: Instead of just patching __call__, we need to create a wrapper class
        # that intercepts ALL calls to the optimizer instance, because NePS might
        # call it in ways we don't expect, or the instance might be pickled/unpickled
        
        class ConditionalOptimizerWrapper:
            """Wrapper that makes the optimizer instance callable with conditional validation"""
            def __init__(self, original_optimizer):
                self._optimizer = original_optimizer
                # Copy all attributes from original optimizer to maintain compatibility
                for attr_name in dir(original_optimizer):
                    if not attr_name.startswith('_') or attr_name in ['__call__', '__class__']:
                        try:
                            attr_value = getattr(original_optimizer, attr_name)
                            if not callable(attr_value) or attr_name == '__call__':
                                setattr(self, attr_name, attr_value)
                        except:
                            pass
            
            def __call__(self, *args, **kwargs):
                """Intercept calls to the optimizer and validate/resample configs"""
                print(f"[Conditional Patch] ConditionalOptimizerWrapper.__call__ called for {optimizer_name}!")
                # Call original optimizer
                result = self._optimizer(*args, **kwargs)
                print(f"[Conditional Patch] Got result type: {type(result)}")
                
                # Handle both single config and list of configs
                if isinstance(result, list):
                    validated_configs = []
                    for config_obj in result:
                        config = config_obj.config.copy()
                        is_valid = validate_config_against_conditions(config, conditions)
                        if not is_valid:
                            print(f"[Conditional Patch] Invalid config detected in {optimizer_name}: {config}")
                        
                        validated_config = _validate_and_resample_config(
                            config, conditions, optimizer_name, max_retries=50
                        )
                        if validated_config:
                            config_obj.config = validated_config
                            if not is_valid:
                                print(f"[Conditional Patch] Resampled to valid config: {validated_config}")
                        else:
                            filtered_config = filter_config_by_conditions(config, conditions)
                            print(f"[Conditional Patch] Could not resample, filtering config: {filtered_config}")
                            config_obj.config = filtered_config
                        validated_configs.append(config_obj)
                    return validated_configs
                else:
                    # Single config
                    if not hasattr(result, 'config'):
                        print(f"[Conditional Patch] ERROR: Result object has no 'config' attribute!")
                        return result
                    config = result.config.copy()
                    is_valid = validate_config_against_conditions(config, conditions)
                    if not is_valid:
                        print(f"[Conditional Patch] Invalid config detected in {optimizer_name}: {config}")
                    
                    validated_config = _validate_and_resample_config(
                        config, conditions, optimizer_name, max_retries=50
                    )
                    if validated_config:
                        result.config = validated_config
                        if not is_valid:
                            print(f"[Conditional Patch] Resampled to valid config: {validated_config}")
                    else:
                        filtered_config = filter_config_by_conditions(config, conditions)
                        print(f"[Conditional Patch] Could not resample, filtering config: {filtered_config}")
                        result.config = filtered_config
                    return result
            
            def __getattr__(self, name):
                """Delegate all other attribute access to the original optimizer"""
                return getattr(self._optimizer, name)
        
        # Return wrapped optimizer instance
        wrapped_instance = ConditionalOptimizerWrapper(optimizer)
        print(f"[Conditional Patch] Created ConditionalOptimizerWrapper for {optimizer_name}.")
        return wrapped_instance
    
    return wrapped_optimizer


def _convert_to_native_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert numpy types and other special types to native Python types.
    This is necessary because OmegaConf and other libraries don't support numpy types.
    
    Args:
        config: Configuration dictionary that may contain numpy types
        
    Returns:
        Configuration dictionary with native Python types
    """
    import numpy as np
    
    converted = {}
    for key, value in config.items():
        # Convert numpy types to native Python types
        # Note: np.unicode_ and np.float_ were removed in NumPy 2.0
        if isinstance(value, np.str_):
            converted[key] = str(value)
        elif isinstance(value, np.bool_):
            converted[key] = bool(value)
        elif isinstance(value, np.integer):
            converted[key] = int(value)
        elif isinstance(value, np.floating):
            # np.float_ was removed in NumPy 2.0, but np.floating still works
            converted[key] = float(value)
        elif isinstance(value, np.ndarray):
            converted[key] = value.tolist()
        else:
            converted[key] = value
    return converted


def _validate_and_resample_config(
    config: Dict[str, Any],
    conditions: List[Dict[str, Any]],
    optimizer_name: str,
    max_retries: int = 50
) -> Optional[Dict[str, Any]]:
    """
    Validate a configuration and resample if invalid.
    
    Args:
        config: Configuration to validate
        conditions: List of condition dictionaries
        optimizer_name: Name of optimizer (for logging)
        max_retries: Maximum number of resampling attempts
        
    Returns:
        Valid configuration or None if max_retries exceeded
    """
    # First check if config is already valid
    if validate_config_against_conditions(config, conditions):
        return config
    
    # Try to resample using ConfigurationSpace
    global _conditional_configspace
    if _conditional_configspace is None:
        return None
    
    # Resample until we get a valid config
    for attempt in range(max_retries):
        try:
            # Sample a new configuration from ConfigurationSpace
            # This automatically respects conditions
            new_config = _conditional_configspace.sample_configuration()
            new_config_dict = dict(new_config)
            
            # Convert numpy types to native Python types
            new_config_dict = _convert_to_native_types(new_config_dict)
            
            # Double-check validation (should always be valid from ConfigurationSpace)
            if validate_config_against_conditions(new_config_dict, conditions):
                if attempt > 0:
                    print(f"[Conditional Patch] Successfully resampled config on attempt {attempt + 1}")
                return new_config_dict
        except Exception as e:
            # If sampling fails, continue to next attempt
            if attempt < 3:  # Only print first few errors to avoid spam
                print(f"[Conditional Patch] Resampling attempt {attempt + 1} failed: {e}")
            continue
    
    # If we can't get a valid config after max_retries, return None
    # The caller will filter the config instead
    print(f"[Conditional Patch] Failed to resample valid config after {max_retries} attempts, will filter instead")
    return None

