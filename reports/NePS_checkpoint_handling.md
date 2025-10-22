# NePS Checkpoint Handling

This document describes how NePS handles crashes and restarts, particularly useful when GPU resources are lost due to 24-hour timeouts.

## Example Configuration

- **max_evaluations**: 3 (number of configurations)
- **inner_folds**: 2
- **outer_folds**: 5 

## Crash and Restart Behavior

### Scenario 1: Crash During Configuration

**Crash Location**: `outer_fold_1`, `config_1`, `inner_fold_1`  
**Restart Behavior**: `outer_fold_1`, `config_2`

**Crash Location**: `outer_fold_2`, `config_2`, `inner_fold_2`  
**Restart Behavior**: `outer_fold_2`, `config_3`

### Scenario 2: Crash at End of Configuration

**Crash Location**: `outer_fold_2`, `config_3`  
**Restart Behavior**: `outer_fold_3`, `config_1`

### Key Rules

> **Rule 1**: If NePS crashes within a configuration (regardless of completed inner folds), it restarts with the next configuration in the same outer fold.

> **Rule 2**: If NePS crashes in the last configuration of an outer fold, it restarts with the first configuration of the next outer fold.

## Dynamic Configuration Updates

### Scenario: Increasing max_evaluations

**Original Setting**: `max_evaluations = 3`  
**Updated Setting**: `max_evaluations = 4`  
**Last Crash**: `outer_fold_3`, `config_1`

**Restart Sequence**:
1. `outer_fold_1`, `config_4` (new configuration)
2. `outer_fold_2`, `config_4` (new configuration)  
3. `outer_fold_3`, `config_2` (continue from crash point)

### Key Rule for Dynamic Updates

> **Rule 3**: When increasing max_evaluations after a crash, NePS starts with `outer_fold_1` and catches up with new configurations for each outer fold until reaching the crash point, then continues normally.
