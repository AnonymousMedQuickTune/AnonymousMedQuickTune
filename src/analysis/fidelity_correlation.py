import re
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
import os

def parse_neps_output(file_path):
    config_losses = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by separator
    entries = content.split('-------------------------------------------------------------------------------')
    
    # First pass: collect all entries
    for entry in entries:
        if not entry.strip():
            continue
            
        loss_match = re.search(r'Loss: ([-\d.]+)', entry)
        config_match = re.search(r'Config ID: (\d+)_(\d+)', entry)
        
        if loss_match and config_match:
            loss = float(loss_match.group(1))
            base_config = config_match.group(1)
            fidelity = int(config_match.group(2))
            
            if base_config not in config_losses:
                config_losses[base_config] = {}
            
            config_losses[base_config][fidelity] = loss
    
    # Debug output
    print("\nDetailed configuration analysis:")
    print("--------------------------------")
    for config, fidelities in config_losses.items():
        fids = sorted(fidelities.keys())
        losses = [fidelities[f] for f in fids]
        print(f"Config {config}: Fidelities {fids}, Losses {losses}")
    
    # Count pairs
    fidelity_pairs = {(0,1): [], (1,2): [], (0,2): []}
    for config, fidelities in config_losses.items():
        fids = set(fidelities.keys())
        for pair in fidelity_pairs:
            if pair[0] in fids and pair[1] in fids:
                fidelity_pairs[pair].append(config)
    
    print("\nPair analysis:")
    print("-------------")
    for pair, configs in fidelity_pairs.items():
        print(f"Fidelity pair {pair} found in configs: {configs} (total: {len(configs)})")
    
    return config_losses

def analyze_correlations(config_losses):
    # Prepare data for correlation analysis
    fidelities = set()
    for configs in config_losses.values():
        fidelities.update(configs.keys())
    fidelities = sorted(list(fidelities))
    
    if len(fidelities) < 2:
        print(f"Found only {len(fidelities)} fidelities. At least 2 fidelities are required for correlation analysis.")
        return []
    
    # For each pair of fidelities, calculate correlation
    results = []
    for i in range(len(fidelities)):
        for j in range(i + 1, len(fidelities)):
            fid1, fid2 = fidelities[i], fidelities[j]
            
            # Collect pairs of losses
            pairs = []
            for config in config_losses.values():
                if fid1 in config and fid2 in config:
                    pairs.append((config[fid1], config[fid2]))
            
            if len(pairs) < 2:
                print(f"Warning: Skipping fidelity pair ({fid1}, {fid2}) - found only {len(pairs)} pairs, need at least 2")
                continue
                
            x, y = zip(*pairs)
            correlation, p_value = stats.pearsonr(x, y)
            spearman_corr, spearman_p = stats.spearmanr(x, y)
            
            results.append({
                'fidelity_pair': (fid1, fid2),
                'num_pairs': len(pairs),
                'pearson_correlation': correlation,
                'pearson_p_value': p_value,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'pairs': pairs
            })
    
    return results

def plot_correlations(results, output_dir='correlation_results'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text results
    with open(os.path.join(output_dir, 'correlation_results.txt'), 'w') as f:
        f.write(f"Analyzing fidelity correlations\n")
        f.write("--------------------------------\n\n")
        
        f.write("Detailed configuration analysis:\n")
        f.write("--------------------------------\n")
        for config, fidelities in config_losses.items():
            fids = sorted(fidelities.keys())
            losses = [fidelities[f] for f in fids]
            f.write(f"Config {config}: Fidelities {fids}, Losses {losses}\n")
        
        f.write("\nPair analysis:\n")
        f.write("-------------\n")
        for pair, configs in fidelity_pairs.items():
            f.write(f"Fidelity pair {pair} found in configs: {configs} (total: {len(configs)})\n")
        
        for result in results:
            f.write(f"\nAnalysis for fidelities {result['fidelity_pair']}:\n")
            f.write(f"Number of configuration pairs: {result['num_pairs']}\n")
            f.write(f"Pearson correlation: {result['pearson_correlation']:.3f} (p-value: {result['pearson_p_value']:.3f})\n")
            f.write(f"Spearman correlation: {result['spearman_correlation']:.3f} (p-value: {result['spearman_p_value']:.3f})\n")

# Main execution
if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/gist/test_budget-corr_0/seed_42/NePS_output/all_losses_and_configs.txt"
    
    config_losses = parse_neps_output(file_path)
    
    if not config_losses:
        print("No data found in the input file.")
        sys.exit(1)
        
    results = analyze_correlations(config_losses)
    
    if not results:
        print("No valid correlation results found.")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(file_path), 'correlation_results')
    
    # Get fidelity pairs from parse_neps_output
    fidelity_pairs = {(0,1): [], (1,2): [], (0,2): []}
    for config, fidelities in config_losses.items():
        fids = set(fidelities.keys())
        for pair in fidelity_pairs:
            if pair[0] in fids and pair[1] in fids:
                fidelity_pairs[pair].append(config)
    
    plot_correlations(results, output_dir)
    print(f"\nResults have been saved in: {output_dir}/")