import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import math
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import os
import re
import config_loader as cfg_loader


import borne_plot as bp


def make_data(inv_df: dict):
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for methode, df in inv_df.items():
        try:
            df = df.assign(mapon=df.get('mapon', 'all'))
            chromosomes_à_exclure = ['chr2003', 'chr2016', 'chr2017']
            df = df[~df['chr'].isin(chromosomes_à_exclure)]

            for _, row in df.iterrows():
                all_data[row['iso']][methode][row['chr']][row['mapon']].append((row['start'], row['end']))

        except Exception as e:
            print(f"Erreur lors de la lecture de ({methode}): {str(e)}")

    return all_data


def pairwise_algn_to_multi(data_iso):    
    """
    Convertit un dictionnaire de détections d'inversions en pair-alignment
    vers un format multi-alignment avec positions fusionnées.
    
    Args:
        data: Dict {méthode: {chromosome: {isolat: [(start, end), ...]}}}
    
    Returns:
        Dict {méthode: {chromosome: [(start, end), ...]}}
    """
    result = {}
    
    for method, chromosomes in data_iso.items():
        result[method] = {}

        for chromosome, isolats in chromosomes.items():
            # Collecter toutes les positions de tous les isolats
            all_positions = []
            for isolat, positions in isolats.items(): all_positions.extend(positions)
            
            # Fusionner les intervalles qui se chevauchent ou sont contigus
            if not all_positions:
                result[method][chromosome] = []
                continue
            
            # Trier les intervalles par position de début
            sorted_intervals = sorted(all_positions)
            merged = [sorted_intervals[0]]
            
            for current_start, current_end in sorted_intervals[1:]:
                last_start, last_end = merged[-1]
                # Si l'intervalle courant chevauche ou est contigu au dernier
                # Fusionner en prenant le max des positions de fin
                if current_start <= last_end + 1: merged[-1] = (last_start, max(last_end, current_end))
                # Ajouter un nouvel intervalle
                else: merged.append((current_start, current_end))
            
            result[method][chromosome] = merged
    
    return result


def size_plot(dfs_dict, inv_colors, output_dir):
    """
    Creates histograms of size distribution for all methods on the same figure.
    Plots are arranged in rows of 3, with unified scales across all subplots.
    Returns a list containing all calculated statistics for each method.
    
    Args:
        dfs_dict: Dictionary {method_name: DataFrame}
        inv_colors: Dictionary {method_name: color}
        output_dir: Output directory path
    
    Returns:
        list: List of statistics dictionaries for each method
    """
    all_stats = []
    all_sizes = {}
    
    # First pass: calculate all sizes and find global size range
    global_min_size = float('inf')
    global_max_size = 0
    
    for method, df in dfs_dict.items():
        df_work = df.copy()
        df_work['size'] = df_work['end'] - df_work['start']
        df_work = df_work[df_work['size'] > 0]
        
        if df_work.empty: continue
        
        # Calculate mean size per cluster
        sizes = df_work.groupby('ninv')['size'].mean()
        all_sizes[method] = [sizes, len(df_work)]
        
        # Update global ranges
        min_val = max(sizes.min(), 1)
        max_val = sizes.max()
        global_min_size = min(global_min_size, min_val)
        global_max_size = max(global_max_size, max_val)
    
    if not all_sizes:
        print("No valid data to plot")
        return all_stats
    
    # Prepare global bins for consistency
    global_bins = np.logspace(np.log10(global_min_size), np.log10(global_max_size), 50)
    
    # Second pass: calculate max frequency using GLOBAL bins
    max_freq = 0
    for method, (sizes, count) in all_sizes.items():
        counts, _ = np.histogram(sizes, bins=global_bins)
        max_freq = max(max_freq, counts.max())
    
    # Add 10% margin to ranges
    y_max = max_freq * 1.1
    x_min = global_min_size * 0.8
    x_max = global_max_size * 1.2
    
    # Calculate grid dimensions
    n_methods = len(all_sizes)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
    
    # Flatten axes array for easier iteration
    if n_methods == 1: axes = np.array([axes])
    elif n_rows == 1: axes = np.array(axes)
    else: axes = axes.flatten()
    
    # Third pass: create plots with unified scales
    for idx, (method, (sizes, count)) in enumerate(all_sizes.items()):
        ax = axes[idx]
        color = inv_colors[method]
        
        # Plot histogram
        ax.hist(sizes, bins=global_bins, alpha=0.7, color=color, edgecolor='black')
        ax.set_xscale('log')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        
        ax.set_xlabel('Mean size of inversion clusters (bp) - log scale', fontsize=22)
        ax.set_ylabel('Frequency', fontsize=22)
        ax.set_title(method, fontsize=24, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        
        # Add statistics lines
        mean_size = sizes.mean()
        median_size = sizes.median()
        
        ax.axvline(mean_size, color='red', linestyle='--', 
                   label=f'Mean: {mean_size:.0f} bp', alpha=0.8, linewidth=2)
        ax.axvline(median_size, color='orange', linestyle='--', 
                   label=f'Median: {median_size:.0f} bp', alpha=0.8, linewidth=2)
        
        ax.legend(fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        
        # Calculate statistics
        stats = {}
        stats['method'] = method
        stats['n_inversions'] = count
        stats['n_clusters'] = len(sizes)
        stats['mean_bp'] = round(mean_size, 2)
        stats['median_bp'] = round(median_size, 2)
        stats['std_bp'] = round(sizes.std(), 2)
        stats['min_bp'] = int(sizes.min())
        stats['max_bp'] = int(sizes.max())
        
        # Percentiles
        for p in [25, 50, 75, 90, 95, 99]:
            stats[f'p{p}_bp'] = round(sizes.quantile(p/100), 2)
        
        # Size categories
        categories = [
            (0, 1_000, "n_lt_1kb"), (1_000, 5_000, "n_1_5kb"),
            (5_000, 10_000, "n_5_10kb"), (10_000, 50_000, "n_10_50kb"),
            (50_000, 100_000, "n_50_100kb"), (100_000, 250_000, "n_100_250kb"),
            (250_000, 500_000, "n_250_500kb"), (500_000, float('inf'), "n_gt_500kb"),
        ]
        
        for min_v, max_v, label in categories:
            if max_v != float('inf'): 
                count_cat = ((sizes >= min_v) & (sizes < max_v)).sum()
            else: 
                count_cat = (sizes >= min_v).sum()
            stats[label] = int(count_cat)
        
        all_stats.append(stats)
    
    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)
    
    save_path = os.path.join(output_dir, 'size_distribution_all_methods.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Size distribution plot saved: {save_path}")
    
    return all_stats


def var_plot(dfs_dict, inv_colors, output_dir):
    """
    Creates variability plots showing relative deviation from mean per cluster for all methods.
    All plots are on the same figure, one per row, with unified Y-axis.
    Y-axis is fixed to [-1, 1] unless data exceeds these bounds.
    
    Args:
        dfs_dict: Dictionary {method_name: DataFrame}
        inv_colors: Dictionary {method_name: color}
        output_dir: Output directory path
    """
    if not dfs_dict:
        print("No data to plot")
        return
    
    methods = list(dfs_dict.keys())
    n_methods = len(methods)
    
    # Calculate all deviations to determine unified Y-axis limits
    all_deviations = []
    processed_data = {}
    
    for method, df in dfs_dict.items():
        if df.empty:
            continue
            
        df_work = df.copy()
        df_work['size'] = df_work['end'] - df_work['start']
        df_work = df_work[df_work['size'] > 0]
        
        if 'ninv' not in df_work.columns:
            print(f"Warning: 'ninv' column missing for {method}, skipping")
            continue
            
        mean_by_cluster = df_work.groupby('ninv')['size'].mean()
        df_work = df_work.merge(mean_by_cluster.rename('mean_cluster'), 
                                left_on='ninv', right_index=True)
        df_work['relative_deviation'] = ((df_work['size'] - df_work['mean_cluster']) / 
                                         df_work['mean_cluster'])
        
        processed_data[method] = df_work
        all_deviations.extend(df_work['relative_deviation'].dropna().tolist())
    
    if not all_deviations:
        print("Warning: No valid deviations to plot")
        return
    
    # Determine unified Y-axis limits: default [-1, 1], adjust if data exceeds
    max_abs_deviation = np.max(np.abs(all_deviations))
    if max_abs_deviation <= 1.0:
        y_lim = 1.0
    else:
        y_lim = max_abs_deviation * 1.1  # 10% margin
    
    # Create figure with subplots (one per row)
    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 5 * n_methods), 
                            sharex=True, sharey=True)
    
    # Handle single method case
    if n_methods == 1:
        axes = [axes]
    
    # Create plots
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        if method not in processed_data:
            ax.set_visible(False)
            continue
        
        df_work = processed_data[method]
        color = inv_colors[method]
        
        # Scatter plot
        ax.scatter(df_work['ninv'], df_work['relative_deviation'],
                   alpha=0.6, s=30, color=color, edgecolors='black',
                   linewidths=0.3, label=method)
        
        # Reference lines
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, 
                   alpha=0.8, label='Mean (deviation = 0)')
        ax.axhline(y=0.5, color='darkorange', linestyle=':', linewidth=1.2, 
                   alpha=0.7, label='±50% deviation')
        ax.axhline(y=-0.5, color='darkorange', linestyle=':', linewidth=1.2, alpha=0.7)
        
        # Styling
        ax.set_title(f"{method}", fontsize=18, fontweight='bold', pad=10)
        ax.set_ylabel('Relative deviation from mean', fontsize=14)
        ax.set_ylim(-y_lim, y_lim)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=12)
        
        # Legend only on first plot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        
        # Statistics text box
        n_points = len(df_work)
        mean_dev = df_work['relative_deviation'].mean()
        std_dev = df_work['relative_deviation'].std()
        textstr = f'n = {n_points}\nμ = {mean_dev:.3f}\nσ = {std_dev:.3f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # X-axis label on bottom plot only
    axes[-1].set_xlabel("Number of inversion clusters", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the combined figure
    save_path = os.path.join(output_dir, 'variability_all_methods.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Variability plot saved: {save_path}")


def plot_inv_by_mapon(inv_df_iso,iso, chr_size_iso, lineage_df, output_dir):
    inversion_color = 'green'

    for method, chr_dic in inv_df_iso.items():
        
        n_chrs = len(chr_dic)
        ncols = 3
        n_rows = math.ceil(n_chrs / ncols) 
        
        fig, axes = plt.subplots(n_rows, ncols, figsize=(27, 4*n_rows))
        
        # Si un seul chromosome, ajuster les axes
        if n_chrs == 1: axes = np.array([[axes]])
        elif n_rows == 1: axes = axes.reshape(1, -1)
        
        # Parcourir les chromosomes
        i = 0
        for chr_name, mapon_dic in chr_dic.items():
            row_idx = i // 3
            col_idx = i % 3
            ax = axes[row_idx, col_idx]
            i+=1

            # Utiliser la taille du chromosome depuis le dic si disponible, sinon calculer
            if ((chr_size_iso['iso'] == iso) & (chr_size_iso['chr'] == chr_name)).any(): max_pos = chr_size_iso[(chr_size_iso['iso'] == iso) & (chr_size_iso['chr'] == chr_name)]['size'].iloc[0]
            else: max_pos = max(end for pos in mapon_dic.values() for start, end in pos)

            # Trier les mapons par lignée
            sorted_mapons = []
            mapon_lineages = {}
            # Récupérer la lignée pour chaque mapon
            for mapon in mapon_dic.keys():
                lineage = lineage_df[lineage_df['iso'] == mapon]['lineage'].iloc[0]
                mapon_lineages[mapon] = lineage
            
            # Trier les mapons d'abord par lignée, puis par nom
            sorted_mapons = sorted(mapon_dic.keys(), key=lambda m: (mapon_lineages.get(m, mapon), m))
            sorted_mapons = sorted_mapons [::-1]

            # Une ligne par mapon
            for j, mapon in enumerate(sorted_mapons):
                y_pos = j
                inversions = mapon_dic[mapon]
                # Tracer les inversions (toutes de la même couleur)
                for start, end in inversions:
                    ax.plot([start, end], [y_pos, y_pos], linewidth=2, color=inversion_color)
            
            # Configuration de l'axe
            ax.set_title(f"{iso} - {chr_name}")
            ax.set_xlim(0, max_pos)
            ax.set_ylim(-0.5, len(sorted_mapons) - 0.5)
            ax.set_yticks(range(len(sorted_mapons)))
            ax.set_yticklabels(sorted_mapons)
            
            # Colorier chaque label y individuellement par lignée
            for j, mapon in enumerate(sorted_mapons):
                lineage = mapon_lineages[mapon]
                color = lineage_df.loc[lineage_df['iso'] == mapon, 'color'].iloc[0] \
                    if (lineage_df['iso'] == mapon).any() else "black"
                ax.get_yticklabels()[j].set_color(color)
            
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Position (bp)')
        
        # Cacher les axes non utilisés dans la dernière ligne si nombre impair de chromosomes
        if n_chrs % 2 != 0: axes[n_rows-1, 1].axis('off')
        

        save_path = os.path.join(output_dir, f"{iso}_{method}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)


def plot_inversion_by_chromosome(data, chr_size_df, isolat_name, method_colors, output_dir):
    """
    Crée une visualisation des inversions chromosomiques pour un isolat.
    Affiche tous les chromosomes avec les inversions détectées par chaque méthode.
    
    Args:
        data (dict): Dictionnaire structuré comme data[iso][methode][chr] 
                     où chaque valeur est une liste de tuples (start, end)
        chr_size_df (df): Tableau contenant les tailles des chromosomes
        isolat_name (str): Nom de l'isolat à afficher
        output_dir (str): Répertoire de sortie pour les figures
    """

    # Convertir les données au format attendu
    iso_data = pairwise_algn_to_multi(data)
    
    # Configuration matplotlib
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['axes.labelsize'], mpl.rcParams['axes.titlesize'] = 12, 14
    mpl.rcParams['xtick.labelsize'], mpl.rcParams['ytick.labelsize'] = 10, 10
    
    # Collecter toutes les méthodes et chromosomes disponibles
    all_methodes, all_chromosomes = set(), set()

    for methode, chr_dict in iso_data.items():
        all_methodes.add(methode)
        for chr_name in chr_dict.keys(): all_chromosomes.add(chr_name)
    
    # Trier les méthodes et chromosomes
    all_methodes = sorted(list(all_methodes))
    
    # Définir un trieur naturel pour les chromosomes
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', s)]
    
    all_chromosomes = sorted(list(all_chromosomes), key=natural_sort_key)
    
    # Créer la figure
    n_chromosomes = len(all_chromosomes)
    fig = plt.figure(figsize=(18, n_chromosomes*0.4 + 1.5), facecolor='white')
        
    # Créer une grille de subplots
    gs = GridSpec(n_chromosomes, 1, figure=fig, hspace=0.05,
                  left=0.12, right=0.95, top=0.88, bottom=0.10)
    
    # Trouver la taille maximale des chromosomes pour l'alignement
    chr_filtered = chr_size_df[chr_size_df['chr'].isin(all_chromosomes)]
    max_chr_size = int(chr_filtered['size'].max())

    # Ajouter un titre général bien séparé
    fig.suptitle(f"Inversions chromosomiques - {isolat_name}", 
                 fontsize=16, fontweight='bold', y=0.95)
    
    n_methods = len(all_methodes)
    # Pour chaque chromosome, créer un subplot
    for chr_idx, chr_name in enumerate(all_chromosomes):
        ax = fig.add_subplot(gs[chr_idx, 0])
        
        # Configuration de base de l'axe
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Ajouter le nom du chromosome à gauche (position fixe basée sur max_chr_size)
        ax.text(-0.02 * max_chr_size, 0.5, chr_name,  transform=ax.transData, 
                ha='right', va='center', fontweight='bold', fontsize=16, color='#333333')
        
        # Configuration de l'axe Y
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        
        # Grille et lignes horizontales
        ax.grid(True, axis='x', linestyle='-', alpha=0.2, color='#cccccc')
        ax.axhline(y=0, color='#aaaaaa', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.axhline(y=1, color='#aaaaaa', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Utiliser 70% de l'espace vertical pour les méthodes
        method_height = min(0.5 / n_methods, 0.2)
        spacing = min((0.7 - method_height * n_methods) / (n_methods + 1), 0.05)
        y_start = 0.15
        
        # Calculer les positions Y pour chaque méthode
        y_positions = {}

        for idx, methode in enumerate(all_methodes):
            y_pos = y_start + idx * (method_height + spacing)
            y_positions[methode] = y_pos + method_height / 2
        
        # Tracer les inversions pour chaque méthode
        for methode in all_methodes:                
            inversions = iso_data[methode][chr_name]
            y_center = y_positions[methode]
            height = method_height * 0.6
            
            # Tracer chaque inversion
            for start, end in inversions:
                rect = plt.Rectangle(
                    (start, y_center - height/2), end - start, 
                    height, facecolor=method_colors[methode],
                    edgecolor='none', alpha=0.8,zorder=10)
                ax.add_patch(rect)
        
        # Obtenir la taille du chromosome actuel
        chr_size = chr_size_df[chr_size_df['chr'] == chr_name]['size'].iloc[0]

        # Ajouter une ligne verticale pour marquer la fin de ce chromosome
        ax.axvline(x=chr_size+1000, color='#0d0a0b', linestyle='--', 
                   alpha=0.6, linewidth=1, zorder=5, label='Fin du chromosome')
        
        # Formatter pour Mpb
        def format_x_tick(x, pos): return '{:.1f}'.format(x / 1e6)
        
        # Grille et ticks - aller jusqu'à la fin maximale pour alignement
        tick_interval = 1e6
        major_ticks = np.arange(0, max_chr_size + tick_interval, tick_interval)
        ax.set_xticks(major_ticks)
        ax.grid(True, which='major', axis='x', linestyle='-', alpha=0.2, color='#dddddd')
        
        # Limites des axes X - utiliser max_chr_size pour que tous soient alignés
        ax.set_xlim(-0.03 * max_chr_size, 1.02 * max_chr_size)
        
        # Configurer l'axe X seulement pour le dernier chromosome
        if chr_idx == n_chromosomes - 1:
            ax.xaxis.set_major_formatter(FuncFormatter(format_x_tick))
            ax.set_xlabel('Position (Mpb)', fontweight='bold', color='#333333', labelpad=10)
        else: ax.set_xticklabels([])
    
    # Ajouter une légende bien en dessous du dernier axe
    if all_methodes:
        handles = [plt.Rectangle((0, 0), 1, 1, color=method_colors.get(methode, 'gray'), alpha=0.8) for methode in all_methodes]

        # Légende positionnée en bas avec bbox_to_anchor pour être sous l'axe X
        fig.legend(handles, all_methodes, loc='upper center', bbox_to_anchor=(0.75, 0.04),
                  ncol=len(all_methodes), frameon=True,  framealpha=1, fontsize=10)
    
    # Enregistrer le graphique
    output_file = os.path.join(output_dir, f"{isolat_name}_inversions.svg")
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Graphique sauvegardé: {output_file}")


def plot_inv_coverage(inv_data, chr_size_df, lineage_df, inv_colors, save_path):
    """
    Create a horizontal barplot comparing the percentage of genome covered by inversions
    for each isolate across different inversion detection methods.
    
    Parameters:
    -----------
    inv_data : dict
        Dictionary with DataFrames for each method
    chr_size_df : DataFrame
        DataFrame with chromosome sizes per isolate
    lineage_df : DataFrame
        DataFrame with columns 'iso', 'lineage', 'color'
    inv_colors : dict
        Dictionary of colors for each method
    save_path : str
        Output path for the PNG file
    """
    
    def merge_intervals(intervals):
        """Merge overlapping intervals"""
        if not intervals:
            return []
        
        intervals_sorted = sorted(intervals, key=lambda x: x[0])
        merged = [intervals_sorted[0]]
        
        for current in intervals_sorted[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def calculate_coverage_length(intervals):
        """Calculate total length covered by merged intervals"""
        merged = merge_intervals(intervals)
        return sum(end - start + 1 for start, end in merged)

    def calculate_coverage_for_file(method_df):
        """Calculate coverage percentage for a given file"""
        df_clean = method_df[['iso', 'chr', 'start', 'end']]
        coverage_percentages = {}
        
        for iso in df_clean['iso'].unique():
            iso_data = df_clean[df_clean['iso'] == iso]
            
            total_covered = 0
            total_genome_size = 0
            
            for chr_name in iso_data['chr'].unique():
                chr_data = iso_data[iso_data['chr'] == chr_name]
                
                intervals = [(row['start'], row['end']) for _, row in chr_data.iterrows()]
                covered_length = calculate_coverage_length(intervals)
                chr_size = int(chr_size_df.loc[(chr_size_df['iso'] == iso) & 
                                               (chr_size_df['chr'] == chr_name), 'size'].iloc[0])
                
                total_covered += covered_length
                total_genome_size += chr_size
            
            if total_genome_size > 0:
                coverage_percentages[iso] = (total_covered / total_genome_size) * 100

        return coverage_percentages
    
    # Calculate coverage for each method
    method_coverages = {}
    method_stats = {}
    
    for method_name, method_df in inv_data.items():
        coverage = calculate_coverage_for_file(method_df)
        method_coverages[method_name] = coverage
        
        if coverage:
            values = list(coverage.values())
            method_stats[method_name] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Get all unique isolats and sort them
    all_isolats = sorted(set().union(*[cov.keys() for cov in method_coverages.values()]))
    
    # Create color dictionary for isolats
    iso_colors = dict(zip(lineage_df['iso'], lineage_df['color']))
    
    # Setup figure dimensions
    n_methods = len(method_coverages)
    fig_height = max(10, len(all_isolats) * 0.35 + 2)
    fig = plt.figure(figsize=(18, fig_height))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1], wspace=0.25)
    ax_main = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])
    
    # === MAIN PLOT ===
    bar_height = 0.25
    y_positions = np.arange(len(all_isolats))
    
    # Create bars for each method (reversed order to match legend)
    methods_list = list(method_coverages.keys())
    for i, method_name in enumerate(reversed(methods_list)):
        percentages = [method_coverages[method_name].get(iso, 0) for iso in all_isolats]
        ax_main.barh(y_positions + i * bar_height, percentages, 
                     bar_height, label=method_name, color=inv_colors[method_name], 
                     alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Customize main plot
    ax_main.set_ylabel('Isolats', fontsize=12, fontweight='bold')
    ax_main.set_xlabel('Genomic Coverage (%)', fontsize=12, fontweight='bold')
    ax_main.set_title('Comparison of Inversion Genomic Coverage by Method', 
                      fontsize=14, fontweight='bold', pad=20)
    
    # Configure Y-axis with lineage colors - center on middle of grouped bars
    middle_offset = (n_methods - 1) * bar_height / 2
    ax_main.set_yticks(y_positions + middle_offset)
    ax_main.set_yticklabels(all_isolats)
    
    # Color isolate labels by lineage
    for i, iso in enumerate(all_isolats):
        label = ax_main.get_yticklabels()[i]
        label.set_color(iso_colors.get(iso, 'black'))
        label.set_fontweight('bold')
    
    ax_main.set_xlim(0, 100)
    ax_main.grid(True, alpha=0.3, axis='x')
    
    # === COMBINED LEGEND ===
    # Get method handles and labels (reversed)
    methods_handles, methods_labels = ax_main.get_legend_handles_labels()
    methods_handles = list(reversed(methods_handles))
    methods_labels = list(reversed(methods_labels))
    
    # Create lineage handles and labels
    lineage_handles = []
    lineage_labels = []
    for lineage in sorted(lineage_df['lineage'].unique()):
        color = lineage_df[lineage_df['lineage'] == lineage]['color'].iloc[0]
        lineage_handles.append(plt.Line2D([0], [0], marker='s', color='w', 
                                         markerfacecolor=color, markersize=10, 
                                         markeredgecolor='black', markeredgewidth=0.8))
        lineage_labels.append(lineage)
    
    # Combine legends
    all_handles = methods_handles + lineage_handles
    all_labels = methods_labels + lineage_labels
    
    # Create single combined legend with white background
    combined_legend = ax_main.legend(all_handles, all_labels,
                                     loc='upper right', 
                                     framealpha=1.0,
                                     fontsize=10, 
                                     edgecolor='black',
                                     facecolor='white',
                                     ncol=1)
    combined_legend.set_title('Methods & Lineages', prop={'size': 11, 'weight': 'bold'})
    
    # === STATISTICS PANEL ===
    ax_stats.axis('off')
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    
    # Stats panel title
    ax_stats.text(0.5, 0.97, 'Statistics', 
                  ha='center', va='top', fontsize=13, fontweight='bold',
                  transform=ax_stats.transAxes)
    
    # Calculate spacing for stat blocks
    y_start = 0.90
    spacing = 0.03
    block_height = 0.20
    
    for i, (method_name, stats) in enumerate(method_stats.items()):
        # Block position
        y_top = y_start - i * (block_height + spacing)
        y_bottom = y_top - block_height
        
        # Colored rectangle for each method
        rect = plt.Rectangle((0.05, y_bottom), 0.9, block_height,
                            facecolor=inv_colors[method_name], alpha=0.15,
                            edgecolor=inv_colors[method_name], linewidth=3,
                            transform=ax_stats.transAxes, zorder=1)
        ax_stats.add_patch(rect)
        
        # Method name at top of block
        ax_stats.text(0.5, y_top - 0.005, method_name,
                     ha='center', va='top', fontsize=14, fontweight='bold',
                     color=inv_colors[method_name], 
                     transform=ax_stats.transAxes, zorder=2)
        
        # Statistics centered in block
        stats_text = (f"Mean   : {stats['mean']:>6.2f}%\n"
                     f"Median : {stats['median']:>6.2f}%\n"
                     f"Std Dev: {stats['std']:>6.2f}%\n"
                     f"Min    : {stats['min']:>6.2f}%\n"
                     f"Max    : {stats['max']:>6.2f}%")
        
        ax_stats.text(0.5, y_top - 0.05, stats_text,
                     ha='center', va='top', fontsize=12, 
                     family='monospace', linespacing=1.4,
                     transform=ax_stats.transAxes, zorder=2)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")


def main(data):
    """Main processing function"""
    
    def print_section(title, char='='):
        """Print a formatted section header"""
        print(f"\n{char*60}")
        print(f"  {title}")
        print(f"{char*60}")
    
    def print_success(message):
        """Print a success message"""
        print(f"✓ {message}")
    
    def print_warning(message):
        """Print a warning message"""
        print(f"⚠ WARNING: {message}")
    
    # Extract data
    inv_df = data['inversions']
    inv_colors = data['colors']        
    all_data = make_data(inv_df)
    output_dir = data['outpath']
    
    print_section("INVERSION ANALYSIS PIPELINE", '=')
    print(f"Output directory: {output_dir}\n")
    
    # ========== Size Distribution Plots ==========
    print_section("Size Distribution Analysis")
    output_dir_stat = os.path.join(output_dir, 'statistics')
    os.makedirs(output_dir_stat, exist_ok=True)

    #all_stats = size_plot(inv_df, inv_colors, output_dir_stat)
    all_stats = None
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        tsv_save_path = os.path.join(output_dir_stat, 'stats_summary.tsv')
        stats_df.to_csv(tsv_save_path, index=False, sep='\t')
        
        print("\nSummary Statistics:")
        print(stats_df.to_string(index=False))
        print_success(f"Statistics saved to: {tsv_save_path}")
    else:
        print_warning("No statistics were generated")
    
    # ========== Variability Plots ==========
    print_section("Variability Analysis")
    #var_plot(inv_df, inv_colors, output_dir_stat)
    print_success("Variability plots generated")
    
    # ========== Per-Isolate Mapping Plots ==========
    iso_reference = data['iso_ref']
    chr_size_df = data['chr_lengths']
    lineage_df = data['lineages']
    
    print_section("Per-Isolate Inversion Mapping")
    print(f"Processing {len(iso_reference)} reference isolate(s)...\n")
    
    output_dir_mapping = os.path.join(output_dir, 'mapping_plots')
    os.makedirs(output_dir_mapping, exist_ok=True)
    
    output_dir_chromosome = os.path.join(output_dir, 'chromosome_plots')
    os.makedirs(output_dir_chromosome, exist_ok=True)
    
    for i, iso in enumerate(iso_reference, 1):
        print(f"  [{i}/{len(iso_reference)}] Processing isolate: {iso}")
        chr_size_iso = chr_size_df[chr_size_df['iso'] == iso]
        #plot_inv_by_mapon(all_data[iso], iso, chr_size_iso, lineage_df, output_dir_mapping)
        #plot_inversion_by_chromosome(all_data[iso], chr_size_iso, iso, inv_colors, output_dir_chromosome)
    
    print_success(f"Mapping plots saved to: {output_dir_mapping}")
    print_success(f"Chromosome plots saved to: {output_dir_chromosome}")
    
    # ========== Genomic Coverage Analysis ==========
    print_section("Genomic Coverage Analysis")
    
    output_dir_coverage = os.path.join(output_dir, 'inv_coverage')
    os.makedirs(output_dir_coverage, exist_ok=True)
    coverage_plot_path = os.path.join(output_dir_coverage, 'genomic_coverage_comparison.png')
    
    #plot_inv_coverage(inv_df, chr_size_df, lineage_df, inv_colors, coverage_plot_path)
    print_success(f"Coverage plot saved to: {coverage_plot_path}")

    #TODO ========== TITI ==========
    bp.execut(data)





    # ========== Pipeline Complete ==========
    print_section("ANALYSIS COMPLETE", '=')
    print(f"All results saved in: {output_dir}\n")


if __name__ == "__main__":
    config = cfg_loader.ConfigLoader("config.yaml")
    data = config.load_all()
    
    try:
        main(data)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR: Analysis failed")
        print(f"{'='*60}")
        print(f"Error details: {str(e)}")
        raise