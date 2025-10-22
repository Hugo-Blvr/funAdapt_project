import pandas as pd
from collections import defaultdict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import math
import multiprocessing as mp
from tqdm import tqdm # type: ignore
from concurrent.futures import ProcessPoolExecutor
import psutil # type: ignore
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec



def to_mean(data):
    for methode, dic_te_class in data.items():
        for te_class, dic_iso in dic_te_class.items():
            for iso, dic_chr in dic_iso.items():
                for chr, strand_dic in dic_chr.items():
                    for strand, pos_dic in strand_dic.items():
                        for pos, values in pos_dic.items():
                            if len(values) > 0: 
                                data[methode][te_class][iso][chr][strand][pos] = [np.mean(values), len(values)]
                            else: data[methode][te_class][iso][chr][strand][pos] = [0, 0]

    for methode, dic_te_class in data.items():
        for te_class, dic_iso in dic_te_class.items():
            # Pour chaque isoforme, calculer son all_chr
            for iso, dic_chr in dic_iso.items():
                if iso == 'all_iso': continue
                
                # Initialiser all_chr pour cet isoforme
                data[methode][te_class][iso]['all_chr'] = {}
                
                # Parcourir les chromosomes de l'isoforme
                for chr, dic_strand in dic_chr.items():
                    if chr == 'all_chr': continue
                    
                    # Parcourir les brins de ce chromosome
                    for strand, pos_dic in dic_strand.items():
                        if strand not in data[methode][te_class][iso]['all_chr']:
                            data[methode][te_class][iso]['all_chr'][strand] = {}
                        
                        # Collecter les données pour chaque position
                        for pos, values in pos_dic.items():
                            if pos not in data[methode][te_class][iso]['all_chr'][strand]:
                                data[methode][te_class][iso]['all_chr'][strand][pos] = [[], 0]
                            
                            data[methode][te_class][iso]['all_chr'][strand][pos][0].append(values[0])
                            data[methode][te_class][iso]['all_chr'][strand][pos][1] += values[1]
                
                # Calculer les moyennes pour all_chr de cet isoforme
                for strand, pos_dic in data[methode][te_class][iso]['all_chr'].items():
                    for pos, values in pos_dic.items():
                        if len(values[0]) > 0:
                            data[methode][te_class][iso]['all_chr'][strand][pos] = [np.mean(values[0]), values[1]]
                        else:
                            data[methode][te_class][iso]['all_chr'][strand][pos] = [0, values[1]]

            # Puis faire l'agrégation globale pour all_iso
            data[methode][te_class]['all_iso'] = {}
            
            # Liste des chromosomes à agréger
            chromosomes = set()
            for iso, dic_chr in dic_iso.items():
                if iso == 'all_iso': continue
                chromosomes.update(chr for chr in dic_chr.keys() if chr != 'all_chr')
            
            # Agréger les données pour chaque chromosome
            for chr in chromosomes:
                data[methode][te_class]['all_iso'][chr] = {}
                
                # Collecter les données de tous les isoformes pour ce chromosome
                chr_data = {}
                for iso, dic_chr in dic_iso.items():
                    if iso == 'all_iso' or chr not in dic_chr: continue
                    
                    for strand, pos_dic in dic_chr[chr].items():
                        if strand not in chr_data:
                            chr_data[strand] = {}
                        
                        for pos, values in pos_dic.items():
                            if pos not in chr_data[strand]:
                                chr_data[strand][pos] = [[], 0]
                            
                            chr_data[strand][pos][0].append(values[0])
                            chr_data[strand][pos][1] += values[1]
                
                # Calculer les moyennes pour ce chromosome
                for strand, pos_dic in chr_data.items():
                    data[methode][te_class]['all_iso'][chr][strand] = {}
                    
                    for pos, values in pos_dic.items():
                        if len(values[0]) > 0:
                            data[methode][te_class]['all_iso'][chr][strand][pos] = [np.mean(values[0]), values[1]]
                        else:
                            data[methode][te_class]['all_iso'][chr][strand][pos] = [0, values[1]]
            
            # Calculer all_chr pour all_iso
            data[methode][te_class]['all_iso']['all_chr'] = {}
            
            # Collecter les données pour all_chr de tous les chromosomes
            all_chr_data = {}
            for chr, chr_data in data[methode][te_class]['all_iso'].items():
                if chr == 'all_chr': continue
                
                for strand, pos_dic in chr_data.items():
                    if strand not in all_chr_data:
                        all_chr_data[strand] = {}
                    
                    for pos, values in pos_dic.items():
                        if pos not in all_chr_data[strand]:
                            all_chr_data[strand][pos] = [[], 0]
                        
                        all_chr_data[strand][pos][0].append(values[0])
                        all_chr_data[strand][pos][1] += values[1]
            
            # Calculer les moyennes pour all_chr
            for strand, pos_dic in all_chr_data.items():
                data[methode][te_class]['all_iso']['all_chr'][strand] = {}
                
                for pos, values in pos_dic.items():
                    if len(values[0]) > 0:
                        data[methode][te_class]['all_iso']['all_chr'][strand][pos] = [np.mean(values[0]), values[1]]
                    else:
                        data[methode][te_class]['all_iso']['all_chr'][strand][pos] = [0, values[1]]

    return data


def process_inversion(inv, full_df_filtered, te_starts, te_ends, range_5to3, range_3to5, window_size_half, mid_size_inv, chr_size):
    """Traite une seule inversion - fonction pour la parallélisation"""
    results = {'5to3': {}, '3to5': {}}
    
    # Traitement côté 5'→3'
    for range_val in range_5to3:
        if range_val > mid_size_inv: continue
        pos = inv['start'] + range_val
        window_start = pos - window_size_half
        window_end = pos + window_size_half
        # Vérification des limites
        if window_start < 0: continue
        # Vérification des inversions chevauchantes
        if range_val < 0:
            mask = (full_df_filtered['start'] < window_end) & (full_df_filtered['end'] > window_start)
            #if not mask.any(): continue
        # Sélection vectorisée des TE dans la fenêtre
        in_window = (te_starts <= window_end) & (te_ends >= window_start)
        if not np.any(in_window):
            results['5to3'][range_val] = 0  # Aucun TE dans cette fenêtre
            continue
        
        # Calcul optimisé des chevauchements
        te_in_window_starts = te_starts[in_window]
        te_in_window_ends = te_ends[in_window]
        overlap_start = np.maximum(te_in_window_starts, window_start)
        overlap_end = np.minimum(te_in_window_ends, window_end)
        overlap_length = np.maximum(0, overlap_end - overlap_start)
        total_bases_in_te = np.sum(overlap_length)
        density = total_bases_in_te / (2 * window_size_half)
        results['5to3'][range_val] = density
    
    # Traitement côté 3'→5'
    for range_val in range_3to5:
        if -range_val > mid_size_inv: continue
            
        pos = inv['end'] + range_val
        window_start = pos - window_size_half
        window_end = pos + window_size_half
        # Vérification des limites
        if window_end > chr_size: continue
        # Vérification des inversions chevauchantes
        if range_val > 0:
            mask = (full_df_filtered['start'] < window_end) & (full_df_filtered['end'] > window_start)
            #if not mask.any(): continue
        # Sélection vectorisée des TE dans la fenêtre
        in_window = (te_starts <= window_end) & (te_ends >= window_start)
        if not np.any(in_window):
            results['3to5'][range_val] = 0  # Aucun TE dans cette fenêtre
            continue
        # Calcul optimisé des chevauchements
        te_in_window_starts = te_starts[in_window]
        te_in_window_ends = te_ends[in_window]
        overlap_start = np.maximum(te_in_window_starts, window_start)
        overlap_end = np.minimum(te_in_window_ends, window_end)
        overlap_length = np.maximum(0, overlap_end - overlap_start)
        total_bases_in_te = np.sum(overlap_length)
        density = total_bases_in_te / (2 * window_size_half)
        results['3to5'][range_val] = density


    return results


def process_chromosome_te_class(args):
    """Traite un chromosome et une classe de TE"""
    methode, te_class, iso, chr, df_inv_chr, df_te_class,range_5to3, range_3to5, window_size_half, chr_size = args
    result_dict = {'5to3': {}, '3to5': {}}
    
    # Initialisation des dictionnaires pour tous les ranges
    for range_val in range_5to3: result_dict['5to3'][range_val] = []
    for range_val in range_3to5: result_dict['3to5'][range_val] = []
    
    # Pré-calcul des valeurs pour éviter les recalculs répétés
    te_starts = df_te_class['start'].values
    te_ends = df_te_class['end'].values
    
    # Traitement de chaque inversion
   
    for _, inv in df_inv_chr.iterrows():
        full_df_filtered = df_inv_chr[(df_inv_chr['start'] != inv['start']) & (df_inv_chr['end'] != inv['end'])].copy()
        mid_size_inv = (inv['end'] - inv['start']) // 2
        
        # Traitement d'une inversion
        inv_results = process_inversion(inv, full_df_filtered, te_starts, te_ends, 
                                       range_5to3, range_3to5, window_size_half, mid_size_inv, chr_size)
        
        # Fusionner les résultats
        for direction in ['5to3', '3to5']:
            for range_val, density in inv_results[direction].items():
                result_dict[direction][range_val].append(density)
    

    return methode, te_class, iso, chr, result_dict


def make_data(inv_datas, te_data, chr_size_dict, inv_size=100000, flanking_region_size=50000, step_size=1000, n_jobs=None):
    """
    Analyse optimisée et parallélisée de la densité de TE autour des inversions
    
    Parameters:
    -----------
    inv_datas : dict
        Dictionnaire contenant les données d'inversion par méthode
    te_data : pd.DataFrame
        DataFrame contenant les données d'éléments transposables
    chr_size_dict : dict
        Dictionnaire des tailles de chromosomes
    inv_size : int, default=10000
        Taille standard des inversions
    flanking_region_size : int, default=4000
        Taille des régions flanquantes à analyser
    step_size : int, default=1000
        Pas d'échantillonnage
    n_jobs : int, default=None
        Nombre de processus parallèles à utiliser. Si None, utilise (CPU count - 1)
        
    Returns:
    --------
    dict
        Dictionnaire structuré contenant les densités de TE
    """
    
    # Déterminer le nombre optimal de processus
    if n_jobs is None:
        n_jobs = max(1, psutil.cpu_count(logical=True) - 1)  # Physique - 1 
    print(f"Utilisation de {n_jobs} processus parallèles")
    
    # Précalcul des intervalles
    range_5to3 = [-dist for dist in reversed(range(0, flanking_region_size + 1, step_size))] + \
                 [dist for dist in range(step_size, inv_size // 2 + 1, step_size)]
    range_3to5 = [-x for x in range_5to3[::-1]]
    
    # Demi-taille de la fenêtre d'analyse
    window_size_half = step_size // 2 
    

    all_te_classes = te_data.copy()
    all_te_classes['class'] = "all_te_class"
    te_data = pd.concat([te_data, all_te_classes], axis=0)
    
    #te_data['class'] = "all_te_class"
    
    # Créer une liste de toutes les tâches à traiter
    tasks = []
    full_data = {}
    for methode, df_inv in inv_datas.items():
        full_data[methode] = {}
        for te_class, df_te_class in te_data.groupby('class'):
            full_data[methode][te_class] = {}
            for iso, df_inv_iso in df_inv.groupby('iso'):
                full_data[methode][te_class][iso] = {}
                for chr, df_inv_chr in df_inv_iso.groupby('chr'):
                    df_te_filtered = df_te_class.query("iso in @iso and chr in @chr")
                    full_data[methode][te_class][iso][chr] = {'5to3': {}, '3to5': {}}
                    chr_size = chr_size_dict[iso][chr]
                    tasks.append((methode, te_class, iso, chr, df_inv_chr, df_te_filtered,
                        range_5to3, range_3to5, window_size_half, chr_size))
    

    print(f"Nombre total de tâches à traiter : {len(tasks)}")
    
    # Exécution parallèle des tâches avec barre de progression
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Utilisation de tqdm pour afficher la progression
        for result in tqdm(executor.map(process_chromosome_te_class, tasks), 
                          total=len(tasks), 
                          desc="Traitement des données", 
                          unit="task"):
            results.append(result)
            
    # Traitement des résultats
    for methode, te_class, iso, chr, result_dict in results:
        for direction in ['5to3', '3to5']:
            for range_val, densities in result_dict[direction].items():
                full_data[methode][te_class][iso][chr][direction][range_val] = densities
                
    full_data = to_mean(full_data)
    

    return full_data


def reduire_intervalles(liste, x):
    liste = sorted(liste)
    pas_de_base = liste[1] - liste[0]
    min_val = liste[0]
    max_val = liste[-1]
    meilleur_resultat = None
    meilleur_total = 0

    for facteur in range(1, (max_val - 0) // pas_de_base + 1):
        pas = facteur * pas_de_base
        # On ne garde le pas que si min_val et max_val sont alignés avec ce pas
        if (0 - min_val) % pas != 0 or (max_val - 0) % pas != 0: continue

        gauche = list(range(min_val + pas, 0, pas))
        droite = list(range(pas, max_val, pas))
        total = 1 + len(gauche) + 1 + len(droite) + 1  # min + gauche + 0 + droite + max
        if total <= x and total > meilleur_total:
            meilleur_resultat = [min_val] + gauche + [0] + droite + [max_val]
            meilleur_total = total

    if meilleur_resultat is None: return [min_val, 0, max_val]

    return meilleur_resultat


def plot_density(data, axes, title, color_dict, args):
    """
    Plot density data with optional histograms showing second values.
    
    Parameters:
    -----------
    data : dict
        Input data dictionary with nested structure
    title : str
        Title for the plot
    show_histogram : bool, default=True
        Whether to show histograms with second values below the plots
    """

    show_histogram = True
    ax1, ax2 = axes[0], axes[1]
    if len(axes) == 2: show_histogram = False
    else : ax3, ax4 = axes[2], axes[3]
        
    ax1.set_title(title, x=1.12, y=1.05, fontsize = 18)

    _5to3 = {}
    _3to5 = {}

    for main_key, strand_dic in data.items():
        for strand, pos_dic in strand_dic.items():
            for pos, value in pos_dic.items():
                if strand == '5to3': _5to3.setdefault(pos, {})[main_key] = value
                else : _3to5.setdefault(pos, {})[main_key] = value
            

    
    df_5to3 = pd.DataFrame.from_dict(_5to3, orient='index').sort_index()
    df_3to5 = pd.DataFrame.from_dict(_3to5, orient='index').sort_index()


    def extract_value(df, i):
        df_values = df.copy()
        for col in df_values.columns:
            df_values[col] = df_values[col].apply(lambda x: x[i])
        return df_values
    
    # Extraction des valeurs
    df_5to3_values = extract_value(df_5to3.copy(), 0)
    df_3to5_values = extract_value(df_3to5.copy(), 0)

    

    x_ticks_5to3 = reduire_intervalles(sorted(df_5to3_values.index.tolist()), 12)
    x_tick_labels_5to3 = [str(int(x)) if x != 0 else "breakpoints 5'" for x in x_ticks_5to3]
    x_ticks_3to5 = reduire_intervalles(sorted(df_3to5_values.index.tolist()), 12)
    x_tick_labels_3to5 = [str(int(x)) if x != 0 else "breakpoints 3'" for x in x_ticks_3to5]

    
    # Première graphique - avant et début de l'inversion
    for column in df_5to3_values.columns:
        line_size = args['line_size']
        if 'all' in column : line_size = line_size*3
        ax1.plot(df_5to3_values.index, df_5to3_values[column], linewidth= line_size, marker='o', markersize= args['marker_size'], label=column, color=color_dict[column])
    
    if not show_histogram: ax1.set_xlabel("Position relative aux breakpoints des inversions (pb)", x=1.01, y= 0.6)
        
    ax1.set_ylabel("Densité")
    ax1.set_ylim(0, 1)  
    ax1.spines['right'].set_visible(False)
    
    if show_histogram: x_tick_labels = ["" for x in x_tick_labels_5to3]
    else: x_tick_labels = x_tick_labels_5to3 
    # Appliquer les ticks sélectionnés
    ax1.set_xticks(x_ticks_5to3)
    ax1.set_xticklabels(x_tick_labels, rotation=90)
    ax1.grid(True, color='gray', alpha=0.4)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.8)

    # Deuxième graphique - fin de l'inversion et après
    for column in df_3to5_values.columns:
        line_size = args['line_size']
        if 'all' in column : line_size = line_size*3
        ax2.plot(df_3to5_values.index, df_3to5_values[column], linewidth= line_size, marker='o',  markersize= args['marker_size'], label=column, color=color_dict[column])

    ax2.set_ylim(0, 1)  
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='y', which='both', length=0, labelleft=False)
    
    if show_histogram: x_tick_labels = ["" for x in x_tick_labels_3to5]
    else: x_tick_labels = x_tick_labels_3to5
    # Appliquer les ticks sélectionnés
    ax2.set_xticks(x_ticks_3to5)
    ax2.set_xticklabels(x_tick_labels, rotation=90)
    ax2.grid(True, color='gray', alpha=0.4)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.8)
    
    if show_histogram:
        df_5to3_histo = extract_value(df_5to3.copy(), 1)
        df_3to5_histo = extract_value(df_3to5.copy(), 1)
        bar_width = 0.8

        # Histogramme gauche (5'→3')
        x_5to3 = np.arange(len(df_5to3_histo.index))  # Positions numériques pour les barres
        x_mapping = dict(zip(df_5to3_histo.index, x_5to3))  # Crée un mapping entre index réels et positions numériques

        for i, column in enumerate(df_5to3_histo.columns):
            if 'all' in column: continue
            width = bar_width / len(df_5to3_histo.columns)
            offset = (i - (len(df_5to3_histo.columns) - 1) / 2) * width
            values = df_5to3_histo[column].values
            ax3.bar(x_5to3 + offset, values, width=width, label=column, alpha=0.7, color=color_dict[column])

        ax3.set_xlabel("Position relative aux breakpoints des inversions (pb)", x=1.01, y= 0.6)
        ax3.set_ylabel("Nb invs")
        ax3.grid(True, axis='y', color='gray', alpha=0.4)
        ax3.spines['right'].set_visible(False)
        ax3.axvline(x=x_mapping[0], color='black', linestyle='--', alpha=0.8)
        # Convertissez vos ticks personnalisés en positions numériques pour qu'ils s'alignent avec les barres
        mapped_ticks = [x_mapping.get(tick, tick) for tick in x_ticks_5to3 if tick in x_mapping]
        ax3.set_xticks(mapped_ticks)
        ax3.set_xticklabels([str(int(x)) if x != 0 else "breakpoints 5'" for x in x_ticks_5to3 if x in df_5to3_histo.index], rotation=90) 
            
        # Histogramme droit (3'→5')
        x_3to5 = np.arange(len(df_3to5_histo.index))  # Positions numériques pour les barres
        x_mapping = dict(zip(df_3to5_histo.index, x_3to5))  # Crée un mapping entre index réels et positions numériques

        for i, column in enumerate(df_3to5_histo.columns):
            if 'all' in column: continue
            width = bar_width / len(df_3to5_histo.columns)
            offset = (i - (len(df_3to5_histo.columns) - 1) / 2) * width
            values = df_3to5_histo[column].values
            ax4.bar(x_3to5 + offset, values, width=width, label=column, alpha=0.7, color=color_dict[column])

        ax4.tick_params(axis='y', which='both', length=0, labelleft=False)
        ax4.grid(True, axis='y', color='gray', alpha=0.4)
        ax4.spines['left'].set_visible(False)
        ax4.axvline(x=x_mapping[0], color='black', linestyle='--', alpha=0.8)
        # Convertissez vos ticks personnalisés en positions numériques pour qu'ils s'alignent avec les barres
        mapped_ticks = [x_mapping.get(tick, tick) for tick in x_ticks_3to5 if tick in x_mapping]
        ax4.set_xticks(mapped_ticks)
        ax4.set_xticklabels([str(int(x)) if x != 0 else "breakpoints 3'" for x in x_ticks_3to5 if x in df_3to5_histo.index], rotation=90)

    # Supprimer les légendes individuelles
    if ax1.get_legend(): ax1.get_legend().remove()
    if ax2.get_legend(): ax2.get_legend().remove()
    #ax1.legend()
    

def save_plot(data,color_dict, output_dir):
    """
    Génère et sauvegarde les graphiques à partir des données.
    
    Parameters:
    -----------
    data : dict
        Dictionnaire de données à tracer
    output_dir : str
        Répertoire de sortie pour les images
    """
    ###################################### PLOT 1 ALL DATA by METHODE ######################################

    data_all_iso = {methode: data[methode]['all_te_class']['all_iso']['all_chr'] for methode in data.keys()}
    
    fig = plt.figure(figsize=(12,8))
    gs0 = fig.add_gridspec(100, 1, wspace=0.5, hspace=4)

    ax_title = fig.add_subplot(gs0[5, 0])
    ax_title.axis('off')       
    ax_title.set_title("Densité de TE autour des inversions par méthode", fontsize=18) 
    

    args = {'line_size': 1,'marker_size': 2}
    gs01 = gridspec.GridSpecFromSubplotSpec(2,2 , gs0[15:72, 0], height_ratios=[10,4])
    axes = [fig.add_subplot(gs01[0, 0]), fig.add_subplot(gs01[0, 1]), fig.add_subplot(gs01[1, 0]), fig.add_subplot(gs01[1, 1])]
    plot_density(data_all_iso, axes, "", color_dict, args)
   

    legend_ax = fig.add_subplot(gs0[97, 0])
    legend_ax.axis('off')  # Cacher les axes
    handles, labels = [], []
    all_te_class = sorted(list(color_dict.keys()))
    for te_class in all_te_class:
        handle = mpatches.Patch(color=color_dict[te_class], label=te_class)
        handles.append(handle)
        labels.append(te_class)

    # Créer la légende
    legend = legend_ax.legend(handles, labels, loc='center', ncol=6, fontsize=10, 
                            frameon=True, title="Méthodes")
    legend.get_title().set_fontsize(12) 

    fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.05) 

   
    fig.savefig(f"{output_dir}/vf_methode_gene.svg", dpi=300)
    plt.close(fig)
    
    ###################################### PLOT 2 CHR by ISO ######################################


    unique_chr_by_methode = {}
    for methode, te_class_dic in data.items():
        unique_chr_by_methode[methode] = set()
        for iso, chr_dic in te_class_dic['all_te_class'].items():
            if iso == 'all_iso': continue
            unique_chr_by_methode[methode].update(chr_dic.keys())

    
    args = {'line_size': 0.5, 'marker_size': 0}

    nb_col = 4
    for methode, te_class_dic in data.items():
        all_chr = sorted(list(unique_chr_by_methode[methode]))
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(len(all_chr))]
        color_dict = dict(zip(all_chr, colors))
        color_dict['all_chr'] = 'black'

        nb_rows = (len(te_class_dic['all_te_class']) + nb_col - 1) // nb_col
        fig = plt.figure(figsize=(nb_col*6,nb_rows*4+5))
        gs0 = fig.add_gridspec(100, 1, wspace=0.5, hspace=4)
        
        ax_title = fig.add_subplot(gs0[1, 0])
        ax_title.axis('off')       
        ax_title.set_title("Densité de TE autour des inversions par chromosomes", fontsize=18) 
        
        gs01 = gridspec.GridSpecFromSubplotSpec(nb_rows, nb_col, subplot_spec=gs0[8:86, 0], wspace=0.3, hspace=0.8)
        plot_idx = 0 
        for iso, chr_dic in te_class_dic['all_te_class'].items(): 
            nrow = plot_idx // nb_col  # On remplit plusieurs colonnes par ligne
            ncol = plot_idx % nb_col  # On alterne entre les colonnes
            plot_idx += 1
            gs01x = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs01[nrow, ncol], wspace=0.1, hspace=0.1)
            axes = [fig.add_subplot(gs01x[0, 0]), fig.add_subplot(gs01x[0, 1])]

            data_chr_by_iso = chr_dic.copy()
            plot_density(data_chr_by_iso,axes, iso, color_dict, args)

        legend_ax = fig.add_subplot(gs0[99, 0])
        legend_ax.axis('off')  # Cacher les axes
        # Créer les éléments de légende manuellement
        handles = []
        labels = []
        chromosomes = sorted(list(color_dict.keys()))
        for chr_name in chromosomes:
            handle = mpatches.Patch(color=color_dict[chr_name], label=chr_name)
            handles.append(handle)
            labels.append(chr_name)

        # Créer la légende
        legend = legend_ax.legend(handles, labels, loc='center', ncol=6, fontsize=10, 
                                frameon=True, title="Chromosomes")
        legend.get_title().set_fontsize(12) 

        fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05) 
        fig.savefig(f"{output_dir}/{methode}.svg", dpi=300)
        plt.close(fig)
    
   

    ###################################### PLOT 3 TE_class by Methode ######################################

    unique_te_class = set()
    for methode, te_class_dic in data.items(): unique_te_class.update(te_class_dic.keys())

    all_class_te = sorted(list(unique_te_class))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(len(all_class_te))]
    color_dict = dict(zip(all_class_te, colors))
    color_dict['all_te_class'] = 'black'

    args = {'line_size': 0.5, 'marker_size': 1}

    nb_col = 2
    nb_rows = (len(data.keys()) + nb_col - 1) // nb_col

    fig = plt.figure(figsize=(nb_col*8,8))
    gs0 = fig.add_gridspec(100, 1, wspace=0.5, hspace=4)
    ax_title = fig.add_subplot(gs0[8, 0])
    ax_title.axis('off')       
    ax_title.set_title("Densité de TE autour des inversions par class de TE", fontsize=18) 

    gs01 = gridspec.GridSpecFromSubplotSpec(nb_rows, nb_col, subplot_spec=gs0[22:70, 0], wspace=0.3, hspace=0.8)
    plot_idx = 0 


    for methode, te_class_dic in data.items():
        nrow = plot_idx // nb_col  # On remplit plusieurs colonnes par ligne
        ncol = plot_idx % nb_col  # On alterne entre les colonnes
        plot_idx += 1
        gs01x = gridspec.GridSpecFromSubplotSpec(2,2 , subplot_spec=gs01[nrow, ncol], wspace=0.1, hspace=0.1, height_ratios=[10,4])
        axes = [fig.add_subplot(gs01x[0, 0]), fig.add_subplot(gs01x[0, 1]), fig.add_subplot(gs01x[1, 0]), fig.add_subplot(gs01x[1, 1])]
        data_te_class = {te_class : data[methode][te_class]['all_iso']['all_chr'] for te_class in data[methode].keys()}
        



        plot_density(data_te_class,axes, methode, color_dict, args)


    legend_ax = fig.add_subplot(gs0[95, 0])
    legend_ax.axis('off')  # Cacher les axes
    handles, labels = [], []
    all_te_class = sorted(list(color_dict.keys()))
    for te_class in all_te_class:
        handle = mpatches.Patch(color=color_dict[te_class], label=te_class)
        handles.append(handle)
        labels.append(te_class)

    # Créer la légende
    legend = legend_ax.legend(handles, labels, loc='center', ncol=6, fontsize=10, 
                            frameon=True, title="Class Te")
    legend.get_title().set_fontsize(12) 

    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.05) 
    fig.savefig(f"{output_dir}/Te_class.svg", dpi=300)
    plt.close(fig)


    
# Point d'entrée principal
def execut(data):
    inv_df = data['inversions']
    inv_colors = data['colors']  
    chr_size_df = data['chr_lengths']
    output_dir = os.path.join(data['outpath'], 'inv_breakpoints')
    os.makedirs(output_dir, exist_ok=True)


    # Lecture des données TE
    te_pathfile =  data['te_path']
    te_df = pd.read_csv(te_pathfile, sep='\t')
    te_df['iso'] = te_df['iso'].str.split('_').str[:2].str.join('_')
    te_df.drop(columns=['oid'], inplace=True)
    #te_df = te_df.rename(columns={'class': 'family'})
    te_df['class'] = te_df['class'].apply(lambda f: 'DNA' if isinstance(f, str) and f.startswith('DNA/') else 
                'LTR' if f in ['LTR','ClassI/Unclassified', 'LTR/Ty3', 'LTR/Copia', 'LTR/TRIM'] else f)
    te_df['ID'] = te_df.index

    
    gene = False
    if gene : 
        te_pathfile = data['gene_path']
        te_df = pd.read_csv(te_pathfile, sep='\t')
        te_df['iso'] = te_df['iso'].str.split('_').str[:2].str.join('_')
        te_df = te_df[te_df['type'] == 'gene'].copy()


    # Lecture des longueurs de chromosomes
    chr_size_dict = defaultdict(lambda: defaultdict(dict))
    for iso, chr, size in zip(chr_size_df['iso'], chr_size_df['chr'], chr_size_df['size']):
        chr_size_dict[iso][chr] = size



    full_data = make_data(inv_df, te_df, chr_size_dict)
    save_plot(full_data,inv_colors, output_dir)

