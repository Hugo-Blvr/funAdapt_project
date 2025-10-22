#!/usr/bin/env python3

#================================================================================
#Genomic Inversion Filtering and Classification Pipeline
#================================================================================
#
#Description:
#    Python pipeline for filtering, and classifying genomic inversions
#    from alignment data (MAF/SyRI formats) with transposable element filtering
#    and lineage-based classification.
#
#Features:
#    • Multi-format support (MAF, SyRI TSV) with automatic detection
#    • Reciprocal inversion merging and spatial clustering
#    • TE-associated inversion filtering (bidirectional and coverage-based)
#    • Parallel processing by chromosome
#    • Flexible JSON configuration
#    • Morphotype (A/B) and lineage classification
#
#Workflow:
#    1. Load and format inversion data
#    2. Filter englobing/redundant inversions
#    3. Merge reciprocal inversions (T↔Q pairs)
#    4. Merge close inversions and filter by size
#    5. Spatial clustering and ninv assignment
#    6. TE filtering (optional): isTE and inTE removal
#    7. Export filtered inversions
#
#Usage:
#    python pipeline.py -i <input_file> -o <output_dir> [options]
#    python pipeline.py --help
#    python pipeline.py --create-default-config config.json
#
#Required Arguments:
#    -i, --input <file>         Alignment file (MAF or SyRI TSV)
#
#Optional Arguments:
#    -o, --output <dir>         Output directory (default: current)
#    -c, --config <file>        JSON configuration file
#    --te-file <file>           TE annotations TSV (overrides config)
#    --lineage-file <file>      Lineage mapping TSV (overrides config)
#    --create-default-config    Generate default config and exit
#
#Output Files:
#    INV_*_nofilter.tsv         Raw formatted inversions (optional)
#    INV_*_clean_wTE.tsv        Pre-TE filtering inversions (optional)
#    INV_*_isTE.tsv             Inversions matching TEs (optional)
#    INV_*_inTE.tsv             Inversions covered by TEs (optional)
#    INV_*_clean.tsv            Final filtered inversions (main output)
#    TE_*_isTE.tsv              TEs matching inversions (optional)
#    TE_*_inTE.tsv              TEs covering inversions (optional)
#
#Key Algorithms:
#    • Sweep-line algorithm for reciprocal inversion detection
#    • Spatial bucketing (50kb) with BFS clustering
#    • Bidirectional overlap checking (≥85% default)
#    • Binary search for TE matching
#
#Configuration Example:
#    {
#      "filtering": {"min_inversion_size": 150},
#      "merging": {"reciprocal_overlap_ratio": 0.85, "fusion_gap_threshold": 100},
#      "clustering": {"min_overlap_ratio": 0.85, "spatial_bucket_size": 50000},
#      "te_filtering": {"enabled": true, "overlap_threshold": 90.0},
#      "output": {"save_nofilter": false, "save_wTE": false, "save_TE_files": false}
#    }
#
#TE Filtering Criteria:
#    isTE: Bidirectional overlap ≥90% (both inversion AND TE covered)
#    inTE: Unidirectional coverage ≥90% (inversion covered by single TE)
#
#Performance:
#    • Parallel processing: auto-detects cores (n_cores - 1)
#    • Vectorized operations with numpy/pandas
#    • Optimized for datasets with multiple chromosomes
#
#Dependencies:
#    Python ≥3.7, pandas, numpy, multiprocessing
#
#Author:  Hugo Bellavoir
#Email:   bellavoirh@gmail.com
#Version: 1.0
#License: MIT
#================================================================================

import sys
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Tuple, Optional, Literal, List, Set
from multiprocessing import Pool, cpu_count
import argparse
from config import PipelineConfig

def detect_alignment_format(filepath: Path) -> Literal['maf', 'syri']:
    """
    Detect alignment file format.
    
    Args:
        filepath: Path to alignment file
        
    Returns:
        'maf' or 'syri'
        
    Raises:
        ValueError: If format is unknown
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    if first_line.startswith('##maf'):
        return 'maf'
    
    if first_line == 'target\tt_chr\tt_start\tt_end\tquery\tq_chr\tq_start\tq_end':
        return 'syri'
    
    raise ValueError(
        f"Unknown file format: {filepath}\n"
        f"Expected:\n"
        f"  - MAF: starts with '##maf'\n"
        f"  - SyRI TSV: header 'target  t_chr  t_start  t_end  query  q_chr  q_start  q_end' --> used inv_calling.sh\n"
        f"Got: {first_line[:80]}"
    )


def are_same_position(start1: int, end1: int, start2: int, end2: int, min_overlap_ratio: float) -> bool:
    """
    Determine if two genomic intervals overlap sufficiently to be considered identical.
    
    Two intervals are considered identical if their overlap represents at least
    `min_overlap_ratio` of the length of EACH interval.
    
    Args:
        start1: Start position of the first interval (inclusive)
        end1: End position of the first interval (exclusive)
        start2: Start position of the second interval (inclusive)
        end2: End position of the second interval (exclusive)
        min_overlap_ratio: Minimum overlap ratio required (between 0 and 1)
                          e.g., 0.85 means 85% of each interval must overlap
    Returns:
        True if intervals overlap sufficiently, False otherwise
    Example:
        >>> are_same_position(100, 200, 110, 210, 0.85)
        True  # 90bp overlap out of 100bp = 90%
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)   
    
    # No overlap
    if overlap_start >= overlap_end: return False
    
    overlap_length = overlap_end - overlap_start
    interval1_length = end1 - start1
    interval2_length = end2 - start2
    
    # Overlap must be >= min_overlap_ratio for BOTH intervals
    return (overlap_length >= min_overlap_ratio * interval1_length and 
            overlap_length >= min_overlap_ratio * interval2_length)


##########################################################################################################################
# Formatting and filtering for Syri Inv
##########################################################################################################################


def format_df_myinv(df_pathfile: str, config: PipelineConfig) -> pd.DataFrame:
    """
    Load and format inversion data from a TSV file with filtering and preprocessing.
    
    This function performs the following operations:
    1. Load TSV data and sort by target, query, chromosome, and start position
    2. Remove duplicate rows
    3. Extract isolate names from target/query columns (specific format: keeps first two parts)
    4. Filter out excluded chromosomes and ensure consistency between target and query isolates
    
    Args:
        df_pathfile: Path to the input TSV file containing inversion data
        config: Pipeline configuration object
            - excluded_chromosomes: List of chromosome names to exclude from analysis
                             Defaults to ['chr2003', 'chr2016', 'chr2017']
    
    Returns:
        Formatted DataFrame with columns: target, query, t_chr, t_start, t_end, 
                                         q_chr, q_start, q_end, ID
        Sorted by target, query, t_chr, t_start
    
    Note:
        The function expects target/query format like "isolate_version_..." and extracts
        only "isolate_version" (first two underscore-separated parts).
    """

    excluded_chromosomes = config.input.excluded_chromosomes

    # Load and initial sorting
    df = pd.read_csv(df_pathfile, sep='\t')
    df = df.sort_values(by=['target', 'query', 't_chr', 't_start']).reset_index(drop=True)
    
    # Remove duplicates
    df_no_duplicates = df.drop_duplicates()
    df_no_duplicates = df_no_duplicates.sort_values(by=['target', 'query', 't_chr', 't_start']).reset_index(drop=True)
    
    # Assign unique IDs
    df_no_duplicates['ID'] = df_no_duplicates.index
    
    # Extract isolate names (keep first two parts: "isolate_version")
    # TODO: Make this configurable instead of hardcoded
    split_target = df_no_duplicates['target'].str.split('_')
    df_no_duplicates['target'] = split_target.str[0] + '_' + split_target.str[1]
    
    split_query = df_no_duplicates['query'].str.split('_')
    df_no_duplicates['query'] = split_query.str[0] + '_' + split_query.str[1]
    
    # Get unique isolates
    isolates = np.union1d(df_no_duplicates['target'].unique(), df_no_duplicates['query'].unique())
    
    # Filter: keep only rows where both target and query are in isolates list
    # and exclude specified chromosomes
    df_filtered = df_no_duplicates.query("target in @isolates and query in @isolates and t_chr not in @excluded_chromosomes").copy()
    df_final = df_filtered.sort_values(by=['target', 'query', 't_chr', 't_start']).reset_index(drop=True)
    
    return df_final


def final_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform inversion data into a unified long format with both target and query perspectives.
    
    This function duplicates each inversion entry to represent both directions:
    - Target perspective: target as 'iso', with its chromosome and positions
    - Query perspective: query as 'iso', with its chromosome and positions
    
    This unified format facilitates downstream analysis by treating all genomic positions
    uniformly regardless of whether they were originally target or query.
    
    Args:
        df: DataFrame with columns: target, t_chr, t_start, t_end, query, q_chr, q_start, q_end, ID
        outfile: Optional path to save the output TSV file. If None, no file is written.
    
    Returns:
        Long-format DataFrame with columns: iso, chr, start, end, mapon, ID
        - iso: The isolate being considered
        - chr: Chromosome of the isolate
        - start/end: Genomic coordinates in the isolate
        - mapon: The other isolate in the inversion pair
        - ID: Unique identifier linking both perspectives of the same inversion
        
        Sorted by iso, chr, start
    
    Example:
        Input row: target=A, t_chr=chr1, t_start=100, t_end=200, query=B, q_chr=chr1, q_start=300, q_end=400, ID=1
        Output rows:
            iso=A, chr=chr1, start=100, end=200, mapon=B, ID=1
            iso=B, chr=chr1, start=300, end=400, mapon=A, ID=1
    """
    # Reassign IDs based on current index
    df['ID'] = df.index
    
    # Create target perspective (T = Target)
    df_target = df[['target', 't_chr', 't_start', 't_end', 'query', 'ID']].copy()
    
    # Create query perspective (Q = Query)
    df_query = df[['query', 'q_chr', 'q_start', 'q_end', 'target', 'ID']].copy()
    
    # Standardize column names
    unified_columns = ['iso', 'chr', 'start', 'end', 'mapon', 'ID']
    df_target.columns = unified_columns
    df_query.columns = unified_columns
    
    # Concatenate both perspectives
    format_data = pd.concat([df_target, df_query], ignore_index=True)
    format_data = format_data.sort_values(by=['iso', 'chr', 'start']).reset_index(drop=True)
    
    return format_data.reset_index(drop=True)


def filter_englobing_inversions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove inversions that exactly encompass multiple smaller inversions.
    
    An inversion A is considered "englobing" (encompassing) if:
    1. It contains multiple smaller inversions B1, B2, ... Bn
    2. A's boundaries exactly match the min/max boundaries of the contained inversions
    3. This is true for BOTH target and query coordinates
    
    This filtering removes redundant large inversions that are actually artifacts
    of fragmented detection of the same region.
    
    Args:
        df: DataFrame with inversion data containing columns:
            target, query, t_chr, t_start, t_end, q_start, q_end
    
    Returns:
        Filtered DataFrame with englobing inversions removed, maintaining original columns
    
    Example:
        If inversion A spans target:[100-300] query:[400-600]
        and contains B1:[100-200, 400-500] and B2:[200-300, 500-600]
        where A.start = min(B1.start, B2.start) and A.end = max(B1.end, B2.end)
        Then A is removed as it's an artifact encompassing B1 and B2
    """
    result_df = df.copy()
    
    # Create composite key for target-query pairs
    result_df['target_query_pair'] = result_df['target'] + '_' + result_df['query']
    
    indices_to_remove = set()
    
    # Group by target-query pairs and chromosome
    for (target_query_pair, chromosome), group in result_df.groupby(['target_query_pair', 't_chr']):
        if len(group) <= 1: continue

        # Convert to numpy arrays for vectorized operations
        group_indices = group.index.values
        t_starts = group['t_start'].values
        t_ends = group['t_end'].values
        q_starts = group['q_start'].values
        q_ends = group['q_end'].values
        
        # Check each inversion for englobing behavior
        for i, current_idx in enumerate(group_indices):
            # Vectorized mask: find inversions potentially contained by current inversion
            is_contained_mask = (
                (t_starts >= t_starts[i]) & (t_ends <= t_ends[i]) & 
                (q_starts >= q_starts[i]) & (q_ends <= q_ends[i]) &
                (group_indices != current_idx))  # Exclude self
            
            if not is_contained_mask.any(): continue
            
            # Verify exact boundary match with contained inversions
            sub_indices = is_contained_mask.nonzero()[0]
            
            current_matches_contained_boundaries = (
                t_starts[i] == t_starts[sub_indices].min() and t_ends[i] == t_ends[sub_indices].max() and
                q_starts[i] == q_starts[sub_indices].min() and q_ends[i] == q_ends[sub_indices].max())
            
            if current_matches_contained_boundaries: indices_to_remove.add(current_idx)
    
    # Remove englobing inversions
    result_df = result_df[~result_df.index.isin(indices_to_remove)]
    
    return result_df.drop(columns=['target_query_pair'])


def handle_duplicate_inversions(df: pd.DataFrame, min_overlap_ratio: float) -> pd.DataFrame:
    """
    Handle reciprocal inversions (same genomic pair with swapped target/query roles) and merge them.
    
    When genome A is aligned to genome B and vice versa, the same inversion can be detected
    twice with roles reversed. This function:
    1. Identifies such reciprocal inversions based on position overlap
    2. Merges them by taking min(start) and max(end) for both genomes
    3. Keeps only one merged version per reciprocal pair
    
    Args:
        df: DataFrame with inversion data containing columns:
            target, query, t_chr, t_start, t_end, q_chr, q_start, q_end, ID
        min_overlap_ratio: Minimum overlap ratio to consider two inversions as duplicates (0-1)
                          e.g., 0.85 means 85% reciprocal overlap required
    
    Returns:
        Filtered DataFrame with merged reciprocal inversions
        
    Algorithm:
        Uses a sweep-line algorithm with spatial indexing for efficiency:
        - Group inversions by genome pair (regardless of target/query order)
        - For each pair, compare forward direction (A→B) with reverse (B→A)
        - Use sorted positions to avoid O(n²) comparisons
        - Merge coordinates when reciprocal overlap >= threshold
    
    Example:
        Input:
            A→B: target_pos=[100-200], query_pos=[300-400], ID=1
            B→A: target_pos=[305-395], query_pos=[105-195], ID=2
        Output (merged):
            A→B: target_pos=[100-200], query_pos=[300-400], ID=1
            (ID=2 removed as duplicate)
    """
    data = df.copy()
    
    # Create unique key for genome pairs (order-independent)
    data['pair_key'] = data.apply(
        lambda row: f"{min(row['target'], row['query'])}_{max(row['target'], row['query'])}_{row['t_chr']}", axis=1)
    
    ids_to_remove = set()
    
    # Process each genome pair separately
    for pair_key, pair_data in data.groupby('pair_key'):
        if len(pair_data) < 2: continue
        
        # Separate forward and reverse directions
        first_target = pair_data['target'].iloc[0]
        forward_direction = pair_data[pair_data['target'] == first_target]
        reverse_direction = pair_data[pair_data['target'] != first_target]
        
        if reverse_direction.empty: continue
        
        # Sort for efficient sweep-line algorithm
        forward_sorted = forward_direction.sort_values('t_start')
        reverse_sorted = reverse_direction.sort_values('q_start')
        
        # Convert to numpy for faster access
        forward_data = forward_sorted[['t_start', 't_end', 'q_start', 'q_end', 'ID']].values
        reverse_data = reverse_sorted[['q_start', 'q_end', 't_start', 't_end', 'ID']].values
        
        # Sweep-line algorithm to find reciprocal matches
        j = 0  # Pointer for reverse_data
        
        for i, forward_inv in enumerate(forward_data):
            t_start1, t_end1, q_start1, q_end1, id1 = forward_inv
            
            # Skip reverse inversions that end before current forward starts
            while j < len(reverse_data) and reverse_data[j][1] < t_start1: j += 1 # q_end2 < t_start1
               
            # Calculate search boundary (max start position for potential matches)
            max_search_start = t_start1 + (t_end1 - t_start1) * (1 - min_overlap_ratio)
            
            # Search for reciprocal matches
            for k in range(j, len(reverse_data)):
                q_start2, q_end2, t_start2, t_end2, id2 = reverse_data[k]
                
                # Stop if reverse inversion starts too far right
                if q_start2 > max_search_start: break
                
                # Check if positions match in both directions
                positions_match = (
                    are_same_position(t_start1, t_end1, q_start2, q_end2, min_overlap_ratio) and
                    are_same_position(q_start1, q_end1, t_start2, t_end2, min_overlap_ratio)
)
                
                if positions_match:
                    # Merge coordinates: take min(start) and max(end) for both genomes
                    forward_idx = forward_sorted.index[i]
                    
                    merged_t_start = min(t_start1, q_start2)
                    merged_t_end = max(t_end1, q_end2)
                    merged_q_start = min(q_start1, t_start2)
                    merged_q_end = max(q_end1, t_end2)
                    
                    data.loc[forward_idx, ['t_start', 't_end', 'q_start', 'q_end']] = [
                        merged_t_start, merged_t_end, merged_q_start, merged_q_end
                    ]
                    
                    ids_to_remove.add(int(id2))
                    break  # Only merge with first match
    
    # Remove duplicate entries
    filtered_data = data[~data['ID'].isin(ids_to_remove)].drop('pair_key', axis=1)
    
    return filtered_data.reset_index(drop=True)


def filter_data_size(df: pd.DataFrame, min_size: int) -> pd.DataFrame:
    """
    Filter inversions by minimum size and ensure they appear in at least two genomes.
    
    This function applies two sequential filters:
    1. Size filter: Remove inversions smaller than min_size base pairs
    2. Reciprocity filter: Keep only inversions with reciprocal evidence
       (i.e., same ID appears at least twice, indicating detection in both genomes)
    
    Args:
        df: Long-format DataFrame with columns: iso, chr, start, end, mapon, ID
        min_size: Minimum inversion size in base pairs (default: 150)
    
    Returns:
        Filtered DataFrame sorted by iso, chr, mapon, start
        Contains only inversions >= min_size bp that appear in multiple genomes
    
    Note:
        IDs appearing only once are removed because they lack reciprocal evidence,
        suggesting potential false positives or alignment artifacts.
    """
    # Filter by minimum size
    df_size_filtered = df[(df['end'] - df['start']) >= min_size].copy()
    
    # Count occurrences of each ID
    id_counts = df_size_filtered['ID'].value_counts()
    
    # Keep only IDs appearing more than once (reciprocal evidence)
    ids_to_keep = id_counts[id_counts > 1].index
    df_reciprocal_filtered = df_size_filtered[df_size_filtered['ID'].isin(ids_to_keep)]
    
    # Sort and reset index
    df_final = df_reciprocal_filtered.sort_values(
        by=['iso', 'chr', 'mapon', 'start'], ignore_index=True)
    
    return df_final


def fusionner_inversions(df: pd.DataFrame, fusion_gap_threshold: int) -> pd.DataFrame:
    """
    Merge inversions that are separated by less than fusion_gap_threshold bp on both sides of the genome pair.
    
    When inversions are close together, they likely represent a single
    fragmented inversion rather than multiple distinct events. This function merges such
    inversions BUT only if the merge is valid in BOTH directions to preserve ID pairing.
    
    Args:
        df: Long-format DataFrame with columns: iso, chr, start, end, mapon, ID
        fusion_gap_threshold: Maximum gap in bp to merge inversions    
    Returns:
        DataFrame with merged inversions, maintaining the same structure
        Merged inversions keep the minimum ID from the group
    
    Algorithm:
        For each genome pair (A-B):
        1. Process forward direction (A→B) to find close inversions
        2. Verify that reverse direction (B→A) also allows merging
        3. Only merge if BOTH directions have < 100bp gaps
        4. Update coordinates to span min(start) to max(end) for both genomes
        
    Example with fusion_gap_threshold=100:
        Genome A: inv1[100-200], inv2[250-350]  (gap = 50bp)
        Genome B: inv1[300-400], inv2[450-550]  (gap = 50bp)
        Result: Merged into single inversion spanning [100-350] in A and [300-550] in B
    """
    df = df.copy()
    merged_results = []
    merged_ids = set()
    
    # Get all unique genome pairs (order-independent)
    genome_pairs = set(
        tuple(sorted([row['iso'], row['mapon']])) 
        for _, row in df.iterrows()
    )
    
    # Process each genome pair
    for genome1, genome2 in genome_pairs:
        for chr_name in df['chr'].unique():
            # Get both directions for this chromosome
            forward = df[(df['iso'] == genome1) & (df['mapon'] == genome2) & 
                (df['chr'] == chr_name)].sort_values(by=['start', 'end'])
            
            reverse = df[(df['iso'] == genome2) & (df['mapon'] == genome1) & 
                (df['chr'] == chr_name)].sort_values(by=['start', 'end'])
            
            if forward.empty or reverse.empty: continue
            
            # Create ID lookup for reverse direction
            reverse_dict = {row['ID']: row for _, row in reverse.iterrows()}
            
            # Sweep through forward direction to identify merge groups
            i = 0
            forward_list = list(forward.iterrows())
            
            while i < len(forward_list):
                if forward_list[i][1]['ID'] in merged_ids:
                    i += 1
                    continue
                
                # Start a new merge group
                merge_group = [i]
                j = i + 1
                
                # Try to extend the merge group
                while j < len(forward_list):
                    if forward_list[j][1]['ID'] in merged_ids:
                        j += 1
                        continue
                    
                    current_inv = forward_list[merge_group[-1]][1]
                    next_inv = forward_list[j][1]
                    
                    # Calculate gap in forward direction
                    gap_forward = next_inv['start'] - current_inv['end']
                    
                    if gap_forward >= fusion_gap_threshold: break  # Too far apart
                        
                    # Verify gap in reverse direction for same IDs
                    if (current_inv['ID'] in reverse_dict and 
                        next_inv['ID'] in reverse_dict):
                        
                        current_rev = reverse_dict[current_inv['ID']]
                        next_rev = reverse_dict[next_inv['ID']]
                        
                        # Order by position in reverse direction
                        if current_rev['start'] > next_rev['start']:
                            current_rev, next_rev = next_rev, current_rev
                        
                        gap_reverse = next_rev['start'] - current_rev['end']
                        
                        # Only merge if BOTH directions have small gaps
                        if gap_reverse < fusion_gap_threshold:
                            merge_group.append(j)
                            j += 1
                        else: break
                    else: break
                
                # Apply merge if group has multiple inversions
                if len(merge_group) > 1:
                    group_inversions = [forward_list[idx][1] for idx in merge_group]
                    group_ids = [inv['ID'] for inv in group_inversions]
                    merged_id = min(group_ids)
                    
                    # Merge forward direction
                    merged_results.append({
                        'iso': genome1, 'chr': chr_name,
                        'start': min(inv['start'] for inv in group_inversions),
                        'end': max(inv['end'] for inv in group_inversions),
                        'mapon': genome2, 'ID': merged_id
                    })
                    
                    # Merge reverse direction
                    reverse_inversions = [reverse_dict[id_val] for id_val in group_ids]
                    merged_results.append({
                        'iso': genome2, 'chr': chr_name,
                        'start': min(inv['start'] for inv in reverse_inversions),
                        'end': max(inv['end'] for inv in reverse_inversions),
                        'mapon': genome1, 'ID': merged_id
                    })
                    
                    merged_ids.update(group_ids)
                
                # Move to next unprocessed inversion
                i = j if len(merge_group) > 1 else i + 1
    
    # Add unmerged inversions
    for _, row in df.iterrows():
        if row['ID'] not in merged_ids: merged_results.append(row.to_dict())
    
    merge_df = pd.DataFrame(merged_results)

    merge_df = merge_df.sort_values(
        by=['iso', 'chr', 'mapon', 'start'], ignore_index=True
    )
    
    return merge_df


##########################################################################################################################
# Formatting and filtering for maf file Inv
##########################################################################################################################


class InversionDetector:
    def __init__(self):
        self.block_num = 0
        self.global_pair_id = 0
        self.results = []  # Store the results
        
    def process_file(self, filename: str = None):
        """Process the alignment file line by line"""
        file_handle = open(filename, 'r') if filename else sys.stdin
        
        try:
            in_block = False
            lines_by_chr = defaultdict(list)
            chrs_in_block = set()
            
            for line in file_handle:
                line = line.strip()
                
                # New alignment section
                if line.startswith('a'):
                    if in_block:
                        self._process_block(lines_by_chr, chrs_in_block)
                    
                    in_block = True
                    self.block_num += 1
                    
                    # Reset for the new block
                    lines_by_chr.clear()
                    chrs_in_block.clear()
                    continue
                
                # Sequence line
                elif line.startswith('s'):
                    if in_block:
                        parts = line.split()
                        if len(parts) >= 6:
                            name_parts = parts[1].split('.')
                            if len(name_parts) >= 2:
                                isolat = name_parts[0]
                                chr_name = name_parts[1]
                                start = int(parts[2])
                                size = int(parts[3])
                                strand = parts[4]
                                
                                chrs_in_block.add(chr_name)
                                lines_by_chr[chr_name].append({
                                    'isolat': isolat,
                                    'chr': chr_name,
                                    'start': start,
                                    'strand': strand,
                                    'size': size,
                                    'end': start + size - 1
                                })
                
                # Ignore empty lines and comments
                elif line == '' or line.startswith('#'):
                    continue
            
            # Process the last block
            if in_block:
                self._process_block(lines_by_chr, chrs_in_block)
                
        finally:
            if filename:
                file_handle.close()
        
        # Return the DataFrame directly
        return self._create_dataframe()
    
    def _process_block(self, lines_by_chr: Dict[str, List[Dict]], 
                      chrs_in_block: Set[str]):
        """Process an alignment block to detect inversions"""
        
        for chr_name in chrs_in_block:
            alignments = lines_by_chr[chr_name]
            
            # Separate by orientation
            plus_alignments = [aln for aln in alignments if aln['strand'] == '+']
            minus_alignments = [aln for aln in alignments if aln['strand'] == '-']
            
            # If both orientations are present, create pairs
            if plus_alignments and minus_alignments:
                self._create_inversion_pairs(plus_alignments, minus_alignments, chr_name)
    
    def _create_inversion_pairs(self, plus_alignments: List[Dict], 
                               minus_alignments: List[Dict], chr_name: str):
        """Create all possible pairs between opposite orientations"""
        
        # Create all possible pairs between + and - strands (multalgn --> pairalgn)
        for plus_aln in plus_alignments:
            for minus_aln in minus_alignments:
                self.global_pair_id += 1
                # Add both lines with the same ID
                self._add_pair(plus_aln, minus_aln, self.global_pair_id)
                self._add_pair(minus_aln, plus_aln, self.global_pair_id)
    
    def _add_pair(self, iso_aln: Dict, mapon_aln: Dict, pair_id: int):
        """Add a pair of alignments to the results"""
        
        iso = iso_aln['isolat']
        mapon = mapon_aln['isolat']
        
        # Add the line to results
        self.results.append({
            'iso': iso,
            'chr': iso_aln['chr'],
            'start': iso_aln['start'],
            'end': iso_aln['end'],
            'mapon': mapon,
            'ID': pair_id
        })
    
    def _create_dataframe(self):
        """Create and return a sorted pandas DataFrame"""
        # Sort by iso, chr, mapon
        sorted_results = sorted(self.results, key=lambda x: (x['iso'], x['chr'], x['mapon'], x['start']))
        
        df = pd.DataFrame(sorted_results)
        return df


##########################################################################################################################
# Inversion Clustering and Lineage-Based Classification
##########################################################################################################################

 
def assign_inversion_type(group: pd.DataFrame) -> pd.DataFrame:
    """
    Assign inversion type based on lineage composition within a group.
    
    Classifies inversions according to whether they occur within or between
    Phytophthora drechsleri lineages (Pd1, Pd2).
    
    Args:
        group: DataFrame group with 'type' column containing pairwise lineage labels
               (e.g., 'Pd1_Pd1', 'Pd1_Pd2', 'Pd2_Pd2')
    
    Returns:
        Same DataFrame with updated 'type' column reflecting group-level classification
    
    Classification rules:
        - All 'Pd1_Pd1' → 'intra_Pd1' (within Pd1 lineage)
        - All 'Pd2_Pd2' → 'intra_Pd2' (within Pd2 lineage)
        - All 'Pd1_Pd2' → 'intra_Pd' (between Pd lineages)
        - Mixed types → 'inter_Pd' (across different species/lineages)
    """
    types = set(group['type'])
    
    if types == {'Pd1_Pd1'}: group.loc[:, 'type'] = 'intra_Pd1'
    elif types == {'Pd2_Pd2'}: group.loc[:, 'type'] = 'intra_Pd2'
    elif types <= {'Pd1_Pd1', 'Pd2_Pd2', 'Pd1_Pd2'} and 'Pd1_Pd2' in types: group.loc[:, 'type'] = 'intra_Pd'
    else: group.loc[:, 'type'] = 'inter_Pd'
    
    return group


def _process_inversions_parallel(df: pd.DataFrame, chromosomes: np.ndarray, 
    min_overlap_ratio: float, bucket_size: int, n_cores: int) -> pd.DataFrame:
    """
    Process inversions in parallel by splitting across chromosomes.
    
    Args:
        df: Full inversion DataFrame
        chromosomes: Array of chromosome names
        min_overlap_ratio: Minimum overlap ratio for clustering
        bucket_size: Size of spatial buckets (e.g., 50000 for 50kb)
        n_cores: Number of parallel processes
    
    Returns:
        Processed DataFrame with globally unique ninv values
    """
    # Split chromosomes into chunks
    chr_chunks = np.array_split(chromosomes, n_cores)
    chunks = []
    
    for chunk_chrs in chr_chunks:
        chunk_df = df[df['chr'].isin(chunk_chrs)].copy()
        chunks.append((chunk_df, min_overlap_ratio, bucket_size))
    
    # Parallel processing
    with Pool(n_cores) as pool:
        results = pool.map(_process_inversions_chunk, chunks)
    
    # Concatenate all results
    merged_df = pd.concat(results, ignore_index=True)

    # Renumber ninv using groupby (un seul pass)
    ninv_offset = 0
    renumbered_groups = []

    for chr_name, group in merged_df.groupby('chr', sort=True):
        group = group.copy()
        unique_ninvs = sorted(group['ninv'].unique())
        ninv_map = {old: new + ninv_offset for new, old in enumerate(unique_ninvs, 1)}
        group['ninv'] = group['ninv'].map(ninv_map)
        ninv_offset += len(unique_ninvs)
        renumbered_groups.append(group)

    return pd.concat(renumbered_groups, ignore_index=True)


def _process_inversions_chunk(args: Tuple[pd.DataFrame, float, int]) -> pd.DataFrame:
    """
    Process a chunk of inversion data to assign ninv and morph values.
    
    Uses spatial indexing with 50kb buckets and BFS clustering to efficiently
    group inversions by ID and spatial overlap.
    
    Args:
        args: Tuple of (DataFrame, min_overlap_ratio, bucket_size)
              DataFrame must contain: iso, chr, start, end, mapon, ID
              min_overlap_ratio: Minimum overlap for considering same position (0-1)
              bucket_size: Size of spatial buckets (e.g., 50000 for 50kb)
    
    Returns:
        DataFrame with assigned 'ninv' and 'morph' columns
        
    Algorithm:
        1. Build spatial index with 50kb buckets per (iso, chr)
        2. For each unprocessed inversion:
           a. Start BFS cluster with morphotype 'A'
           b. Find inversions with same ID (assign opposite morph if reciprocal)
           c. Find spatially overlapping inversions (same iso/chr)
           d. Assign same ninv to entire cluster
    """
    df, min_overlap_ratio, bucket_size  = args
    
    # Build ID-based index
    id_groups = df.groupby('ID').groups
    
    # Build spatial index with 50kb buckets
    spatial_buckets = defaultdict(lambda: defaultdict(list))
    
    for idx, row in df.iterrows():
        bucket = row['start'] // bucket_size
        key = (row['iso'], row['chr'])
        spatial_buckets[key][bucket].append((idx, row['start'], row['end']))
    
    # Sort buckets by start position
    for key in spatial_buckets:
        for bucket in spatial_buckets[key]:
            spatial_buckets[key][bucket].sort(key=lambda x: x[1])
    
    ninv_counter = 1
    processed = set()
    
    # Process inversions in order
    for idx in df.index:
        if idx in processed: continue
        
        # BFS to build cluster
        to_process = deque([idx])
        cluster_indices = set()
        
        df.loc[idx, 'morph'] = 'A'
        
        while to_process: 
            current_idx = to_process.popleft()
            
            if current_idx in processed: continue
            
            processed.add(current_idx)
            cluster_indices.add(current_idx)
            
            current_row = df.loc[current_idx]
            current_morph = df.loc[current_idx, 'morph']
            
            # 1. Find inversions with same ID (reciprocal pairs)
            current_id = current_row['ID']
            if current_id in id_groups:
                for same_id_idx in id_groups[current_id]:
                    if same_id_idx in processed: continue
                    
                    same_id_row = df.loc[same_id_idx]
                    
                    # Determine morphotype based on genome pair orientation
                    is_reciprocal_pair = (
                        (same_id_row['iso'], same_id_row['mapon']) == 
                        (current_row['mapon'], current_row['iso'])
                    )
                    
                    # Paire réciproque : inverser la morphologie
                    if is_reciprocal_pair: new_morph = 'A' if current_morph == 'B' else 'B'
                    # Même direction ou overlap spatial : forcer 'A'
                    # (seules les paires réciproques peuvent avoir morph='B')
                    else: new_morph = 'A'


                    df.loc[same_id_idx, 'morph'] = new_morph
                    to_process.append(same_id_idx)
            
            # 2. Find spatially overlapping inversions
            current_iso = current_row['iso']
            current_chr = current_row['chr']
            current_start = current_row['start']
            current_end = current_row['end']
            
            key = (current_iso, current_chr)
            if key in spatial_buckets:
                # Search in adjacent buckets
                start_bucket = current_start // bucket_size
                end_bucket = current_end // bucket_size

                for bucket_id in range(start_bucket, end_bucket + 1):
                    if bucket_id not in spatial_buckets[key]: continue
                    
                    for candidate_idx, start, end in spatial_buckets[key][bucket_id]:
                        if candidate_idx in processed: continue
                        
                        # Early stopping: candidate starts too far right
                        if start > current_end + min_overlap_ratio * (current_end - current_start): break
                        
                        # Skip if candidate ends before current starts
                        if end < current_start - min_overlap_ratio * (current_end - current_start): continue
                        
                        # Check spatial overlap
                        if are_same_position(current_start, current_end, start, end, min_overlap_ratio):
                            df.loc[candidate_idx, 'morph'] = current_morph
                            to_process.append(candidate_idx)
        
        # Assign ninv to entire cluster
        for cluster_idx in cluster_indices: df.loc[cluster_idx, 'ninv'] = ninv_counter
        
        ninv_counter += 1
    
    return df


def add_ninv_morph_columns(inv_df: pd.DataFrame, lineage_dict: Dict[str, str], 
    config: PipelineConfig, lineage_file: bool) -> pd.DataFrame:
    """
    Add inversion numbering (ninv) and morphotype (A/B) columns to inversions.
    
    This function groups inversions into clusters based on:
    1. Same ID (reciprocal detection in genome pairs)
    2. Spatial overlap ≥ min_overlap_ratio
    
    Each cluster receives a unique 'ninv' identifier. Within a cluster, inversions
    are assigned morphotype 'A' or 'B' based on their orientation in genome pairs.
    
    Args:
        inv_df: DataFrame with inversion data containing columns:
                iso, chr, start, end, mapon, ID
        lineage_dict: Mapping from isolate name to lineage (e.g., {'iso1': 'Pd1'})
        outfile_name: Path to save the processed DataFrame
        config: Pipeline configuration object : 
            - min_overlap_ratio: Minimum overlap ratio (0-1) to consider inversions as same position
            - n_cores: Number of CPU cores for parallel processing. 
                If None, uses (total_cores - 1)
    
    Returns:
        DataFrame with added columns: type, ninv, morph
        - type: Lineage-based classification (e.g., 'intra_Pd1', 'inter_Pd')
        - ninv: Unique inversion cluster identifier (integer)
        - morph: Orientation morphotype ('A' or 'B')
        
    Algorithm:
        Uses parallel processing by chromosome when multiple chromosomes available.
        For each chromosome:
        1. Build spatial index with 50kb buckets for efficient lookup
        2. Use BFS to cluster inversions by ID and spatial overlap
        3. Assign morphotype based on genome pair orientation
    
    Performance:
        - Parallelizes by chromosome for datasets with many chromosomes
        - Uses spatial bucketing to avoid O(n²) comparisons
        - Falls back to sequential processing for small datasets
    """
    if inv_df.empty: return inv_df
    
    df = inv_df.copy()

    min_overlap_ratio = config.clustering.min_overlap_ratio
    n_cores = config.performance.n_cores
    bucket_size = config.clustering.spatial_bucket_size
    
    # Step 1: Vectorized lineage type assignment
    if lineage_file:
        iso_lineages = df['iso'].map(lineage_dict).values
        mapon_lineages = df['mapon'].map(lineage_dict).values
        df['type'] = [
            '_'.join(sorted([iso, mapon])) 
            for iso, mapon in zip(iso_lineages, mapon_lineages)
        ]
    else: df['type'] = 'unknown'

    # Initialize new columns
    df['ninv'] = -1
    df['morph'] = ''
    
    # Step 2: Determine parallelization strategy
    chromosomes = df['chr'].unique()
    
    if n_cores is None:
        n_cores = min(cpu_count() - 1, len(chromosomes))
        n_cores = max(1, n_cores)  # Ensure at least 1 core
    
    if len(chromosomes) >= n_cores:
        # Parallel processing by chromosome
        df_final = _process_inversions_parallel(df, chromosomes, min_overlap_ratio, bucket_size, n_cores)
    else:
        # Sequential processing
        df_final = _process_inversions_chunk((df, min_overlap_ratio, bucket_size))

    # Step 3: Finalize with type assignment
    result_df = df_final[['iso', 'chr', 'start', 'end', 'mapon', 'ID', 'type', 'ninv', 'morph']]
    if lineage_file:
        new_dfs = []
        for ninv_value in result_df['ninv'].unique():
            subset = result_df[result_df['ninv'] == ninv_value]
            subset = assign_inversion_type(subset)
            new_dfs.append(subset)
        result_df = pd.concat(new_dfs, ignore_index=True)
        
    else : result_df['type'] = 'unknown'
    result_df = result_df.sort_values(
        by=['ninv', 'iso', 'chr', 'start']
    ).reset_index(drop=True)
    
    
    return result_df


##########################################################################################################################
# Filtering inversions corresponding to transposable elements (TE)
##########################################################################################################################


def del_inv_isTE(df_inv: pd.DataFrame, df_te: pd.DataFrame, threshold: float, 
    outpath_te: Path, outpath_inv: Path) -> pd.DataFrame:
    """
    Identify and remove inversions that correspond exactly to transposable elements (TEs).
    
    An inversion is considered to match a TE if both overlap by at least `threshold`%
    of their respective lengths. This bidirectional criterion ensures the inversion
    and TE occupy essentially the same genomic space.
    
    Args:
        df_inv: DataFrame with inversion data containing columns:
                iso, chr, start, end, mapon, ninv
        df_te: DataFrame with TE annotations containing columns:
               iso, chr, start, end, ID
        threshold: Minimum overlap percentage (0-100) required in BOTH directions
                  e.g., 90 means both inversion and TE must be ≥90% covered
        outpath_te: Path to save TEs that match inversions
        outpath_inv: Path to save inversions that match TEs
    
    Returns:
        DataFrame of inversions WITHOUT TE matches (clean structural variants)
        
    Side effects:
        Writes two TSV files:
        - outpath_te: TEs that correspond to inversions
        - outpath_inv: Inversions that are actually TEs (filtered out)
    
    Note:
        This removes false positive inversions caused by TE mobility,
        which can create alignment artifacts resembling inversions.
    """
    inv_df = df_inv.copy()
    inv_df['length'] = abs(inv_df['end'] - inv_df['start'])
    
    # Sets to track matching elements
    matching_inv_ids = set()
    matching_te_ids = set()
    
    # Process by chromosome and isolate for efficiency
    for (chr_val, iso_val), inv_group in inv_df.groupby(['chr', 'iso']):
        # Get TEs for this chromosome/isolate combination
        matching_tes = df_te[
            (df_te['chr'] == chr_val) & 
            (df_te['iso'] == iso_val)
        ].copy()
        
        if len(matching_tes) == 0: continue
        
        matching_tes['te_length'] = matching_tes['end'] - matching_tes['start']
        
        # For each inversion, search for exact TE match
        for idx, inv in inv_group.iterrows():
            # Find candidate TEs that overlap
            candidate_tes = matching_tes[
                (matching_tes['start'] <= inv['end']) & 
                (matching_tes['end'] >= inv['start'])
            ]
            
            if len(candidate_tes) == 0: continue
            
            # Vectorized overlap calculation
            overlap_starts = np.maximum(inv['start'], candidate_tes['start'].values)
            overlap_ends = np.minimum(inv['end'], candidate_tes['end'].values)
            overlap_lengths = np.maximum(0, overlap_ends - overlap_starts)
            
            # Calculate overlap percentages for both inversion and TEs
            inv_overlap_pcts = (overlap_lengths / inv['length']) * 100
            te_overlap_pcts = (overlap_lengths / candidate_tes['te_length'].values) * 100
            
            # Exact match criterion: BOTH must be >= threshold%
            exact_match_mask = (inv_overlap_pcts >= threshold) & (te_overlap_pcts >= threshold)
            
            if exact_match_mask.any():
                # Take first exact match found
                exact_te = candidate_tes.loc[exact_match_mask].iloc[0]
                matching_inv_ids.add(inv['ninv'])
                matching_te_ids.add(exact_te['ID'])
                break  # One TE match per inversion is sufficient
    
    # Create output DataFrames
    # Inversions WITHOUT TE correspondence
    inv_no_isTE_df = inv_df[~inv_df['ninv'].isin(matching_inv_ids)].sort_values(
        by=['ninv', 'iso', 'chr', 'start'], ignore_index=True)
    
    # Inversions that ARE TEs (filtered out)
    inv_isTE_df = inv_df[inv_df['ninv'].isin(matching_inv_ids)].sort_values(
        by=['ninv', 'iso', 'chr', 'start'], ignore_index=True).drop(columns='length')

    # TEs that correspond to inversions
    TE_isTE_df = df_te[df_te['ID'].isin(matching_te_ids)].sort_values(
        by=['iso', 'chr', 'start'], ignore_index=True)
    
    # Save filtered data
    if config.output.save_TE_files:
        if not TE_isTE_df.empty:
            TE_isTE_df.to_csv(outpath_te, sep='\t', index=False)
        if not inv_isTE_df.empty:
            inv_isTE_df["ninv"] = pd.factorize(inv_isTE_df["ninv"])[0] + 1
            inv_isTE_df.to_csv(outpath_inv, sep='\t', index=False)
        else : print("No inversions correspond to TEs.")
        
    return inv_no_isTE_df


def del_inv_inTE(inversions_df: pd.DataFrame, df_te: pd.DataFrame, threshold: float, 
    outpath_te: Path, outpath_inv: Path) -> pd.DataFrame:
    """
    Filter inversions that are covered by ≥threshold% by a SINGLE transposable element.
    
    Unlike del_inv_isTE which requires bidirectional overlap, this function uses a
    unidirectional criterion: an inversion is filtered if ANY single TE covers at least
    threshold% of the inversion's length, regardless of how much of the TE is covered.
    
    This catches inversions that fall within larger TEs, which may represent
    alignment artifacts or genuine rearrangements within mobile elements.
    
    Args:
        inversions_df: DataFrame with inversion data containing columns:
                      iso, chr, start, end, mapon, ninv
        df_te: DataFrame with TE annotations containing columns:
               iso, chr, start, end, ID
        threshold: Minimum percentage (0-100) of inversion that must be covered
                  by a single TE. e.g., 90 means ≥90% of inversion covered
        outpath_te: Path to save TEs that cover inversions
        outpath_inv: Path to save inversions covered by TEs
    
    Returns:
        DataFrame of inversions NOT covered by TEs (clean structural variants)
        
    Side effects:
        Writes two TSV files:
        - outpath_te: TEs that cover inversions 
        - outpath_inv: Inversions covered by TEs (filtered out)
    
    Note:
        Uses binary search for efficient TE lookup by position.
        Only checks if individual TEs cover the inversion, not cumulative coverage.
    """
    # Index TEs by (iso, chr) for efficient lookup
    te_grouped = defaultdict(list)
    
    for _, te in df_te.iterrows():
        key = (te['iso'], te['chr'])
        # Store as tuple (start, end, ID) for fast access
        te_grouped[key].append((int(te['start']), int(te['end']), te['ID']))
    
    # Sort each TE list by start position for binary search
    for key in te_grouped: te_grouped[key].sort(key=lambda x: x[0])
    
    inversions_to_remove = set()
    tes_to_keep = set()
    
    # Check each inversion for TE coverage
    for _, inv in inversions_df.iterrows():
        key = (inv['iso'], inv['chr'])
        te_list = te_grouped.get(key, [])
        
        if not te_list: continue
        
        inv_start, inv_end = int(inv['start']), int(inv['end'])
        inv_length = inv_end - inv_start + 1
        min_coverage = (threshold / 100) * inv_length
        
        # Binary search to find first potentially overlapping TE
        left, right = 0, len(te_list) - 1
        start_idx = len(te_list)
        
        while left <= right:
            mid = (left + right) // 2
            te_start, te_end, te_id = te_list[mid]
            
            if te_end >= inv_start:  # TE ends after inversion starts
                start_idx = mid
                right = mid - 1
            else: left = mid + 1
        
        # Test each TE individually starting from start_idx
        found_covering_te = False
        
        for i in range(start_idx, len(te_list)):
            te_start, te_end, te_id = te_list[i]
            
            # Stop if TE starts after inversion ends
            if te_start > inv_end: break
            
            # Calculate overlap with this individual TE
            overlap_start = max(inv_start, te_start)
            overlap_end = min(inv_end, te_end)
            
            if overlap_start <= overlap_end:
                single_te_coverage = overlap_end - overlap_start + 1
                
                # If this single TE covers ≥threshold% of inversion
                if single_te_coverage >= min_coverage:
                    inversions_to_remove.add(inv['ninv'])
                    tes_to_keep.add(te_id)
                    found_covering_te = True
                    break  # No need to check other TEs
        
        # Early stopping if covering TE found
        if found_covering_te: continue
    
    # Create output DataFrames
    # Inversions NOT covered by TEs (keep these)
    inv_no_inTE_df = inversions_df[~inversions_df['ninv'].isin(inversions_to_remove)].sort_values(
        by=['ninv', 'iso', 'chr', 'start'], ignore_index=True)
    
    # Inversions covered by TEs (filtered out)
    inv_inTE_df = inversions_df[inversions_df['ninv'].isin(inversions_to_remove)].sort_values(
        by=['ninv', 'iso', 'chr', 'start'], ignore_index=True)
    
    # TEs involved in coverage
    TE_inTE_df = df_te[df_te['ID'].isin(tes_to_keep)].sort_values(
        by=['iso', 'chr', 'start'], ignore_index=True)
    
    # Save results
    if config.output.save_TE_files:
        if not TE_inTE_df.empty:
            TE_inTE_df.to_csv(outpath_te, sep='\t', index=False)
        if not inv_inTE_df.empty:
            inv_inTE_df["ninv"] = pd.factorize(inv_inTE_df["ninv"])[0] + 1
            inv_inTE_df.to_csv(outpath_inv, sep='\t', index=False)
        else : print("No inversions covered by TEs found.")
    
    return inv_no_inTE_df


def process_inversions_and_export_te(inv_df: pd.DataFrame, te_df: pd.DataFrame, threshold: float, 
    name_data: str, out_path_isTE: Path, out_path_inTE: Path, out_path_noTE: Path) -> None:
    """
    Process inversions with TE filtering pipeline and export all results.
    
    This function applies a two-step TE filtering process:
    1. Remove inversions that exactly match TEs (bidirectional overlap ≥threshold%)
    2. Remove inversions covered by TEs (unidirectional coverage ≥threshold%)
    
    Args:
        inv_df: DataFrame with inversion data containing columns:
                iso, chr, start, end, mapon, ninv
        te_df: DataFrame with TE annotations containing columns:
               iso, chr, start, end, ID
        threshold: Minimum overlap percentage (0-100) for TE filtering
        name_data: Dataset name used for intermediate file naming
        out_path_isTE: Path to save inversions that ARE TEs (exact matches)
        out_path_inTE: Path to save inversions covered by TEs
        out_path_noTE: Path to save final clean inversions (no TE association)
    
    Returns:
        None (writes results to files)
        
    Side effects:
        Creates up to 4 TSV files:
        - {name_data}_isTE.tsv: TEs matching inversions (step 1)
        - out_path_isTE: Inversions that are TEs (step 1)
        - {name_data}_inTE.tsv: TEs covering inversions (step 2)
        - out_path_inTE: Inversions covered by TEs (step 2)
        - out_path_noTE: Final clean inversions
    
    Pipeline:
        inv_df → del_inv_isTE → df_no_isTE → del_inv_inTE → df_noTE (final)
    """
    # Step 1: Filter inversions that ARE TEs (exact match)
    isTE_te_outpath = out_path_noTE.parent / f'TE_{name_data}_isTE.tsv'
    inv_no_isTE_df = del_inv_isTE(
        inv_df, te_df, threshold, isTE_te_outpath, out_path_isTE)
    
    # Step 2: Filter inversions covered by TEs (unidirectional coverage)
    inTE_te_outpath = out_path_noTE.parent / f'TE_{name_data}_inTE.tsv'
    inv_noTE_df = del_inv_inTE(
        inv_no_isTE_df, te_df, threshold, inTE_te_outpath, out_path_inTE)
    
    # Sort final clean inversions
    inv_noTE_df = inv_noTE_df.sort_values(by=['ninv', 'iso', 'chr', 'start'], ignore_index=True)
    
    # Save final results
    if not inv_noTE_df.empty: 
        inv_noTE_df["ninv"] = pd.factorize(inv_noTE_df["ninv"])[0] + 1
        inv_noTE_df.to_csv(out_path_noTE, sep='\t', index=False)
    else : print("No clean inversions remaining after TE filtering.")


##########################################################################################################################
# Loading TE and lineage data / Argument parsing / Main function 
##########################################################################################################################

def load_te_data(te_file: Path) -> pd.DataFrame:
    """
    Load and preprocess transposable element data.
    
    Args:
        te_file: Path to TE annotation TSV file
                Must contain columns: iso, chr, start, end, class
    
    Returns:
        Preprocessed DataFrame with columns: iso, chr, start, end, family, ID
        - iso: Isolate name (extracted as first two underscore-separated parts)
        - family: Renamed from 'class' column
        - ID: Row index
    """
    te_df = pd.read_csv(te_file, sep='\t')
    
    # Extract isolate names (keep first two parts: "isolate_version")
    te_df['iso'] = te_df['iso'].str.split('_').str[:2].str.join('_')
    
    # Remove unnecessary columns and rename
    if 'oid' in te_df.columns: te_df.drop(columns=['oid'], inplace=True)
    
    te_df = te_df.rename(columns={'class': 'family'})
    te_df['ID'] = te_df.index
    
    return te_df


def load_lineage_data(lineage_file: Path) -> Dict[str, str]:
    """
    Load lineage information for isolates.
    
    Args:
        lineage_file: Path to lineage TSV file
                     Must contain columns at positions 1 (iso) and 3 (lineage)
    
    Returns:
        Dictionary mapping isolate names to lineage identifiers
        e.g., {'iso_01': 'Pd1', 'iso_02': 'Pd2'}
    """
    lineage_df = pd.read_csv(lineage_file, sep='\t', usecols=[1, 3])
    lineage_df.columns = ['iso', 'lineage']
    lineage_df = lineage_df.drop_duplicates(subset='iso')
    
    return lineage_df.set_index('iso')['lineage'].to_dict()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process genomic inversions with TE filtering pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True, type=Path,
        help="Input TSV file with raw inversion data"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=Path.cwd(), type=Path,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=Path, default=None,
        help="Path to JSON configuration file. If not provided, uses default values."
    )
    
    parser.add_argument(
        "--te-file",
        type=Path, default=None,
        help="Path to TE annotation file (overrides config file)"
    )
    
    parser.add_argument(
        "--lineage-file",
        type=Path, default=None,
        help="Path to lineage file (overrides config file)"
    )
    
    parser.add_argument(
        "--create-default-config",
        type=Path,default=None,
        help="Create a default configuration file at specified path and exit"
    )
    
    return parser.parse_args()


def main(input_file: Path, output_dir: Path, config: PipelineConfig,
    te_df: Optional[pd.DataFrame] = None, lineage_dict: Optional[Dict[str, str]] = None) -> None:
    """
    Main pipeline for processing genomic inversions with TE filtering.
    
    This pipeline performs the following steps:
    1. Load and format raw inversion data
    2. Filter englobing (redundant) inversions
    3. Handle reciprocal inversions (merge T→Q and Q→T detections)
    4. Merge close inversions and filter by minimum size
    5. Assign inversion clusters (ninv) and morphotype (A/B)
    6. Filter TE-associated inversions
        
    Args:
            input_file: Path to input TSV file with raw inversion data
            output_dir: Directory to save output files
            config: Pipeline configuration object
            te_df: Optional DataFrame with transposable element annotations
                Required if config.te_filtering.enabled is True
            lineage_dict: Optional mapping from isolate name to lineage identifier
                        Required if config.lineage.enabled is True

    Returns:
        None (writes multiple TSV files)
        
    Output files:
        - INV_{filename}_nofilter.tsv: Initial formatted data (optional)
        - INV_{filename}_clean_wTE.tsv: Inversions before TE filtering (optional)
        - INV_{filename}_isTE.tsv: Inversions that ARE TEs (optional)
        - INV_{filename}_inTE.tsv: Inversions covered BY TEs (optional)
        - TE_{filename}_isTE.tsv: TEs matching inversions (exact) (optional)
        - TE_{filename}_inTE.tsv: TEs covering inversions (optional)
        - INV_{filename}_clean.tsv: Final clean inversions (no TE association)


    Raises:
        FileNotFoundError: If input_file does not exist
        ValueError: If required data (TE or lineage) is missing when enabled
        ValueError: If required columns are missing from input data
    """
    # Validate dependencies
    if config.te_filtering.enabled and te_df is None:
        raise ValueError("TE filtering is enabled but no TE data provided")
    
    if config.lineage.enabled and lineage_dict is None:
        raise ValueError("Lineage classification is enabled but no lineage data provided")
    
    filename = input_file.stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    out_path_nofilter = output_path / f"INV_{filename}_nofilter.tsv" if config.output.save_nofilter else None
    out_path_clean_wTE = output_path / f"INV_{filename}_clean_wTE.tsv" if config.output.save_wTE else None
    out_path_isTE = output_path / f"INV_{filename}_isTE.tsv" if config.output.save_TE_files else None
    out_path_inTE = output_path / f"INV_{filename}_inTE.tsv" if config.output.save_TE_files else None
    out_path_noTE = output_path / f"INV_{filename}_clean.tsv" 
        
    print(f"Input file: {input_file}\n")
    print("=" * 70)
    print("INVERSION PROCESSING PIPELINE")
    print("=" * 70)
    

    try:
        file_format = detect_alignment_format(input_file)
        print(f"\nDetected alignment format: {file_format}")
    except ValueError as e:
        print(e)
        sys.exit(1)

    current_step, n_step = 1,4

    if file_format == 'syri' : 
        # Step 1: Load and format data
        print(f"\n[Step {current_step}/6] Loading and formatting inversion data...")
        df_nofilter = format_df_myinv(str(input_file), config)
        if config.output.save_nofilter: 
            save_df_nofilter = final_format(df_nofilter)
            save_df_nofilter.to_csv(out_path_nofilter, sep='\t', index=False)
        print(f"  → Loaded {len(df_nofilter)} inversions")
        
        # Step 2: Filter englobing inversions
        print(f"\n[Step {current_step+1}/6] Filtering englobing inversions...")
        df_englob_filtered = filter_englobing_inversions(df_nofilter)
        removed_englob = len(df_nofilter) - len(df_englob_filtered)
        print(f"  → Removed {removed_englob} englobing inversions")
        print(f"  → Remaining: {len(df_englob_filtered)} inversions")

        
        # Step 3: Handle reciprocal inversions
        print(f"\n[Step {current_step+2}/6] Merging reciprocal inversions (T↔Q pairs)...")
        df_reciprocal = handle_duplicate_inversions(
            df_englob_filtered, config.merging.reciprocal_overlap_ratio)
        removed_reciprocal = len(df_englob_filtered) - len(df_reciprocal)
        print(f"  → Merged {removed_reciprocal} reciprocal duplicates")
        print(f"  → Remaining: {len(df_reciprocal)} inversions")
        
        # Convert to long format
        df_reciprocal_long = final_format(df_reciprocal)
        current_step, n_step = 3, 6

    elif file_format == 'maf': 
        print(f"\n[Step {current_step}/4] Loading and formatting inversion data...")
        df_detector = InversionDetector()
        df_reciprocal_long = df_detector.process_file(input_file)
        if config.output.save_nofilter: 
            df_reciprocal_long.to_csv(out_path_nofilter, sep="\t", index=False)
        print(f"  → Loaded {len(df_reciprocal_long)} inversions")


    # Step 4: Merge close inversions and filter by size
    print(f"\n[Step {current_step+1}/{n_step}] Merging close inversions (< {config.merging.fusion_gap_threshold}bp apart)...")
    df_merged = fusionner_inversions(df_reciprocal_long, config.merging.fusion_gap_threshold)
    df_size_filtered = filter_data_size(df_merged, config.filtering.min_inversion_size)
    removed_size = len(df_merged) - len(df_size_filtered)
    print(f"  → Removed {removed_size} inversions < {config.filtering.min_inversion_size}bp or without reciprocal evidence")
    print(f"  → Remaining: {len(df_size_filtered)} inversions")
    
    # Step 5: Assign ninv and morphotype (requires lineage data)
    if config.lineage.enabled:
        print(f"\n[Step {current_step+2}/{n_step}] Assigning inversion clusters (ninv) and morphotype...")
        df_clustered = add_ninv_morph_columns(
            df_size_filtered, lineage_dict, config, lineage_file=True)

    else: 
        print(f"\n[Step {current_step+2}/{n_step}]  Assigning inversion clusters (ninv) and Skipping morphotype (lineage data disabled)")
        df_clustered = add_ninv_morph_columns(
            df_size_filtered, lineage_dict, config, lineage_file=False)
        n_clusters = df_clustered['ninv'].nunique()
        print(f"  → Identified {n_clusters} unique inversion clusters")

    if config.output.save_wTE: 
        df_clustered["ninv"] = pd.factorize(df_clustered["ninv"])[0] + 1
        df_clustered.to_csv(out_path_clean_wTE, sep='\t', index=False)
    # Step 6: TE filtering (optional)
    if config.te_filtering.enabled:
        print(f"\n[Step {current_step+3}/{n_step}] Filtering TE-associated inversions...")
        print(f"  → TE overlap threshold: {config.te_filtering.overlap_threshold}%")
        initial_count = len(df_clustered)
        
        process_inversions_and_export_te(
            df_clustered, te_df, 
            config.te_filtering.overlap_threshold, filename, 
            out_path_isTE, out_path_inTE, out_path_noTE
        )
        
        if out_path_noTE.exists():
            df_final = pd.read_csv(out_path_noTE, sep='\t')
            removed_te = initial_count - len(df_final)
            final_clusters = df_final['ninv'].nunique() if 'ninv' in df_final.columns else 'N/A'
            
            print(f"  → Removed {removed_te} TE-associated inversions")
            print(f"  → Final: {len(df_final)} inversions in {final_clusters} clusters")
    else:
        print("\n[Step 6/6] Skipping TE filtering (disabled in config)")
        df_final = df_clustered
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)
    print(f"\nOutput directory: {output_path}\n")


# Pipeline : 

if __name__ == "__main__":
    args = parse_arguments()
    
    # Handle config creation request
    if args.create_default_config:
        PipelineConfig.create_default_config(args.create_default_config)
        exit(0)
    
    # Load or create configuration
    if args.config:
        print(f"\nLoading configuration from: {args.config}")
        config = PipelineConfig.from_json(args.config)
    else:
        print("No configuration file provided, using default values")
        config = PipelineConfig()
    
    # Override config with command-line arguments
    if args.te_file: config.te_filtering.te_file = str(args.te_file)
    if args.lineage_file: config.lineage.lineage_file = str(args.lineage_file)
    
    # Validate configuration
    try: config.validate()
    except (ValueError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        exit(1)
    
    # Load optional data based on configuration
    te_df = None
    lineage_dict = None
    
    if config.te_filtering.enabled:
        if config.te_filtering.te_file:
            print(f"Loading TE data from: {config.te_filtering.te_file}")
            te_df = load_te_data(Path(config.te_filtering.te_file))
            print(f"  → Loaded {len(te_df)} TE annotations")
        else:
            print("WARNING: TE filtering enabled but no TE file provided. Disabling TE filtering.")
            config.te_filtering.enabled = False
    
    if config.lineage.enabled:
        if config.lineage.lineage_file:
            print(f"Loading lineage data from: {config.lineage.lineage_file}")
            lineage_dict = load_lineage_data(Path(config.lineage.lineage_file))
            print(f"  → Loaded {len(lineage_dict)} lineage mappings")
        else:
            print("WARNING: Lineage classification enabled but no lineage file provided. Disabling lineage.")
            config.lineage.enabled = False
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    
    print()
    
    # Run pipeline
    try:
        main(
            input_file=args.input, output_dir=args.output, config=config,
            te_df=te_df, lineage_dict=lineage_dict)
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)