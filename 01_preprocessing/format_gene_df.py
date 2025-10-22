#!/usr/bin/env python3
"""
=============================================================================
Funannotate GFF Parser - Gene Annotation Data Processor
=============================================================================

Description : Python script for processing and combining Funannotate .gff
              files into unified TSV datasets for comprehensive gene
              annotation analysis and comparative genomics studies

Features :
  • Batch processing of multiple Funannotate .gff annotation files
  • Gene structure parsing with exon/intron boundary detection
  • Automatic intron generation from exon coordinates
  • Product annotation extraction and mapping
  • Chromosome name standardization and processing
  • CDS filtering and redundancy removal for clean datasets

Analysis Workflow :
  1. Directory scanning for Funannotate .gff files
  2. Individual file parsing with comment line filtering
  3. Gene structure decomposition (gene, exon, RNA features)
  4. Product annotation extraction from RNA features
  5. Automated intron generation between exons
  6. Data standardization and coordinate sorting

Usage : python3 funannotate_parser.py -i <input_directory> -o <output.tsv> [options]
        python3 funannotate_parser.py --help for more information

Required arguments :
  -i, --input      : Directory path containing Funannotate .gff files
  -o, --output     : Output TSV file path for processed annotations

Optional arguments :
  -h, --help       : Display help message

Output structure :
  [output_file].tsv    # Complete gene annotation dataset
                       # Columns: iso, chr, start, end, type, oid, product

Data Processing Features :
  • Gene structure parsing: gene → exon → intron relationships
  • Product mapping: RNA feature products assigned to gene elements
  • Coordinate validation: start/end position verification
  • Type filtering: CDS removal (redundant with exon data)
  • Chromosome renaming: custom naming scheme application

Chromosome Processing :
  • Standard chromosomes: chr[N] → chr[N+1000]
  • Special chromosomes: chr[3,16,17] → chr[N+2000]
  • Non-standard names: preserved as-is

Intron Generation Algorithm :
  • Gene boundary analysis: full gene span determination
  • Exon gap detection: spaces between consecutive exons
  • 5'/3' UTR regions: non-exonic gene regions included
  • Coordinate validation: proper start/end relationships

Supported Feature Types :
  • gene: primary gene features with coordinates
  • exon: coding sequence boundaries  
  • intron: generated from exon gaps (automatic)
  • [type]RNA: various RNA types with product annotations

Error Handling :
  • Missing file detection and reporting
  • Malformed GFF validation and skipping
  • Gene structure consistency checks
  • Product annotation error recovery

Author    : Hugo Bellavoir
Email     : bellavoirh@gmail.com
Version   : 1.0
License   : MIT

Dependencies : python3, pandas, pathlib
Tested on    : Linux/Unix systems

=============================================================================

Script for processing Funannotate .gff files and combining them into a single TSV file.
This script compiles annotation data, filters it, and formats it for analysis.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import warnings
# Suppress pandas parser warnings
warnings.simplefilter(action='ignore', category=pd.errors.ParserWarning)

def process_chromosome_name(chrom_name):
    """
    Processes chromosome names according to a specific rule.
    Args:
        chrom_name (str): Chromosome name to process (e.g., 'chr3')
    Returns:
        str: Transformed chromosome name
    """
    try:
        chrom_num = int(chrom_name.replace('chr', ''))
        if chrom_num in [3, 16, 17]: return f'chr{chrom_num + 2000}'
        else: return f'chr{chrom_num + 1000}'
    except ValueError: return chrom_name


def create_df_all_gene(directory_path, outfile_path):
    """
    Creates a unified DataFrame from all .gff files in a directory.
    Args:
        directory_path (str): Path to directory containing .gff files
        outfile_path (str): Path for output TSV file
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    all_dfs = []
    file_count = 0
    
    # Check that directory exists
    if not os.path.isdir(directory_path):
        print(f"ERROR: Directory '{directory_path}' does not exist", file=sys.stderr)
        return False
    
    # Create output directory if it doesn't exist
    outfile_dir = os.path.dirname(outfile_path)
    if outfile_dir and not os.path.exists(outfile_dir):
        try:
            os.makedirs(outfile_dir)
            print(f"Output directory created: {outfile_dir}")
        except OSError as e:
            print(f"Error creating output directory: {e}", file=sys.stderr)
            return False
    
    # Scan all .gff files in the directory
    out_files = list(Path(directory_path).glob('*.gff*'))
    if not out_files:
        print(f"Warning: No .gff files found in '{directory_path}'", file=sys.stderr)
        return False
    
    print(f"Processing {len(out_files)} .gff files...")
    
    for file_path in out_files:
        try:
            print(f"Processing file: {file_path.name}")   
            df = pd.read_csv(file_path, sep=r'\t',comment='##', header=None)
            
            # Check that file contains data
            if df.empty:
                print(f"Warning: File {file_path.name} is empty or malformed", file=sys.stderr)
                continue
            
            # Assign original column names
            df.columns = ['chr', 'tools', 'type', 'start', 'end', 'score', 'strand', 'phase', 'atr']
            df = df[df['type'] != 'CDS']  # remove CDS lines as they == exon
            # Reformat DataFrame
            df['iso'] = file_path.stem 
            df['chr'] = df['chr'].apply(process_chromosome_name)
            df = df[['iso', 'chr', 'start', 'end', 'type', 'atr']]

            # 1. Extract `oid` from attribute
            df['oid'] = df['atr'].str.extract(r'ID=([^;]+)')[0].str.split('-').str[0]

            # 2. Extract `product` only for RNA
            df['product'] = '.' 
            mask = df['type'].str.endswith('RNA', na=False)
            df.loc[mask, 'product'] = df.loc[mask, 'atr'].str.extract(r'product=([^;]+)')[0]
            df.drop(columns=['atr'], inplace=True)

            # 3. Retrieve products at each oid level (if RNA present)
            product_map = df[mask][['oid', 'product']].dropna().drop_duplicates()
            df = df[~mask] # remove RNA lines as they == gene
            df = df.merge(product_map, on='oid', how='left', suffixes=('', '_from_rna'))
            df['product'] = df['product_from_rna'].combine_first(df['product'])
            df.drop(columns=['product_from_rna'], inplace=True)

            # 4. Generate introns by isolating regions not covered by exons
            def generate_introns(subdf):
                gene = subdf[subdf['type'] == 'gene']     
                if len(gene) != 1:
                    print(f"ALERT: {len(gene)} genes found for oid {subdf['oid'].iloc[0]}")
                    return pd.DataFrame()
                gene = gene.iloc[0]

                exons = subdf[subdf['type'] == 'exon'].sort_values(by='start')
                introns = []
                if len(exons) == 0: return pd.DataFrame()
                
                # Check if there's space between gene start and first exon
                first_exon = exons.iloc[0]
                if gene['start'] < first_exon['start']:
                    introns.append({'iso': gene['iso'], 'chr': gene['chr'],
                        'start': gene['start'], 'end': first_exon['start'] - 1,
                        'type': 'intron', 'oid': gene['oid'], 'product': gene['product']})
                
                # Create introns between consecutive exons
                for i in range(1, len(exons)):
                    prev_exon = exons.iloc[i-1]
                    curr_exon = exons.iloc[i]
                    
                    intron_start = prev_exon['end'] + 1
                    intron_end = curr_exon['start'] - 1
                    
                    if intron_start <= intron_end:
                        introns.append({'iso': gene['iso'], 'chr': gene['chr'],
                            'start': intron_start, 'end': intron_end,
                            'type': 'intron', 'oid': gene['oid'], 'product': gene['product']})
                
                # Check if there's space between last exon and gene end
                last_exon = exons.iloc[-1]
                if last_exon['end'] < gene['end']:
                    introns.append({'iso': gene['iso'], 'chr': gene['chr'],
                        'start': last_exon['end'] + 1, 'end': gene['end'],
                        'type': 'intron', 'oid': gene['oid'], 'product': gene['product']})
                
                return pd.DataFrame(introns)

            # Add introns
            intron_dfs = df.groupby('oid')[df.columns].apply(generate_introns).reset_index(drop=True)
            if not intron_dfs.empty: df = pd.concat([df, intron_dfs], ignore_index=True)
            
            df.reset_index(drop=True, inplace=True)
            all_dfs.append(df)
            file_count += 1
            
        except Exception as e:
            print(f"Error processing file {file_path.name}: {e}", file=sys.stderr)
            continue
    
    # Check if files were processed successfully
    if not all_dfs:
        print("Error: No files could be processed correctly", file=sys.stderr)
        return False
        
    try:
        # Merge all DataFrames
        df = pd.concat(all_dfs, ignore_index=True)
        # Export filtered DataFrame to TSV file
        df = df.sort_values(by=['iso', 'chr', 'start'])
        df.to_csv(outfile_path, index=False, sep='\t')
        
        print(f"Processing completed: {file_count} files processed")
        print(f"Total number of genes: {len(df['oid'].unique())}")
        print(f"Results saved in: {outfile_path}")
            
        return True
        
    except Exception as e:
        print(f"Error during data merging or export: {e}", file=sys.stderr)
        return False


def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="Process Funannotate .gff files and combine them into a single TSV file.",
        epilog="Example: %(prog)s -i /path/to/files -o result.tsv"
    )
    parser.add_argument('-i', '--input', required=True, help="Path to directory containing .gff files")
    parser.add_argument('-o', '--output', required=True, help="Path for output TSV file")

    # Parse arguments
    args = parser.parse_args()
    # Execute main function
    success = create_df_all_gene(args.input, args.output)
    # Set exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()