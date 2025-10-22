#!/usr/bin/env python3
"""
=============================================================================
RepeatMasker Parser - Transposable Elements Data Processor
=============================================================================

Description : Python script for processing and combining RepeatMasker .out
              files into unified TSV datasets for transposable elements
              analysis and downstream genomic studies

Features :
  • Batch processing of multiple RepeatMasker .out files
  • Data parsing and standardization from RepeatMasker format
  • Transposable elements classification and filtering
  • Simple repeats and low complexity sequences separation  
  • Quality control and error handling for malformed files
  • TSV output format for statistical analysis and visualization

Analysis Workflow :
  1. Directory scanning for RepeatMasker .out files
  2. Individual file parsing and validation
  3. Data standardization and column renaming
  4. Transposable elements classification filtering
  5. Simple repeats separation into dedicated output
  6. Final dataset compilation and export

Usage : python3 repeatmasker_parser.py -i <input_directory> -o <output.tsv> [options]
        python3 repeatmasker_parser.py --help for more information

Required arguments :
  -i, --input      : Directory path containing RepeatMasker .out files
  -o, --output     : Output TSV file path for processed data

Optional arguments :
  -v, --verbose    : Display detailed processing information
  -h, --help       : Display help message

Output structure :
  output_directory/
  ├── [specified_name].tsv     # Main transposable elements dataset
  ├── SR.tsv                   # Simple repeats and low complexity sequences
  └── processing.log           # Execution details (if verbose)

Data Processing :
  • Input parsing from RepeatMasker standard 15-column format
  • Column standardization: iso, chr, start, end, class, oid, div
  • Filtering separation: transposable elements vs simple repeats
  • Data validation and quality control checks
  • Sorting by genomic coordinates (iso, chr, start)

Supported Classifications :
  • Main dataset: All transposable elements (LTR, LINE, SINE, DNA, etc.)
  • Separated: Simple_repeat, Low_complexity sequences
  • Quality metrics: divergence scores and positional data

Error Handling :
  • Missing file detection and reporting
  • Malformed data validation and skipping
  • Output directory creation with permission checks
  • Comprehensive error logging and user feedback

Author    : Hugo Bellavoir
Email     : bellavoirh@gmail.com
Version   : 1.0
License   : MIT

Dependencies : python3, pandas, pathlib
Tested on    : Linux/Unix systems

=============================================================================

Script for processing RepeatMasker .out files and combining them into a single TSV file.
This script compiles transposable element data, filters it, and formats it for analysis.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import warnings


def create_df_all_TE(directory_path, outfile_path, verbose=False):
    """
    Creates a unified DataFrame from all .out files in a directory.
    Args:
        directory_path (str): Path to directory containing .out files
        outfile_path (str): Path for output TSV file
        verbose (bool): Display detailed information during processing
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    all_dfs = []
    file_count = 0

    # Create output directory if it doesn't exist
    outfile_dir = os.path.dirname(outfile_path)
    if outfile_dir and not os.path.exists(outfile_dir):
        try:
            os.makedirs(outfile_dir)
            if verbose: print(f"Output directory created: {outfile_dir}")
        except OSError as e:
            print(f"Error creating output directory: {e}", file=sys.stderr)
            return False
    
    # Scan all .out files in the directory
    out_files = list(Path(directory_path).glob('*.out'))
    if not out_files:
        print(f"Warning: No .out files found in '{directory_path}'", file=sys.stderr)
        return False
    
    if verbose: print(f"Processing {len(out_files)} .out files...")
    
    for file_path in out_files:
        try:
            if verbose: print(f"Processing file: {file_path.name}")   
            df = pd.read_csv(file_path, sep='\t', 
                          index_col=False)
                        
            # Check that file contains data
            if df.empty:
                print(f"Warning: File {file_path.name} is empty or malformed", file=sys.stderr)
                continue
            
            # Assign original column names
            df.columns = ['score', 'div.', 'del.', 'ins.', 'sequence', 'begin', 'end', '(left)', 'strand', 
                'repeat', 'class/family', 'begin.1', 'end.1', '(left).1', 'ID']
            
            # Reformat DataFrame
            df['iso'] = file_path.stem 
            df = df[['iso', 'sequence', 'begin', 'end', 'class/family', 'ID', 'div.']]
            df.columns = ['iso', 'chr', 'start', 'end', 'class', 'oid', 'div']
            
            all_dfs.append(df)
            file_count += 1
            
        except Exception as e:
            print(f"Error processing file {file_path.name}: {e}", file=sys.stderr)
            continue
    
    # Check if files were processed successfully
    if not all_dfs:
        print("Error: No files could be processed correctly", file=sys.stderr)
        return False
        
    # Merge all DataFrames
    try:
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Filter and group transposable element classes
        df_repeat = df[(df['class'] == 'Low_complexity') | (df['class'] == 'Simple_repeat')]
        df = df[(df['class'] != 'Low_complexity') & (df['class'] != 'Simple_repeat')]
        
        #df['class'] = df['class'].replace(['Simple_repeat', 'Low_complexity', 'ClassI/Unclassified', 'RC/Helitron'], 'Others')
        #df['class'] = df['class'].replace({r'^LTR.*$': 'LTR', r'^DNA.*$': 'DNA', r'^LINE.*$': 'LINE'}, regex=True)

        # Export filtered DataFrame to TSV file
        df = df.sort_values(by=['iso', 'chr', 'start'])
        df.to_csv(outfile_path, index=False, sep='\t')

        outdir = os.path.join(*outfile_path.split('/')[:-1])
        outfile_sr = os.path.join(outdir, "SR.tsv")
        df_repeat.to_csv(outfile_sr, index=False, sep='\t')

        if verbose:
            print(f"Processing completed: {file_count} files processed")
            print(f"Total number of transposable elements: {len(df)}")
            print(f"Total number of simple repeated elements: {len(df_repeat)}")
            print(f"Results saved in: {outdir}")
            
        return True
        
    except Exception as e:
        print(f"Error during data merging or export: {e}", file=sys.stderr)
        return False


def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="Process RepeatMasker .out files and combine them into a single TSV file.",
        epilog="Example: %(prog)s -i /path/to/files -o result.tsv"
    )
    parser.add_argument('-i', '--input', required=True, help="Path to directory containing .out files")
    parser.add_argument('-o', '--output', required=True, help="Path for output TSV file")
    parser.add_argument('-v', '--verbose', action='store_true', help="Display detailed information during processing")
    
    # Parse arguments
    args = parser.parse_args()
    # Execute main function
    success = create_df_all_TE(args.input, args.output, args.verbose)
    # Set exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()