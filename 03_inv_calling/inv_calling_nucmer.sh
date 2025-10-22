#!/bin/bash
#
# =============================================================================
# Genome Inversion Calling Pipeline - Nucmer & SyRI based Analysis
# =============================================================================
#
# Description :
#   Bash script for detecting genomic inversions across multiple FASTA genomes
#   using whole-genome alignments (Nucmer) and structural variant calling (SyRI).
#   Designed for comparative genomics, structural variation discovery,
#   and downstream evolutionary or functional analyses.
#
# Features :
#   • Automated pairwise comparisons of all FASTA genomes in a directory
#   • Chromosome-wise alignment and inversion detection
#   • Parallelization with customizable thread usage
#   • Integrated logging system with progress and error reporting
#   • Output filtering and formatting into clean TSV datasets
#
# Analysis Workflow :
#   1. Input directory scanning for FASTA genome files
#   2. Pairwise comparison of all genome combinations
#   3. Chromosome extraction and alignment with Nucmer
#   4. Filtering alignments using delta-filter and show-coords
#   5. Structural variant calling with SyRI
#   6. Extraction and formatting of inversion coordinates
#   7. Aggregation of results into a single TSV file
#
# Usage :
#   ./inv_calling.sh -d <input_directory> -o <output.tsv> [options]
#   ./inv_calling.sh --help   # for more information
#
# Required arguments :
#   -d, --directory <folder>   Directory containing FASTA genome files
#
# Optional arguments :
#   -o, --output <path>        Path to the output TSV file
#                              (default: ./inv_calling.tsv)
#   -t, --threads <int>        Number of threads to use (default: 8)
#   -h, --help                 Display help message
#
# Output structure :
#   output_directory/
#   ├── inv_calling.tsv        # Detected inversions (main dataset)
#   ├── inv_calling_nucmer.log # Execution logs
#
# Data Processing :
#   • Input validation (directories, files, prerequisites)
#   • Sequence extraction for common homologous chromosomes
#   • Nucmer alignment with filtering for reliable matches
#   • SyRI-based structural variant detection
#   • AWK-based parsing to isolate inversions and avoid duplicates
#   • Sorted and standardized TSV output
#
# Supported Analyses :
#   • Genome-wide inversion detection
#   • Chromosome-level inversion mapping
#   • Multi-genome comparative genomics
#
# Error Handling :
#   • Prerequisite tools check (samtools, awk, nucmer, delta-filter, show-coords, syri)
#   • Input directory and file validation
#   • Empty or missing sequence extraction checks
#   • Logging of warnings and errors with timestamps
#
# Author    : Hugo Bellavoir
# Email     : bellavoirh@gmail.com
# Version   : 1.0
# License   : MIT
#
# Dependencies : bash, samtools, awk, nucmer, delta-filter, show-coords, syri
# Tested on    : Linux/Unix systems
#
# =============================================================================
#
# Script for automated detection of genomic inversions from multiple FASTA
# genomes using Nucmer alignments and SyRI structural variant calling.
#
# =============================================================================


RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m';
BLUE='\033[0;34m'; NC='\033[0m'

help() {
    echo -e "\n${BLUE}Usage:${NC} $0 -d <in_directory> -o <out_directory/file.tsv> [options]\n"
    echo -e "${BLUE}Required options:${NC}"
    echo "  -d, --directory <folder>   Directory containing FASTA files to analyze"
    echo -e "${BLUE}Optional options:${NC}"
    echo "  -h, --help                  Display this help message"
    echo "  -o, --output <path_file>    Path to the output file (default: ./inv_calling_nucmer.tsv)"
    echo "  -t, --threads <int>         Number of threads to use (default: 8)"    
    echo -e "\n${YELLOW}IMPORTANT: ${NC}Homologous chromosomes must have the same identifier and be on the same strand.\n"
    exit 0
}

check_prerequisites() {
    local missing_tools=()
    for tool in samtools awk nucmer delta-filter show-coords syri; do
        command -v "$tool" &>/dev/null || missing_tools+=("$tool")
    done

    if ((${#missing_tools[@]})); then
        echo -e "\n${RED}Error: The following tools are not installed or not in PATH:${NC}"
        printf "${RED}  - %s\n${NC}" "${missing_tools[@]}"
        echo -e "Please install these tools before running this script.\n"
        exit 1
    fi
}

check_directory() {
    local directory="$1"
    local description="$2"
    # Check if directory exists and is readable
    if [ ! -d "$directory" ] || [ ! -r "$directory" ]; then
        echo -e "${RED}Error: The $description directory '$directory' does not exist or is not readable.${NC}"
        log_entry "ERROR: Directory '$directory' not found or not readable"
        exit 1
    fi
}

create_directory() {
    local directory="$1"
    # If directory doesn't exist, try to create it
    if [ ! -d "$directory" ]; then
        log_entry "Creating directory '$directory'"
        mkdir -p "$directory" || { 
            echo -e "${RED}Error: Unable to create directory '$directory'.${NC}"
            log_entry "ERROR: Creation of directory '$directory' failed"
            exit 1
        }
    fi
    # Check if directory is writable
    if [ ! -w "$directory" ]; then
        echo -e "${RED}Error: Directory '$directory' is not writable.${NC}"
        log_entry "ERROR: Directory '$directory' not writable"
        exit 1
    fi
}

log_entry() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"; }

execute_command() {
    local description="$1"           
    local command="$2"              
    local ignore_error="${3:-false}" # Flag indicating if errors should be ignored (default "false")
    
    log_entry "EXECUTION: $description" 
    
    OUTPUT=$(eval "$command" 2>&1) # Execute command in subshell and capture output
    local STATUS=$?  # Get command return code
    
    if [ $STATUS -eq 0 ] || [ "$ignore_error" == "true" ]; then
        return 0  # if command executed successfully or if errors are ignored
    else
        # If an error occurred, display error message and log it
        echo -e "  ${RED}✗${NC} Error during $description"
        log_entry "ERROR: $description: $OUTPUT"
        return 1
    fi
}

# Initialize default variables
OUTFILE="./inv_calling.tsv"
THREADS=8
INDIR=""

# Process options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            help
            ;;
        -d|--directory)
            INDIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTFILE="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            if ! [[ "$THREADS" =~ ^[0-9]+$ ]]; then
                echo -e "${RED}Error: The number of threads must be a positive integer.${NC}"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            echo "Use '$0 --help' to see available options."
            exit 1
            ;;
    esac
done

# =========== Parameter validation and output environment preparation ========
if [ -z "$INDIR" ]; then
    echo -e "${RED}Error: The -d/--directory option is mandatory.${NC}"
    echo "Use '$0 --help' for more information."
    exit 1
fi

if [[ "$OUTFILE" != *.tsv ]]; then
    echo -e "${RED}Error: The path $OUTFILE is not a valid .tsv file path.${NC}"
    exit 1 
fi

check_prerequisites

OUTDIR=$(dirname "$OUTFILE")
if [ "$OUTDIR" = "." ]; then
    OUTDIR=$(pwd)
fi
OUTDIR=$(realpath "$OUTDIR")

LOG_FILE="$OUTDIR/inv_calling_nucmer.log"
[ -f "$LOG_FILE" ] && rm "$LOG_FILE"

check_directory "$INDIR" "input"
INDIR=$(realpath "$INDIR")

create_directory "$OUTDIR"

OUTFILE=$(realpath "$OUTFILE")
[ -f "$OUTFILE" ] && rm "$OUTFILE"

TMP_DIR="$OUTDIR/tmp_dir_nucmer_output"
create_directory "$TMP_DIR"
[ "$(ls -A "$TMP_DIR" 2>/dev/null)" ] && rm "$TMP_DIR"/*
# ======================================================================================



# ===== Retrieve and check the list of FASTA files in the directory =====
fasta_files=($(find "$INDIR" -maxdepth 1 -type f -iname "*.fasta"))
if [ ${#fasta_files[@]} -eq 0 ]; then
    echo -e "${RED}Error: No .fasta files found in directory $INDIR.${NC}"
    log_entry "ERROR: No .fasta files found in directory $INDIR"
    exit 1
fi
# ======================================================================================

# Display parameter summary
echo -e "${BLUE}====== Parameters ======${NC}"
echo -e "Input directory:            ${GREEN}$INDIR${NC}"
echo -e "Output file:                ${GREEN}$OUTFILE${NC}"
echo -e "Threads:                    ${GREEN}$THREADS${NC}"
echo -e "FASTA files:                ${GREEN}${#fasta_files[@]}${NC}\n"

# Calculate total number of comparisons
total_comparisons=$(( ${#fasta_files[@]} * (${#fasta_files[@]} - 1) ))
current_comparison=0

echo -e "target\tt_chr\tt_start\tt_end\tquery\tq_chr\tq_start\tq_end" >> $OUTFILE

# Main loop to compare all genome pairs
for ((i=0; i<${#fasta_files[@]}; i++)); do
    for ((j=0; j<${#fasta_files[@]}; j++)); do
        if [[ $i -ne $j ]]; then 
            current_comparison=$((current_comparison + 1))
            target_file="${fasta_files[i]}"
            query_file="${fasta_files[j]}"
            target_name=$(basename "${target_file}" | cut -d. -f1)
            query_name=$(basename "${query_file}" | cut -d. -f1)
            
            # Calculate and display progress percentage
            percent=$((current_comparison * 100 / total_comparisons))
            echo -e "\n${BLUE}[$percent%] Analyzing:${NC} ${GREEN}$target_name${NC} - ${GREEN}$query_name${NC}"
            log_entry "\n\t======= Analysis of pair: $target_name - $query_name ($percent%) ======="
            
            # Extract chromosomes for each file and keep common ones
            chrs_query=$(grep -o '^>[^ ]*' "$query_file" | sed 's/^>//' | sort | uniq)
            chrs_target=$(grep -o '^>[^ ]*' "$target_file" | sed 's/^>//' | sort | uniq)
            common_chrs=$(comm -12 <(echo "$chrs_target") <(echo "$chrs_query"))
            
            # Check if there are common chromosomes
            if [ -z "$common_chrs" ]; then
                echo -e "  ${YELLOW}⚠ No common chromosomes found${NC}"
                log_entry "WARNING: No common chromosomes between $target_name and $query_name"
                continue
            fi

            # Display progress for chromosomes
            num_chrs=$(echo "$common_chrs" | wc -l)
            chr_idx=0            
            # Loop through common chromosomes
            for chr in $common_chrs; do
                chr_idx=$((chr_idx + 1))
                echo -ne "\r\033[K[${chr_idx}/${num_chrs}] Processing ${GREEN}$chr${NC}..."
                log_entry " --> Processing chromosome: $chr ($target_name - $query_name)"
                
                # Temporary file names
                chr_target_file="$TMP_DIR/target.fasta"
                chr_query_file="$TMP_DIR/query.fasta"
                output_bam="$TMP_DIR/algnt.bam"
                syri_out="$TMP_DIR/syri.out"

                # ==================== Extract sequences from common chromosome ====================
                AWK_EXTRACT="awk -v chr=\">$chr\" '
                    \$0 ~ chr\"([ \\t]|$)\" {print_flag=1} 
                    \$0 ~ \"^>\" && \$0 !~ chr\"([ \\t]|$)\" {print_flag=0} 
                    print_flag
                ' \"$target_file\" > \"$chr_target_file\""
                if ! execute_command "Extracting chromosome $chr from $target_name" "$AWK_EXTRACT"; then
                    echo -e "${YELLOW}✗${NC}"
                    continue
                fi
                
                AWK_EXTRACT="awk -v chr=\">$chr\" '
                    \$0 ~ chr\"([ \\t]|$)\" {print_flag=1} 
                    \$0 ~ \"^>\" && \$0 !~ chr\"([ \\t]|$)\" {print_flag=0} 
                    print_flag
                ' \"$query_file\" > \"$chr_query_file\""
                if ! execute_command "Extracting chromosome $chr from $query_name" "$AWK_EXTRACT"; then
                    echo -e "${YELLOW}✗${NC}"
                    continue
                fi

                # Check that extracted files are not empty
                if [ ! -s "$chr_target_file" ] || [ ! -s "$chr_query_file" ]; then
                    echo -e "${YELLOW}✗${NC} (empty file)"
                    log_entry "WARNING: Empty extraction file for chromosome $chr"
                    continue
                fi
                # =====================================================================================


                # ========= Run Nucmer to map chromosomes and filter results =============
                # -c 1500 : minimum cluster length  
                # -b 200 : break length for clusters  
                # -l 500 : minimum match length  
                # Then filters alignments (-m, -i 75%, -l 500) and generates coordinate file.
                #   -m : keep only maximal alignments  
                #   -i 75 : minimum 75% identity  
                #   -l 500 : minimum alignment length 500 bp  
                # Finally, show-coords generates a tab-delimited coordinate file.
                NUCMER_CMD="nucmer -c 1500 -t \"$THREADS\" -b 200 -l 500 \"$chr_target_file\" \"$chr_query_file\" -p \"$TMP_DIR/out\" && \
                            delta-filter -m -i 75 -l 500 \"$TMP_DIR/out.delta\" > \"$TMP_DIR/out.filtered.delta\" && \
                            show-coords -THrd \"$TMP_DIR/out.filtered.delta\" > \"$TMP_DIR/out.filtered.coords\""

                if ! execute_command "Aligning $chr with Nucmer" "$NUCMER_CMD"; then
                    echo -e "${YELLOW}✗${NC}"
                    continue
                fi
                # =====================================================================================


                # ========== Run syri for structural variant detection =================
                SYRI_CMD="syri -c "$TMP_DIR/out.filtered.coords" -d "$TMP_DIR/out.filtered.delta" -r "$chr_target_file" -q "$chr_query_file" --dir "$TMP_DIR" --nc $THREADS > /dev/null 2>&1"
                
                if ! execute_command "Detecting structural variants with SyRI" "$SYRI_CMD" true; then
                    echo -e "${YELLOW}✗${NC}"
                    log_entry "Error for $target_name - $query_name : $chr"
                    continue
                fi
                # Check if SyRI output file exists
                if [ ! -f "$syri_out" ]; then
                    echo -e "${YELLOW}✗${NC} (missing file)"
                    log_entry "Missing SyRI output file for $target_name - $query_name : $chr"
                    continue
                fi
                # =====================================================================================


                # ==================== Extraction and formatting of inversions ===========================
                AWK_EXTRACT="awk -F'\\t' -v OFS='\\t' -v tgt_name=\"$target_name\" -v qry_name=\"$query_name\" '
                (\$9 ~ /INV/) {
                    tstart = (\$2 < \$3) ? \$2 : \$3;
                    tend = (\$2 > \$3) ? \$2 : \$3;
                    qstart = (\$7 < \$8) ? \$7 : \$8;
                    qend = (\$7 > \$8) ? \$7 : \$8;
                    pair = tstart \"-\" tend \"-\" qstart \"-\" qend;
                    if (!(pair in seen)) {
                        seen[pair] = 1;
                        print tgt_name, \$1, tstart, tend, qry_name, \$6, qstart, qend
                    }
                }' \"$syri_out\" >> \"$OUTFILE\""
                if ! execute_command "Extracting inversions for $chr" "$AWK_EXTRACT" true; then
                    echo -e "${YELLOW}✗${NC}"
                    continue
                fi
                # =====================================================================================

            done
        fi
    done
done

rm -r $TMP_DIR

log_entry "Analysis completed. Results available in '$OUTFILE'"
echo -e "\n\n${BLUE}Analysis completed.${NC}\nResults available in: ${GREEN}'$OUTFILE'${NC}"
exit 0