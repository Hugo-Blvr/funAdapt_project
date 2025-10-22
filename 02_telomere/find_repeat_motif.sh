#!/bin/bash
#
# =============================================================================
# Telomere Motif Finder - Repetitive Telomeric Motif Detection Tool
# =============================================================================
#
# Description : Bash script for automated identification of telomeric motifs
#               in genomic sequences from long-read sequencing data
#               (Oxford Nanopore Technologies)
#
# Features :
#   • Long-read alignment to reference genome (minimap2)
#   • Soft-clip extraction at chromosomal extremities (5' and 3')
#   • Repetitive motif detection using TRF (Tandem Repeats Finder)
#   • Analysis and classification of potential telomeric motifs
#   • Detailed TSV format report generation
#
# Analysis Pipeline :
#   1. Genome indexing and alignment with long reads
#   2. Extraction of soft-clipped regions at extremities (≥50nt)
#   3. FASTA formatting of sequences of interest
#   4. TRF analysis to identify tandem repeats
#   5. Filtering and classification of motifs (≥3 copies)
#   6. Generation of candidate telomeric motif lists
#
# Usage : ./find_repeat_motif.sh -g <genome.fasta> -l <longreads.fastq> [options]
#         ./find_repeat_motif.sh --help for more information
#
# Required arguments :
#   -g, --genome     : Reference genome FASTA file
#   -l, --longreads  : Long reads FASTQ file (ONT)
#
# Optional arguments :
#   -o, --output     : Output directory (default: current directory)
#   -t, --threads    : Number of threads (default: 8)
#   -h, --help       : Display help message
#
# Prerequisites :
#   • minimap2       : Long-read alignment
#   • samtools       : BAM/SAM file manipulation
#   • trf            : Tandem Repeats Finder
#   • python3        : Motif processing (create_list_motif.py required)
#
# Output structure :
#   output/
#   ├── bam/                     # Alignment files
#   ├── soft_clip/               # Extracted soft-clipped sequences
#   ├── trf/                     # TRF results and filtered motifs
#   ├── *_list_motif_start.tsv   # 5' telomeric motifs
#   ├── *_list_motif_end.tsv     # 3' telomeric motifs
#   └── find_repeat_motif.log    # Execution log
#
# TRF Parameters : 2 7 7 80 4 10 500
#   Match=2, Mismatch=7, Delta=7, PM=80, PI=4, Minscore=10, MaxPeriod=500
#
# Applied Filters :
#   • Soft-clips ≥ 50 nucleotides at chromosomal extremities
#   • Repetitive motifs with ≥ 3 consecutive copies
#   • Regions of interest: first/last 1000bp + 8000bp extension
#
# Author    : Hugo Bellavoir
# Email     : bellavoirh@gmail.com
# Version   : 1.0
# License   : MIT
#
# Tested on : Linux with bash ≥ 4.0
#
# =============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m';
BLUE='\033[0;34m'; NC='\033[0m'

help() {
    echo
    echo -e "${BLUE}Usage:${NC} $0 -g <genome.fasta> -l <longreads.fastq> [options]\n"
    echo -e "${BLUE}Required options:${NC}"
    echo "  -g, --genome <file>       FASTA file of the reference genome"
    echo "  -l, --longreads <file>    FASTQ file of long reads (ONT)"
    echo -e "${BLUE}Optional arguments:${NC}"
    echo "  -h, --help                Display this help message"
    echo "  -o, --output <directory>  Output directory for results (default: current directory)"
    echo "  -t, --threads <number>    Number of threads to use (default: 8)"
    echo -e "\n${YELLOW}IMPORTANT: ${NC}The Python script 'create_list_motif.py' must be present in the same directory as $0\n"
    exit 0
}

check_prerequisites() {
    local missing_tools=()
    for tool in minimap2 samtools trf python3; do
        command -v "$tool" &>/dev/null || missing_tools+=("$tool")
    done

    if ((${#missing_tools[@]})); then
        echo -e "\n${RED}Error: The following tools are not installed or not in PATH:${NC}"
        printf "${RED}  - %s\n${NC}" "${missing_tools[@]}"
        echo -e "Please install these tools before running this script.\n"
        exit 1
    fi
}

check_file() {
    local file="$1"
    local description="$2"

    if [ ! -f "$file" ] || [ ! -r "$file" ]; then
        echo -e "${RED}Error: The $description file '$file' is not found or not readable.${NC}"
        log_message "ERROR: File '$file' not found or not readable"
        exit 1
    elif [ ! -s "$file" ]; then
        echo -e "${YELLOW}Warning: The $description file '$file' is empty.${NC}"
        log_message "WARNING: File '$file' is empty"
    fi
}

create_directory() {
    local directory="$1"
    # If directory doesn't exist, try to create it
    if [ ! -d "$directory" ]; then
        log_message "Creating directory '$directory'"
        mkdir -p "$directory" || { 
            echo -e "${RED}Error: Cannot create directory '$directory'.${NC}"
            log_message "ERROR: Failed to create directory '$directory'"
            exit 1
        }
    fi
    # Check that the directory is writable
    if [ ! -w "$directory" ]; then
        echo -e "${RED}Error: Directory '$directory' is not writable.${NC}"
        log_message "ERROR: Directory '$directory' not writable"
        exit 1
    fi
}

log_message() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"; }

execute_command() {
    local description="$1"           
    local command="$2"              
    local ignore_error="${3:-false}" # Flag indicating if errors should be ignored (default "false")
    
    log_message "EXECUTION: $description" 
    
    OUTPUT=$(eval "$command" 2>&1) # Execute command in subshell and capture output
    local STATUS=$?  # Get command's return code
    
    if [ $STATUS -eq 0 ] || [ "$ignore_error" == "true" ] || { [[ "$description" == *"TRF"* ]] && [ -f "$DAT_FILE_START" ]; }; then
        return 0  # If command executed successfully or if errors are ignored
    else
        # If an error occurred, display error message and log it
        echo -e "  ${RED}✗${NC} Error during $description"
        log_message "ERROR: $description: $OUTPUT"
        return 1
    fi
}

# Initialize default variables
SCRIPT_DIR=$(dirname "$(realpath "$0")")
GENOME=""
LONGREADS=""
OUTDIR="."
THREADS=8

# Process options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            help
            ;;
        -g|--genome)
            GENOME="$2"
            shift 2
            ;;
        -l|--longreads)
            LONGREADS="$2"
            shift 2
            ;;
        -o|--output)
            OUTDIR="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
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
if [ -z "$GENOME" ] || [ -z "$LONGREADS" ]; then
    echo -e "${RED}Error: Options -g/--genome and -l/--longreads are mandatory.${NC}"
    echo "Use '$0 --help' for more information."
    exit 1
fi

check_prerequisites
LOG_FILE="$OUTDIR/find_repeat_motif.log"
[ -f "$LOG_FILE" ] && rm "$LOG_FILE"

check_file "$GENOME" "genome"
check_file "$LONGREADS" "long reads"
GENOME=$(realpath "$GENOME")
LONGREADS=$(realpath "$LONGREADS")

# Check Python script existence
PYTHON_SCRIPT="$SCRIPT_DIR/create_list_motif.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python script '$PYTHON_SCRIPT' does not exist.${NC}"
    log_message "ERROR: Python script '$PYTHON_SCRIPT' not found"
    exit 1
fi

# Create output directories
create_directory "$OUTDIR"
OUTDIR=$(realpath "$OUTDIR")
BAM_OUTPUT_DIR="$OUTDIR/bam"
SOFT_CLIP_OUTPUT_DIR="$OUTDIR/soft_clip"
TRF_OUTPUT_DIR="$OUTDIR/trf"
create_directory "$BAM_OUTPUT_DIR"
create_directory "$SOFT_CLIP_OUTPUT_DIR" 
create_directory "$TRF_OUTPUT_DIR"

# Define files
GENOME_BASE=$(basename "${GENOME}" | cut -d. -f1)
MMI_FILE="$BAM_OUTPUT_DIR/${GENOME_BASE}.mmi"
SORTED_BAM="$BAM_OUTPUT_DIR/${GENOME_BASE}_mapping.sorted.bam"
CHR_LENGTHS_FILE="$OUTDIR/${GENOME_BASE}_chr_lengths.tsv"

# Soft-clips files
SOFT_CLIP_START_TSV="$SOFT_CLIP_OUTPUT_DIR/${GENOME_BASE}_softclip_start.tsv"
SOFT_CLIP_END_TSV="$SOFT_CLIP_OUTPUT_DIR/${GENOME_BASE}_softclip_end.tsv"
SOFT_CLIP_START_FASTA="$SOFT_CLIP_OUTPUT_DIR/${GENOME_BASE}_softclip_start.fasta"
SOFT_CLIP_END_FASTA="$SOFT_CLIP_OUTPUT_DIR/${GENOME_BASE}_softclip_end.fasta"

# TRF files
DAT_FILE_START="$TRF_OUTPUT_DIR/${GENOME_BASE}_softclip_start.fasta.2.7.7.80.4.10.500.dat"
DAT_FILE_END="$TRF_OUTPUT_DIR/${GENOME_BASE}_softclip_end.fasta.2.7.7.80.4.10.500.dat"
FILTERED_REPEATS_START="$TRF_OUTPUT_DIR/${GENOME_BASE}_filtered_repeats_start.fasta"
FILTERED_REPEATS_END="$TRF_OUTPUT_DIR/${GENOME_BASE}_filtered_repeats_end.fasta"

# Output files
OUT_NAME_START="$OUTDIR/${GENOME_BASE}_list_motif_start.tsv"
OUT_NAME_END="$OUTDIR/${GENOME_BASE}_list_motif_end.tsv"
# ======================================================================================


# Display parameter summary
echo -e "${BLUE}====== Parameters ======${NC}"
echo -e "Genome:                 ${GREEN}$GENOME${NC}"
echo -e "Long reads:             ${GREEN}$LONGREADS${NC}"
echo -e "Output directory:       ${GREEN}$OUTDIR${NC}"
echo -e "Threads:                ${GREEN}$THREADS${NC}\n"


############################# Step 1: Long reads alignment ############################
echo -e "${BLUE}Aligning long reads to genome $(basename "${GENOME}")${NC}"
if [ ! -f "$MMI_FILE" ]; then
    echo -e "\tIndexing genome..."
    execute_command "Genome indexing" "minimap2 -d \"$MMI_FILE\" \"$GENOME\""
fi
if [ ! -f "$SORTED_BAM" ]; then
    echo -e "\tAligning long reads..."
    execute_command "Long reads alignment" \
        "minimap2 -t $THREADS -ax map-ont \"$MMI_FILE\" \"$LONGREADS\" | \
        samtools view -bS | samtools sort -@ $THREADS -o \"$SORTED_BAM\""
fi
if [ ! -f "${SORTED_BAM}.bai" ]; then
    echo -e "\tIndexing BAM file...\n"
    execute_command "BAM indexing" "samtools index \"$SORTED_BAM\""
fi
################################################################################################################


############################# Step 2: Extraction of soft-clips of interest and FASTA formatting ############################
echo -e "${BLUE}Extracting soft-clips of interest${NC}"

# Extract chromosome lengths
if [ ! -f "$CHR_LENGTHS_FILE" ]; then
    echo -e "\tExtracting chromosome lengths..."
    execute_command "Extracting chromosome lengths" \
        "samtools idxstats \"$SORTED_BAM\" > \"$CHR_LENGTHS_FILE\""
fi

# Check if output files exist
if [ ! -f "$SOFT_CLIP_START_TSV" ] || [ ! -f "$SOFT_CLIP_END_TSV" ]; then
    # Build AWK command with inline script
    # Analyzes BAM alignments using samtools + awk to detect heavily soft-clipped reads
    # at the chromosome ends (5' and 3'), aiming to identify tandem repeat patterns
    # indicative of telomeric sequences.
    # - Reads chromosome lengths from $CHR_LENGTHS_FILE
    # - For each aligned read (excluding empty sequences '*'):
    #     * Computes the start and end positions based on the CIGAR string
    #     * Determines the soft-clipping length at the beginning of the read
    #     * If the read starts within the first 1000 bp and extends beyond 8000 bp,
    #       with soft-clipping ≥ 50 nt → writes to $SOFT_CLIP_START_TSV
    #     * If the read covers the last 8000 to 1000 bp of the chromosome,
    #       with soft-clipping ≥ 50 nt → writes to $SOFT_CLIP_END_TSV

    AWK_CMD="samtools view \"$SORTED_BAM\" | awk -v OFS='\t' -v chr_lengths_file=\"$CHR_LENGTHS_FILE\" -v soft_clip_start_tsv=\"$SOFT_CLIP_START_TSV\" -v soft_clip_end_tsv=\"$SOFT_CLIP_END_TSV\" '
    BEGIN {
        while ((getline < chr_lengths_file) > 0) { chr_lengths[\$1] = \$2 }
        close(chr_lengths_file)
    }
    {
        if (\$10 == \"*\") next
        chr = \$3; start = \$4
        if (!(chr in chr_lengths)) next
        ref_length = chr_lengths[chr]
        
        if (match(\$6, /^([0-9]+)S/, m)) { soft_clip = m[1] } else { soft_clip = 0 }
        
        cigar = \$6; mapped_length = 0
        while (match(cigar, /([0-9]+)([MID])/)) {
            len = substr(cigar, RSTART, RLENGTH - 1)
            op = substr(cigar, RSTART + RLENGTH - 1, 1)
            if (op == \"M\" || op == \"D\") mapped_length += len
            cigar = substr(cigar, RSTART + RLENGTH)
        }
        
        end = start + mapped_length - 1
        min_start = ref_length - 8000
        min_end = ref_length - 1000
        
        if (start <= 1000 && end >= 8000 && soft_clip >= 50) {
            print \$1, chr, start, end, \$10 >> soft_clip_start_tsv
        }
        
        if (start <= min_start && end >= min_end && soft_clip >= 50) {
            print \$1, chr, start, end, \$10 >> soft_clip_end_tsv
        }
    }'"
    
    echo -e "\tExtracting soft-clips..."
    if execute_command "Soft-clips extraction" "$AWK_CMD" false; then
        if ! [[ -f "$SOFT_CLIP_START_TSV" && -f "$SOFT_CLIP_END_TSV" ]]; then
            log_message "ERROR: No soft-clips extracted"
            exit 1
        fi
    else
        log_message "ERROR: Soft-clips extraction failed"
        exit 1
    fi
fi

# Format to FASTA files
echo -e "\tFormatting extracted soft-clips to .fasta...\n"
if [ ! -f "$SOFT_CLIP_START_FASTA" ]; then
    execute_command "Formatting soft-clips (5')" "awk -F '\t' '{print \">\"\$2\":\"\$1\":\"\$3\"-\"\$4\"\\n\"\$5}' \"$SOFT_CLIP_START_TSV\" > \"$SOFT_CLIP_START_FASTA\""
fi
if [ ! -f "$SOFT_CLIP_END_FASTA" ]; then
    execute_command "Formatting soft-clips (3')" "awk -F '\t' '{print \">\"\$2\":\"\$1\":\"\$3\"-\"\$4\"\\n\"\$5}' \"$SOFT_CLIP_END_TSV\" > \"$SOFT_CLIP_END_FASTA\""
fi

# Check presence of sequences
for fasta_file in "$SOFT_CLIP_START_FASTA" "$SOFT_CLIP_END_FASTA"; do
    if [ ! -s "$fasta_file" ]; then
        echo -e "${YELLOW}Warning: File '$fasta_file' is empty. No soft-clips were found.${NC}"
        log_message "WARNING: File '$fasta_file' is empty"
    fi
done

rm $CHR_LENGTHS_FILE
################################################################################################################


############################ Step 3: Detection of repeat motifs with TRF ############################
echo -e "${BLUE}Detecting repeat motifs in extracted soft-clips${NC}"
# Run TRF for 5' sequences
if [ -s "$SOFT_CLIP_START_FASTA" ] && [ ! -f "$DAT_FILE_START" ]; then
    echo -e "\tTRF analysis of 5' soft-clips..."
    execute_command "TRF analysis of 5' soft-clips" "(cd \"$TRF_OUTPUT_DIR\" && trf \"$SOFT_CLIP_START_FASTA\" 2 7 7 80 4 10 500 -f -h -d)"
fi
# Run TRF for 3' sequences
if [ -s "$SOFT_CLIP_END_FASTA" ] && [ ! -f "$DAT_FILE_END" ]; then
    echo -e "\tTRF analysis of 3' soft-clips..."
    execute_command "TRF analysis of 3' soft-clips" "(cd \"$TRF_OUTPUT_DIR\" && trf \"$SOFT_CLIP_END_FASTA\" 2 7 7 80 4 10 500 -f -h -d)"
fi

# Function to filter repeats
filter_repeats() {
    local input_file="$1"
    local output_file="$2"
    
    if [ -f "$input_file" ] && [ -s "$input_file" ] && [ ! -f "$output_file" ]; then
        awk 'BEGIN {seq_name=""}
            /^Sequence:/ {seq_name=$2}
            $4 >= 3 {
                print ">" seq_name " repeat:" $1 "-" $2 " motif=" $14 " copies=" $4
                print $15
            }' "$input_file" > "$output_file"
            
        if ! [ -s "$output_file" ]; then
            echo -e "${YELLOW}Warning: No repetitive motifs found in $(basename "$input_file").${NC}"
            log_message "WARNING: No repetitive motifs found in '$(basename "$input_file")'"
            touch "$output_file"  # Create empty file to avoid errors
            return 0  # Consider this case a success
        fi
    else
        echo -e "${YELLOW}Warning: File '$(basename "$input_file")' does not exist or is empty.${NC}"
        log_message "WARNING: File '$(basename "$input_file")' not available for filtering"
        touch "$output_file"  # Create empty file to avoid errors
        return 0  # Consider this case a success
    fi
}

# Filter repeats
if [ -s "$DAT_FILE_START" ] && [ ! -f "$FILTERED_REPEATS_START" ]; then
    echo -e "\tFiltering 5' repetitive motifs..."
    execute_command "Filtering repetitive motifs (5')" "filter_repeats '$DAT_FILE_START' '$FILTERED_REPEATS_START'"
fi
if [ -s "$DAT_FILE_END" ] && [ ! -f "$FILTERED_REPEATS_END" ]; then
    echo -e "\tFiltering 3' repetitive motifs...\n"
    execute_command "Filtering repetitive motifs (3')" "filter_repeats '$DAT_FILE_END' '$FILTERED_REPEATS_END'"
fi
################################################################################################################


############################# Step 4: Generating telomere list ############################
# Generate telomere lists
echo -e "${BLUE}Generating lists of potential telomeric motifs...${NC}"
# Process 5' motifs
if [ -s "$FILTERED_REPEATS_START" ] && [ ! -f "$OUT_NAME_START" ]; then
    echo -e "\tGenerating list of 5' telomeric motifs..."
    execute_command "Generating list of 5' telomeric motifs" "python3 \"$PYTHON_SCRIPT\" \"$FILTERED_REPEATS_START\" \"$OUT_NAME_START\""
elif [ ! -s "$FILTERED_REPEATS_START" ]; then
    echo -e "${YELLOW}Warning: No 5' repetitive motifs to process.${NC}"
    log_message "WARNING: No 5' repetitive motifs to process"
    touch "$OUT_NAME_START"  # Create empty file 
fi

# Process 3' motifs
if [ -s "$FILTERED_REPEATS_END" ] && [ ! -f "$OUT_NAME_END" ]; then
    echo -e "\tGenerating list of 3' telomeric motifs...\n"
    execute_command "Generating list of 3' telomeric motifs" "python3 \"$PYTHON_SCRIPT\" \"$FILTERED_REPEATS_END\" \"$OUT_NAME_END\""
elif [ ! -s "$FILTERED_REPEATS_END" ]; then
    echo -e "${YELLOW}Warning: No 3' repetitive motifs to process.${NC}"
    log_message "WARNING: No 3' repetitive motifs to process"
    touch "$OUT_NAME_END"  # Create empty file
fi
################################################################################################################

#rm -r "$BAM_OUTPUT_DIR"
#rm -r "$SOFT_CLIP_OUTPUT_DIR" 
#rm -r "$TRF_OUTPUT_DIR"

log_message "\nAnalysis complete. Results available in '$OUTDIR'"
echo -e "\n${BLUE}Analysis complete.${NC}\nResults available in: ${GREEN}'$OUTDIR'${NC}"
exit 0