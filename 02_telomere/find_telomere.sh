#!/bin/bash
#
# =============================================================================
# Telomere Search Tool - Batch Telomeric Motif Detection and Visualization
# =============================================================================
#
# Description : Bash script for high-throughput detection of telomeric motifs
#               across multiple FASTA genome files using TIDK (Telomere
#               Identification Toolkit) with optional interactive visualization
#
# Features :
#   • Batch processing of multiple FASTA files in a directory
#   • Telomeric motif detection using TIDK search functionality
#   • Automatic tripling of input motifs for enhanced detection
#   • Comprehensive logging and error handling
#   • Optional HTML visualization generation (requires visu_telo.py)
#   • Support for .fasta and .fa file extensions
#
# Analysis Workflow :
#   1. Directory scanning for FASTA files (.fasta extension)
#   2. Motif tripling for improved repeat detection sensitivity
#   3. TIDK search execution on each genome file
#   4. Results compilation in TSV format
#   5. Optional interactive HTML visualization generation
#   6. Comprehensive logging of all operations
#
# Usage : ./telo_search.sh -d <directory> -m <motif> [options]
#         ./telo_search.sh --help for more information
#
# Required arguments :
#   -d, --directory  : Path to directory containing FASTA files (.fasta or .fa)
#   -m, --motif      : DNA motif to search for (e.g., TTAGGG, TTTAGGG)
#
# Optional arguments :
#   -o, --output     : Output directory (default: ./out_telomere_region)
#   -v, --visu       : HTML visualization output filename
#   -h, --help       : Display help message
#
# Prerequisites :
#   • tidk           : Telomere Identification Toolkit
#   • find           : File search utility (standard Unix tool)
#   • python         : Required only for visualization option
#   • visu_telo.py   : Python visualization script (same directory)
#
# Output structure :
#   output_directory/
#   ├── *_telomeric_repeat_windows.tsv  # TIDK results for each genome
#   ├── telo_search.log                 # Execution log
#   └── [visualization.html]            # Optional interactive visualization
#
# TIDK Integration :
#   • Telomeric repeat window detection and quantification
#   • Position mapping and statistical analysis
#   • TSV output format for downstream analysis
#
# Supported Formats :
#   • Input : FASTA files (.fasta, .fa extensions)
#   • Output: TSV (TIDK results), HTML (visualization), LOG (execution)
#
# Error Handling :
#   • File accessibility validation
#   • Empty file detection and skipping
#   • Missing dependency checks
#   • Comprehensive logging system
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
    echo -e "${BLUE}Usage:${NC} $0 -d <directory> -m <motif> [options]\n"
    echo -e "${BLUE}Required options:${NC}"
    echo "  -d, --directory <directory>  Path to the directory containing FASTA files (.fasta or .fa)"
    echo "  -m, --motif <motif>          DNA motif to search for"
    echo -e "${BLUE}Optional options:${NC}"
    echo "  -h, --help                   Display this help message"
    echo "  -o, --output <directory>     Specify the output directory (default: ./out_telomere_region)"
    echo "  -v, --visu <visu_file>       Output filename for visualization"
    echo -e "\n${YELLOW}IMPORTANT: ${NC}The Python script 'visu_telo.py' must be present in the same directory as $0"
    echo -e " if the -v/--visu option is used.\n"
    exit 0
}

check_prerequisites() {
    local tools=(tidk find)
    [[ -n "$VISU_FILE_NAME" ]] && tools+=(python)  # Add python if VISU_FILE_NAME is defined

    local missing_tools=()
    for tool in "${tools[@]}"; do
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
    # Check existence and readability of the directory
    if [ ! -d "$directory" ] || [ ! -r "$directory" ]; then
        echo -e "${RED}Error: The $description directory '$directory' does not exist or is not readable.${NC}"
        log_entry "ERROR: Directory '$directory' not found or not readable"
        exit 1
    fi
}

create_directory() {
    local directory="$1"
    # If the directory doesn't exist, try to create it
    if [ ! -d "$directory" ]; then
        log_entry "Creating directory '$directory'"
        mkdir -p "$directory" || { 
            echo -e "${RED}Error: Unable to create directory '$directory'.${NC}"
            log_entry "ERROR: Failed to create directory '$directory'"
            exit 1
        }
    fi
    # Check that the directory is writable
    if [ ! -w "$directory" ]; then
        echo -e "${RED}Error: The directory '$directory' is not writable.${NC}"
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
    
    OUTPUT=$(eval "$command" 2>&1) # Execute the command in a subshell and capture output
    local STATUS=$?  # Capture the return code of the command
    
    if [ $STATUS -eq 0 ] || [ "$ignore_error" == "true" ]; then
        return 0  # if the command executed successfully or if we're ignoring errors
    else
        # If an error occurred, display an error message and log it
        echo -e "  ${RED}✗${NC} Error during $description"
        log_entry "ERROR: $description: $OUTPUT"
        return 1
    fi
}

# Initialize default variables
SCRIPT_DIR=$(dirname "$(realpath "$0")")
INDIR=""
MOTIF=""
OUTDIR="./out_telomere_region"
VISU_FILE_NAME=""

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
        -m|--motif)
            MOTIF="$2"
            shift 2
            ;;
        -o|--output)
            OUTDIR="$2"
            shift 2
            ;;
        -v|--visu)
            VISU_FILE_NAME="$(realpath "$2")"
            shift 2
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            echo "Use '$0 --help' to see available options."
            exit 1
            ;;
    esac
done

# =========== Parameter validation and preparation of the output environment ========
if [ -z "$INDIR" ] || [ -z "$MOTIF" ]; then
    echo -e "${RED}Error: The options -d/--directory and -m/--motif are required.${NC}"
    echo "Use '$0 --help' for more information."
    exit 1
fi

check_prerequisites
LOG_FILE="$OUTDIR/telo_search.log"
[ -f "$LOG_FILE" ] && rm "$LOG_FILE"

check_directory "$INDIR" "input"
INDIR=$(realpath "$INDIR")

MOTIF_TRIPLE="${MOTIF}${MOTIF}${MOTIF}"
GENERATED_FILES=()
# Check the existence of the Python script if visualization is requested + check the visualization extension
PYTHON_SCRIPT="$SCRIPT_DIR/visu_telo.py"
if [ -n "$VISU_FILE_NAME" ]; then
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo -e "${RED}Error: The Python script '$PYTHON_SCRIPT' does not exist.${NC}"
        echo "This script is necessary for the visualization option."
        log_entry "ERROR: Python script '$PYTHON_SCRIPT' not found"
        exit 1
    fi
    if [ ! -r "$PYTHON_SCRIPT" ]; then
        echo -e "${RED}Error: The Python script '$PYTHON_SCRIPT' is not readable.${NC}"
        log_entry "ERROR: Python script '$PYTHON_SCRIPT' not readable"
        exit 1
    fi

    if [[ "$VISU_FILE_NAME" != *.html ]]; then
        VISU_FILE_NAME="${VISU_FILE_NAME}.html"
    fi
fi

OUTDIR=$(realpath "$OUTDIR")
create_directory "$OUTDIR"
# ======================================================================================



# ===== Retrieve and check the list of FASTA files in the directory =====
fasta_files=($(find "$INDIR" -maxdepth 1 -type f -iname "*.fasta"))
if [ ${#fasta_files[@]} -eq 0 ]; then
    echo -e "${RED}Error: No .fasta files found in directory $INDIR.${NC}"
    log_entry "ERROR: No .fasta files found in directory $INDIR"
    exit 1
fi
# ======================================================================================

# Display a summary of operations to perform
echo -e "${BLUE}=== Operations Summary ===${NC}"
echo -e "Input directory:    ${GREEN}$INDIR${NC}"
echo -e "Output directory:   ${GREEN}$OUTDIR${NC}"
echo -e "Search motif:       ${GREEN}$MOTIF${NC}"
echo -e "FASTA files:        ${GREEN}${#fasta_files[@]}${NC}"
if [ -n "$VISU_FILE_NAME" ]; then
    echo -e "HTML file:          ${GREEN}$VISU_FILE_NAME${NC}\n"
else
    echo
fi


FILES_PROCESSED=0
# Loop through each FASTA file
echo -e "${BLUE}Starting telomere search${NC}"
for FASTA_FILE in "${fasta_files[@]}"; do
    FILENAME=$(basename "$FASTA_FILE" | cut -d. -f1)  # Get the filename
    echo -e "${NC}\tProcessing file: ${GREEN}$FILENAME${NC} (${YELLOW}$((++FILES_PROCESSED))/${#fasta_files[@]}${NC})"

    # Check that the file is readable
    if [ ! -r "$FASTA_FILE" ]; then
        echo -e "${RED}Warning: The file '$FASTA_FILE' is not readable. Skipped.${NC}"
        log_entry "ERROR: File not readable: '$FASTA_FILE'"
        continue
    fi
    # Check that the file is not empty
    if [ ! -s "$FASTA_FILE" ]; then
        echo -e "${YELLOW}Warning: The file '$FASTA_FILE' is empty. Skipped.${NC}"
        log_entry "WARNING: Empty file: '$FASTA_FILE'"
        continue
    fi

    # Execute the tidk search command with the tripled motif
    TIDK_CMD="tidk search -s \"$MOTIF_TRIPLE\" -o \"${FILENAME}_${MOTIF}\" -d \"$OUTDIR\" \"$FASTA_FILE\""
    if execute_command "TIDK analysis of '$FILENAME'" "$TIDK_CMD" true; then
        GENERATED_TSV="${OUTDIR}/${FILENAME}_${MOTIF}_telomeric_repeat_windows.tsv"
        GENERATED_FILES+=("$GENERATED_TSV")
    else
        log_entry "ERROR: Failed to analyze '$FILENAME'"
        exit 1
    fi

done

log_entry "Analysis completed. Results available in '$OUTDIR'"
echo -e "\n${BLUE}Analysis completed.${NC}\nResults available in: ${GREEN}'$OUTDIR'${NC}"

# If a visualization file is specified, run the visualization script
if [ -n "$VISU_FILE_NAME" ]; then
    echo
    echo -e "${BLUE}Creating visualization${NC}"
    GENERATED_FILES="${GENERATED_FILES[*]}"
    VISU_CMD="python \"$PYTHON_SCRIPT\" --input_files \"$GENERATED_FILES\" --output_file \"$VISU_FILE_NAME\""
    if execute_command "Generating visualization" "$VISU_CMD"; then
        log_entry "Visualization successfully generated in: '$VISU_FILE_NAME'"
        echo -e "\tVisualization successfully generated in: ${GREEN}'$VISU_FILE_NAME'${NC}"
    else
        echo -e "${RED}\tError generating visualization${NC}"
        exit 1
    fi
fi

exit 0