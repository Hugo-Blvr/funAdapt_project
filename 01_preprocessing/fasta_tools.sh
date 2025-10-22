#!/bin/bash
# =============================================================================
# FASTA Toolkit - FASTA File Manipulation Utility
# =============================================================================
#
# Description : Bash script for analysis and manipulation of FASTA files
#               (genomes, transcriptomes, proteomes)
#
# Features :
#   • sizes   : Sequence length extraction (iso, chr, size)
#   • convert : Case conversion (uppercase/lowercase)  
#   • chr     : Chromosome/contig manipulation (extraction/removal)
#
# Usage : ./fasta_tools.sh <command> [options]
#         ./fasta_tools.sh --help for more information
#
# Author    : Hugo Bellavoir
# Email     : bellavoirh@gmail.com
# Version   : 1.0
# License   : MIT
#
# Dependencies : bash (>=4.0), awk, sed
# Tested on    : Linux
#
# =============================================================================

set -euo pipefail

show_help() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
  sizes   <fasta_folder_or_file> [output_file]
           → extrait la taille de chaque séquence (iso, chr, size)
           → output_file par défaut: sizes.tsv

  convert <fasta_folder_or_file> <upper|lower>
           → convertit les séquences en majuscules ou minuscules

  chr     <path_to_fasta> <chr_name> <remove|extract>
           → supprime ou extrait un chromosome/contig donné

Examples:
  $0 sizes genomes/ custom_output.tsv
  $0 convert genome.fa upper
  $0 chr genome.fa chr2 remove
EOF
}

# -------------------------- SIZES --------------------------
sizes() {
    input="$1"
    output="${2:-sizes.tsv}"  # Par défaut 'sizes.tsv' si pas d'argument

    > "$output"

    # Détermine si input est un dossier ou un fichier
    if [ -d "$input" ]; then
        fasta_list=("$input"/*.fasta "$input"/*.fa)
    elif [ -f "$input" ]; then
        fasta_list=("$input")
    else
        echo "Erreur: $input n'est ni un dossier ni un fichier valide"
        exit 1
    fi

    for fasta in "${fasta_list[@]}"; do
        [ -e "$fasta" ] || continue
        filename=$(basename "$fasta")
        iso="${filename%%.*}" # enlève tout après le premier point

        awk -v iso="$iso" '
            /^>/ {
                if (seqname != "") {
                    printf "%s\t%s\t%d\n", iso, seqname, length(seq) >> "'$output'"
                }
                seqname = substr($0, 2)
                seq = ""
                next
            }
            { seq = seq $0 }
            END {
                if (seqname != "") {
                    printf "%s\t%s\t%d\n", iso, seqname, length(seq) >> "'$output'"
                }
            }
        ' "$fasta"
    done

    sed -i '1i iso\tchr\tsize' "$output"
    echo "Tailles écrites dans $output"
}

# -------------------------- CONVERT --------------------------
convert() {
    input="$1"
    mode="$2"

    if [ "$mode" != "upper" ] && [ "$mode" != "lower" ]; then
        echo "Erreur : le deuxième argument doit être 'upper' ou 'lower'."
        exit 1
    fi

    if [ "$mode" == "upper" ]; then
        AWK_CMD='{if($0 ~ /^>/) print $0; else print toupper($0)}'
    else
        AWK_CMD='{if($0 ~ /^>/) print $0; else print tolower($0)}'
    fi

    if [ -d "$input" ]; then
        for file in "$input"/*.fasta "$input"/*.fa; do
            [ -e "$file" ] || continue
            awk "$AWK_CMD" "$file" > "$file.tmp" && mv "$file.tmp" "$file"
        done
        echo "Conversion $mode terminée pour tous les fichiers dans $input."
    elif [ -f "$input" ]; then
        awk "$AWK_CMD" "$input" > "$input.tmp" && mv "$input.tmp" "$input"
        echo "Conversion $mode terminée pour le fichier $input."
    else
        echo "Erreur : $input n'est ni un fichier ni un dossier."
        exit 1
    fi
}

# -------------------------- CHR --------------------------
chr() {
    fasta_file="$1"
    chr_name="$2"
    action="$3"
    temp_file=$(mktemp)

    if [ ! -f "$fasta_file" ]; then
        echo "Erreur : fichier '$fasta_file' introuvable."
        exit 1
    fi

    # Vérifier d'abord si le chromosome existe dans le fichier
    chr_exists=$(awk -v chr="$chr_name" '
        /^>/ {
            # Extraire le nom exact du chromosome (premier champ après >)
            chrname = substr($1, 2)
            if (chrname == chr) {
                print "found"
                exit
            }
        }
    ' "$fasta_file")

    if [ "$chr_exists" != "found" ]; then
        echo "Erreur : chromosome '$chr_name' introuvable dans '$fasta_file'."
        echo "Chromosomes disponibles :"
        awk '/^>/ { print "  " substr($1, 2) }' "$fasta_file" | head -10
        [ $(awk '/^>/ { count++ } END { print count+0 }' "$fasta_file") -gt 10 ] && echo "  ... (et $(( $(awk '/^>/ { count++ } END { print count+0 }' "$fasta_file") - 10 )) autres)"
        exit 1
    fi

    case "$action" in
        remove)
            awk -v chr="$chr_name" '
                BEGIN {skip=0}
                /^>/ {
                    # Extraire le nom exact du chromosome
                    chrname = substr($1, 2)
                    if (chrname == chr) {
                        skip=1
                    } else {
                        skip=0
                    }
                }
                !skip
            ' "$fasta_file" > "$temp_file"
            mv "$temp_file" "$fasta_file"
            echo "Chromosome '$chr_name' supprimé de '$fasta_file'."
            ;;
        extract)
            awk -v chr="$chr_name" '
                BEGIN {extract=0}
                /^>/ {
                    # Extraire le nom exact du chromosome
                    chrname = substr($1, 2)
                    if (chrname == chr) {
                        extract=1
                    } else {
                        extract=0
                    }
                }
                extract
            ' "$fasta_file" > "${chr_name}.fasta"
            echo "Chromosome '$chr_name' extrait dans '${chr_name}.fasta'."
            ;;
        *)
            echo "Erreur : action doit être 'remove' ou 'extract'."
            exit 1
            ;;
    esac
}

# -------------------------- DISPATCH --------------------------
if [ "$#" -lt 1 ]; then
    show_help
    exit 1
fi

cmd="$1"
shift

case "$cmd" in
    sizes)   sizes "$@" ;;
    convert) convert "$@" ;;
    chr)     chr "$@" ;;
    *)       show_help; exit 1 ;;
esac