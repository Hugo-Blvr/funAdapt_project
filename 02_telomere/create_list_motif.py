import re
import sys

# Dictionnaire pour stocker les informations
data = {}
# Dictionnaire pour compter le nombre de chromosomes par motif canonique
motif_chromosome_count = {}

# Fonction pour obtenir la forme canonique d'un motif par rotation circulaire 
def get_canonical_motif(motif):
    rotations = [motif[i:] + motif[:i] for i in range(len(motif))]
    return min(rotations)  # Retourne la permutation la plus petite (lexicographiquement)

# Récupérer les arguments passés
fasta_path = sys.argv[1]
output_file = sys.argv[2]

# Lire le fichier FASTA et traiter les données
with open(fasta_path, 'r') as f:
    for line in f:
        if line.startswith('>'):
            # Extraire les informations à partir de la ligne d'en-tête
            match = re.match(r"^>(\S+):.*motif=(\S+).*copies=(\S+)", line)
            if match:
                chromosome = match.group(1).split(':')[0]
                motif = match.group(2)
                copies = float(match.group(3))
                
                if len(motif) >= 5 and len(motif) <= 8:
                    # Normaliser le motif (forme canonique)
                    canonical_motif = get_canonical_motif(motif)
                    
                    # Créer la clé (chromosome, motif canonique)
                    key = (chromosome, canonical_motif)
                    
                    # Si la clé n'existe pas encore, l'initialiser
                    if key not in data:
                        data[key] = {'max_copies': copies}
                    else:
                        # Mettre à jour max_copies et incrémenter le compteur
                        data[key]['max_copies'] = max(data[key]['max_copies'], copies)

                    # Compter le nombre de chromosomes distincts pour chaque motif canonique
                    if canonical_motif not in motif_chromosome_count:
                        motif_chromosome_count[canonical_motif] = set()
                    motif_chromosome_count[canonical_motif].add(chromosome)

# Créer et écrire dans le fichier TSV
with open(output_file, 'w') as out:
    out.write("chromosome\tmotif\tmax_copies\n")
    
    # Trier par motif
    for key in sorted(data.keys(), key=lambda x: x[1]):
        chromosome, motif = key
        max_copies = data[key]['max_copies']
        # Vérifier si le motif apparaît dans au moins 3 chromosomes
        if len(motif_chromosome_count[motif]) >= 3 and max_copies >=10:
            out.write(f"{chromosome}\t{motif}\t{int(max_copies)}\n")
