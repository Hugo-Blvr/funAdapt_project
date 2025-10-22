import os
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def parse_mapping_file(mapping_file):
    """Parse le fichier de correspondance TSV."""
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            old_name, new_name = line.strip().split('\t')
            mapping[old_name] = new_name
    return mapping

def analyze_contig_presence(fasta_files):
    """Analyse la présence des contigs dans tous les fichiers FASTA."""
    contig_presence = defaultdict(int)
    total_files = len(fasta_files)
    
    # Compte le nombre d'occurrences de chaque contig
    for fasta_file in fasta_files:
        contigs = set()
        for record in SeqIO.parse(fasta_file, "fasta"):
            contigs.add(record.id)
        for contig in contigs:
            contig_presence[contig] += 1
    
    # Classifie les contigs selon les règles :
    # - core : présent dans tous les fichiers
    # - accessoire : présent dans 2 ou plus fichiers, mais pas tous
    # - unique : présent dans 1 seul fichier
    core_contigs = {c for c, count in contig_presence.items() if count == total_files}
    accessory_contigs = {c for c, count in contig_presence.items() if 1 < count < total_files}
    unique_contigs = {c for c, count in contig_presence.items() if count == 1}
    
    print(f"Nombre total de fichiers FASTA : {total_files}")
    print(f"Contigs core (présents dans tous les fichiers) : {len(core_contigs)}")
    print(f"Contigs accessoires (présents dans 2+ fichiers mais pas tous) : {len(accessory_contigs)}")
    print(f"Contigs uniques (présents dans 1 seul fichier) : {len(unique_contigs)}")
    
    return core_contigs, accessory_contigs, unique_contigs

def generate_new_names(core_contigs, accessory_contigs, unique_contigs):
    """Génère les nouveaux noms"""
    mapping = {}
    
    # Extrait les numéros X de chrX et trie
    def extract_num(chr_name): return int(chr_name.replace('chr', ''))
    
    # Traite les contigs core (1000 + X)
    for contig in sorted(core_contigs, key=extract_num):
        mapping[contig] = f'chr{1000 + extract_num(contig)}'
    
    # Traite les contigs accessoires (2000 + X)
    for contig in sorted(accessory_contigs, key=extract_num): 
        mapping[contig] = f'chr{2000 + extract_num(contig)}'
    
    # Traite les contigs uniques (3000 + X)
    for contig in sorted(unique_contigs, key=extract_num):
        mapping[contig] = f'chr{3000 + extract_num(contig)}'
    
    return mapping

def save_mapping(mapping, output_file):
    """Sauvegarde la correspondance dans un fichier TSV."""
    with open(output_file, 'w') as f:
        for old_name, new_name in mapping.items():
            f.write(f'{old_name}\t{new_name}\n')

def rename_fasta(input_file, output_file, mapping):
    """Renomme les contigs dans un fichier FASTA."""
    records = []
    for record in SeqIO.parse(input_file, "fasta"):
        if record.id in mapping:
            new_record = SeqRecord(
                seq=record.seq,
                id=mapping[record.id],
                description=f""
            )
            records.append(new_record)
        else:
            records.append(record)
    
    SeqIO.write(records, output_file, "fasta")

def main(input_dir, mapping_file=None):
    """Fonction principale."""
    # Récupère tous les fichiers FASTA du dossier
    fasta_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                   if f.endswith(('.fasta', '.fa'))]
    
    if not fasta_files:
        print(f"Aucun fichier FASTA trouvé dans {input_dir}")
        return
    
    if mapping_file:
        # Utilise le mapping fourni
        mapping = parse_mapping_file(mapping_file)
        print(f"Utilisation du fichier de mapping : {mapping_file}")
    else:
        # Crée un nouveau mapping
        print("Analyse des contigs...")
        core, accessory, unique = analyze_contig_presence(fasta_files)
        mapping = generate_new_names(core, accessory, unique)
        mapping_output = os.path.join(input_dir, 'contig_mapping.tsv')
        save_mapping(mapping, mapping_output)
        print(f"Fichier de mapping créé : {mapping_output}")
    
    # Renomme chaque fichier FASTA
    print("Renommage des fichiers FASTA...")
    for fasta_file in fasta_files:
        base_name = os.path.basename(fasta_file)
        output_file = os.path.join(input_dir, base_name)
        rename_fasta(fasta_file, output_file, mapping)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(f"Usage: python3 rename_chr.py input_directory [mapping_file]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    mapping_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(input_dir, mapping_file)