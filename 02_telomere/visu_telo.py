import pandas as pd
import json
import numpy as np
import os
import sys
from pathlib import Path

def parse_tsv_file(file_path):
    df = pd.read_csv(file_path, sep="\t")
    df["total_repeats"] = df["forward_repeat_number"] + df["reverse_repeat_number"]
    
    chromosomes = []
    for chrom, group in sorted(df.groupby('id'), key=lambda x: int(x[0].replace('chr', ''))):
        chrom_size_kb = round(group['window'].max() / 1000, 2)
        repeats_data = group[['window', 'total_repeats']].to_dict('records')
        color_window_size = max(1, len(group) // 80)
        color_data = []
        
        for i in range(0, len(group), color_window_size):
            window_group = group.iloc[i:i+color_window_size]
            color_data.append({
                'window': f"{window_group['window'].min()} - {window_group['window'].max()}",
                'total_repeats': int(window_group['total_repeats'].sum())
            })
        
        chromosomes.append({
            'id': chrom,
            'size_kb': chrom_size_kb,
            'repeats_data': repeats_data,
            'color_data': color_data
        })
    
    return chromosomes

def generate_multi_genome_html(genome_data):
    # Convert genome_data to JSON for JavaScript
    data_json = json.dumps(genome_data)
    
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>Genome Telomeric Repeats Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px; margin: 0 auto;
            padding: 10px; min-height: 100vh;
        }
        .container {
            background: white; padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: calc(100vh - 20px); display: flex;
            flex-direction: column; overflow: hidden;
        }
        h1 {
            margin: 0 0 10px 0; font-size: 1.5em;
        }
        .tabs {
            display: flex;
            border-bottom: 1px solid #ccc;
            margin-bottom: 10px;
            flex-wrap: wrap;      /* Permet aux onglets de passer à la ligne */
            gap: 4px;            /* Espace entre les onglets */
            max-height: 100px;    /* Hauteur maximale pour la zone des onglets */
            overflow-y: auto;     /* Ajoute un défilement vertical si nécessaire */
        }
        .tab {
            padding: 8px 16px;
            cursor: pointer;
            border: 1px solid transparent;
            border-bottom: none;
            margin-bottom: -1px;
            background: #f8f9fa;
            border-radius: 4px 4px 0 0;
            margin-right: 4px;
        }
        .tab.active {
            background: white;
            border-color: #ccc;
            border-bottom-color: white;
        }
        .genome-view {
            display: none;
            height: calc(100% - 50px);
        }
        .genome-view.active {
            display: block;
        }
        .grid-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px; padding: 10px;
            overflow-y: auto; flex-grow: 1;
        }
        .chromosome-container {
            margin: 2; padding: 10px;
            background: #ffffff; border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .chromosome-title {
            font-weight: bold; margin: 3px 0; font-size: 0.9em;
        }
        .repeats-graph {
            height: 40px; position: relative;
            margin: 3px 0; background: #f8f9fa;
        }
        .repeats-bar {
            position: absolute;
            bottom: 0; width: 2px;
            margin-right: 1px; background-color: #2166ac;
            transition: background-color 0.3s;
        }
        .repeats-bar:hover {
            background-color: #ff4444;
        }
        .chromosome {
            position: relative;
            height: 25px; width: 100%;
            margin: 1px auto; background: #f0f0f0;
            border-radius: 12px; overflow: hidden;
            border: 1px solid #cccccc;
        }
        .repeat-segment {
            position: absolute;
            height: 100%; top: 0;
            transition: background-color 0.3s;
        }
        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white; padding: 4px 8px; border-radius: 4px;
            font-size: 12px; pointer-events: none;
            z-index: 1000; display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Genome Telomeric Repeats Visualization</h1>
        <div class="tabs" id="tabs"></div>
        <div id="genome-views"></div>
        <div class="tooltip" id="tooltip"></div>
    </div>

    <script>
        const genomeData = ''' + data_json + ''';
        const tooltip = document.getElementById('tooltip');
        const tabsContainer = document.getElementById('tabs');
        const genomesContainer = document.getElementById('genome-views');
        
        // Create tabs and genome views
        Object.keys(genomeData).forEach((genomeName, index) => {
            // Create tab
            const tab = document.createElement('div');
            tab.className = `tab ${index === 0 ? 'active' : ''}`;
            tab.textContent = genomeName;
            tab.onclick = () => switchTab(genomeName);
            tabsContainer.appendChild(tab);
            
            // Create genome view
            const genomeView = document.createElement('div');
            genomeView.className = `genome-view ${index === 0 ? 'active' : ''}`;
            genomeView.id = `genome-${genomeName}`;
            
            const gridContainer = document.createElement('div');
            gridContainer.className = 'grid-container';
            genomeView.appendChild(gridContainer);
            genomesContainer.appendChild(genomeView);
            
            // Process chromosome data
            const chromosomes = genomeData[genomeName];
            const allRepeatsData = chromosomes.flatMap(chrom => chrom.repeats_data);
            const allColorData = chromosomes.flatMap(chrom => chrom.color_data);
            const globalMaxRepeatsForGraph = Math.max(...allRepeatsData.map(d => d.total_repeats));
            const globalMaxRepeatsForColor = Math.max(...allColorData.map(d => d.total_repeats));
            
            chromosomes.forEach(chromosome => {
                const chromContainer = document.createElement('div');
                chromContainer.className = 'chromosome-container';
                
                const title = document.createElement('div');
                title.className = 'chromosome-title';
                title.textContent = `${chromosome.id} (${chromosome.size_kb} kb)`;
                chromContainer.appendChild(title);
                
                const repeatsGraph = document.createElement('div');
                repeatsGraph.className = 'repeats-graph';
                
                chromosome.repeats_data.forEach((d, i) => {
                    const bar = document.createElement('div');
                    bar.className = 'repeats-bar';
                    bar.style.left = `${(i / (chromosome.repeats_data.length - 1)) * 100}%`;
                    bar.style.height = `${(d.total_repeats / globalMaxRepeatsForGraph) * 40}px`;
                    
                    bar.addEventListener('mousemove', (e) => {
                        tooltip.style.display = 'block';
                        tooltip.style.left = `${e.pageX + 10}px`;
                        tooltip.style.top = `${e.pageY - 25}px`;
                        tooltip.textContent = `Position: ${d.window}, Repeats: ${Math.round(d.total_repeats)}`;
                    });
                    
                    bar.addEventListener('mouseout', () => {
                        tooltip.style.display = 'none';
                    });
                    
                    repeatsGraph.appendChild(bar);
                });
                
                chromContainer.appendChild(repeatsGraph);
                
                const chromVisual = document.createElement('div');
                chromVisual.className = 'chromosome';
                
                const width = 100 / chromosome.color_data.length;
                
                chromosome.color_data.forEach((d, i) => {
                    const segment = document.createElement('div');
                    segment.className = 'repeat-segment';
                    segment.style.left = `${i * width}%`;
                    segment.style.width = `${width}%`;
                    const ratio = d.total_repeats / globalMaxRepeatsForColor;
                    const blue = 255;
                    const red = Math.round(255 * (1 - ratio));
                    const green = Math.round(255 * (1 - ratio));
                    segment.style.backgroundColor = `rgb(${red}, ${green}, ${blue})`;
                    segment.title = `Position: ${d.window}, Repeats: ${d.total_repeats}`;
                    chromVisual.appendChild(segment);
                });
                
                chromContainer.appendChild(chromVisual);
                gridContainer.appendChild(chromContainer);
            });
        });
        
        function switchTab(genomeName) {
            // Update tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
                if (tab.textContent === genomeName) {
                    tab.classList.add('active');
                }
            });
            
            // Update genome views
            document.querySelectorAll('.genome-view').forEach(view => {
                view.classList.remove('active');
                if (view.id === `genome-${genomeName}`) {
                    view.classList.add('active');
                }
            });
        }
    </script>
</body>
</html>
    '''
    
    return html_content

def main(input_files,output_file):
    # Get all TSV files in the directory
    tsv_files = list([Path(file.strip()) for file in input_files.split()])
    
    # Process each TSV file and store the data
    genome_data = {}
    for tsv_file in tsv_files:
        
        # Extract genome name from filename (assuming format: NAME_*.tsv)
        # ==================================SPECIFIQUE A NOS DONNEES =============================
        genome_name = tsv_file.name.split('_')[0]+'_'+tsv_file.name.split('_')[1]
        # ========================================================================================
        
        genome_data[genome_name] = parse_tsv_file(tsv_file)
    
    # Generate HTML with all genome data
    html_content = generate_multi_genome_html(genome_data)
    
    # Write the HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    # Gestion des arguments en ligne de commande
    if len(sys.argv) != 5:
        print("Usage: python script.py --input_files <dossier> --output_file <fichier_html>")
        sys.exit(1)

    input_files = sys.argv[2]  # Dossier d'entrée
    output_file = sys.argv[4]  # Fichier de sortie HTML

    main(input_files, output_file)

