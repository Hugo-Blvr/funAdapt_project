import os
import yaml  # type: ignore
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Set, List
import pandas as pd


class ConfigLoader:
    """Loads and validates configuration from a YAML file."""

    # Default color palette (matplotlib Set1-like)
    DEFAULT_COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    
    def __init__(self, config_yaml_path: str, log_level: int = logging.INFO):
        """
        Initialize the configuration loader.
        
        Args:
            config_yaml_path: Path to the YAML configuration file
            log_level: Logging level (default: logging.INFO)
        """
        self._setup_logging(log_level)
        self.config_path = Path(config_yaml_path)
        self._inversions_paths_cache = None
        
        # Vérification simple avant de charger
        if not self.config_path.exists():
            print(f"Configuration file not found: {self.config_path.absolute()}")
            exit(1)
        
        self.config = self._load_yaml()
        self._validate_config()
    
    def _setup_logging(self, log_level: int):
        """Configure logging with a clean format."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError("YAML configuration file is empty")
        
        self.logger.info(f"Configuration loaded from: {self.config_path}")
        return config
    
    def _detect_separator(self, file_path: str) -> str:
        """
        Automatically detect the separator in a delimited file.
        Args:
            file_path: Path to the file
        Returns:
            Detected separator (comma, tab, etc.)
        """
        file = Path(file_path)
        ext = file.suffix.lower()

        if ext == ".csv": return ","
        elif ext == ".tsv": return "\t"
        elif ext == ".txt":
            with open(file, "r", encoding="utf-8") as f: first_line = f.readline()
            
            tab_count = first_line.count("\t")
            comma_count = first_line.count(",")
            
            if tab_count > comma_count: return "\t"
            elif comma_count > 0: return ","
            else:
                self.logger.warning(
                    f"Could not detect separator in {file.name}, defaulting to tab")
                return "\t"
        else: return "\t"

    def _validate_config(self):
        """Validate the structure of the configuration."""
        if 'input' not in self.config:
            raise ValueError("Missing required section 'input' in configuration")
        
        # Check required paths
        required_paths = ['outpath']
        paths = self.config['input'].get('paths', {})
        
        for req_path in required_paths:
            if req_path not in paths:
                raise ValueError(
                    f"Missing required path in configuration: input.paths.{req_path}")
        
        # Check inversions (at least one method required)
        inversions = paths.get('inversions', {})
        if not inversions:
            raise ValueError(
                "At least one inversion file must be specified in input.paths.inversions")
    
    def _validate_file(self, file_path: str, required_cols: Set[str], label: str) -> Optional[str]:
        """
        Validate a CSV/TSV file and return its path if valid.
        Args:
            file_path: Path to the file to validate
            required_cols: Set of required column names
            label: Label for logging (e.g., method name)
        Returns:
            File path if valid, None otherwise
        """
        file = Path(file_path)
        
        if not file.is_file():
            self.logger.warning(f"[{label}] File not found: '{file_path}'")
            return None
        
        sep = self._detect_separator(file_path)
        
        try: df = pd.read_csv(file, sep=sep, nrows=5)
        except Exception as e:
            self.logger.error(f"[{label}] Failed to read file: {e}")
            return None
        
        found_cols = set(df.columns)
        missing_cols = required_cols - found_cols
        
        if missing_cols:
            self.logger.error(
                f"[{label}] Missing required columns: {sorted(missing_cols)}\n"
                f"         Found columns: {sorted(found_cols)}\n"
                f"         Required columns: {sorted(required_cols)}")
            return None
        
        self.logger.info(f"[{label}] File validated successfully ({len(df.columns)} columns)")
        return file_path

    def _get_inversions_paths(self) -> Dict[str, str]:
        """
        Get paths to valid inversion files.
        Returns:
            Dictionary mapping method names to validated file paths  
        Raises:
            FileNotFoundError: If no valid inversion files are found
        """
        if self._inversions_paths_cache is not None: return self._inversions_paths_cache
        inversions = self.config["input"]["paths"].get("inversions", {})
        
        if not inversions: 
            raise ValueError("No inversion paths defined in configuration")
        
        required_cols = {"iso", "chr", "start", "end", "mapon"}
        valid_files = {}
        
        for method, inv_file in inversions.items():
            validated = self._validate_file(inv_file, required_cols, method)
            if validated: valid_files[method] = validated
        
        if not valid_files:
            raise FileNotFoundError(
                f"No valid inversion files found.\n"
                f"Attempted files: {list(inversions.values())}")
        
        self._inversions_paths_cache = valid_files
        return valid_files

    def _load_inversions(self) -> Dict[str, pd.DataFrame]:
        """
        Load all inversion DataFrames.
        Returns:
            Dictionary mapping method names to DataFrames
        """
        inversions_paths = self._get_inversions_paths()
        inversions_dfs = {}
        
        for method, file_path in inversions_paths.items():
            sep = self._detect_separator(file_path)
            df = pd.read_csv(file_path, sep=sep)
            inversions_dfs[method] = df
            self.logger.info(f"Loaded {len(df)} inversions from [{method}]")
        return inversions_dfs

    def _get_inversion_colors(self) -> Dict[str, str]:
        """
        Get color mapping for inversion methods.
        
        Returns:
            Dictionary mapping method names to hex colors
        """
        inversions_paths = self._get_inversions_paths()
        colors_config = self.config['input'].get('colors', {}).get('inversions_colors', {})
        default_color = colors_config.get('other', None)

        color_map = {}
        color_idx = 0

        for method_key in inversions_paths.keys():
            if method_key in colors_config: color_map[method_key] = colors_config[method_key]
            elif default_color: color_map[method_key] = default_color
            else:
                color_map[method_key] = self.DEFAULT_COLORS[color_idx % len(self.DEFAULT_COLORS)]
                color_idx += 1

        return color_map

    def _load_lineages(self, available_isos: List[str]) -> Optional[pd.DataFrame]:
        """
        Load and process lineages DataFrame.
        Args:
            available_isos: List of available isolates to filter by
        Returns:
            DataFrame with columns ['iso', 'lineage', 'color'] or None
        """
        lineage_file = self.config['input']['paths'].get('lineages')
        
        if not lineage_file:
            self.logger.info("No lineages file specified in configuration")
            return None
        
        required_cols = {"O_SSC", "Species"}
        validated = self._validate_file(lineage_file, required_cols, "lineages")
        
        if not validated:
            self.logger.warning("Lineages file validation failed (non-blocking)")
            return None

        # Load and rename columns
        sep = self._detect_separator(validated)
        df = pd.read_csv(validated, sep=sep)
        df = df.rename(columns={"O_SSC": "iso", "Species": "lineage"})[["iso", "lineage"]]

        # Filter by available isolates
        df = df.query("iso in @available_isos")
        df = df.drop_duplicates(subset='iso')

        # Add colors
        lineages_colors = self.config['input'].get('colors', {}).get('lineages_colors', {})
        default_color = lineages_colors.get('other', None)
        unique_lineages = df['lineage'].unique()

        color_map = {}
        color_idx = 0

        for lineage in unique_lineages:
            if lineage in lineages_colors: color_map[lineage] = lineages_colors[lineage]
            elif default_color: color_map[lineage] = default_color
            else:
                color_map[lineage] = self.DEFAULT_COLORS[color_idx % len(self.DEFAULT_COLORS)]
                color_idx += 1

        df['color'] = df['lineage'].map(color_map)
        self.logger.info(f"Loaded {len(df)} lineages ({len(unique_lineages)} unique species)")
        
        return df

    def _load_chr_lengths(self, available_isos: List[str]) -> Optional[pd.DataFrame]:
        """
        Load chromosome lengths DataFrame.
        Args:
            available_isos: List of available isolates to filter by
        Returns:
            DataFrame with chromosome lengths or None
        """
        chr_file = self.config['input']['paths'].get('chr_lengths')
        
        if not chr_file:
            self.logger.info("No chromosome lengths file specified in configuration")
            return None
        
        required_cols = {"iso", "chr", "size"}
        validated = self._validate_file(chr_file, required_cols, "chr_lengths")
        
        if not validated:
            self.logger.warning("Chromosome lengths file validation failed (non-blocking)")
            return None
        
        sep = self._detect_separator(validated)
        df = pd.read_csv(validated, sep=sep)
        
        # Filter by available isolates
        df = df.query("iso in @available_isos")
        
        self.logger.info(f"Loaded chromosome lengths for {df['iso'].nunique()} isolates")
        return df

    def _get_outpath(self) -> str:
        """
        Get output path and create directory if it doesn't exist.
        
        Returns:
            Output path as string
        """
        outpath = Path(self.config['input']['paths']['outpath'])

        if outpath.exists():
            if outpath.is_dir(): self.logger.info(f"Output directory exists: {outpath}")
            else: raise ValueError(f"Output path exists but is not a directory: {outpath}")
        else:
            outpath.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created output directory: {outpath}")

        return str(outpath)

    def _get_iso_ref(self, available_isos: List[str]) -> List[str]:
        """
        Get validated reference isolates list.
        
        Args:
            available_isos: List of available isolates in the data
            
        Returns:
            List of validated reference isolates. If not defined or empty after
            validation, returns all available isolates.
        """
        iso_ref_config = self.config['input'].get('iso_ref')
        
        if iso_ref_config is None:
            self.logger.info("No iso_ref defined, using all available isolates")
            return available_isos
        
        # Convert string to list
        if isinstance(iso_ref_config, str): iso_ref_config = [iso_ref_config]
        
        if not isinstance(iso_ref_config, list):
            self.logger.warning(
                f"iso_ref must be a string or list, got {type(iso_ref_config).__name__}. "
                f"Using all available isolates")
            return available_isos
        
        # Filter valid isolates
        valid_isos = [iso for iso in iso_ref_config if iso in available_isos]
        invalid_isos = [iso for iso in iso_ref_config if iso not in available_isos]
        
        if invalid_isos:
            self.logger.warning(
                f"Isolates not found in data: {invalid_isos}")
        
        if not valid_isos:
            self.logger.warning(
                "No valid iso_ref found, using all available isolates")
            return available_isos
        
        self.logger.info(f"Validated {len(valid_isos)} reference isolates: {valid_isos}")
        return valid_isos

    def validate_optional_data(self, inv_df, lineage_df, chr_df):
        """
        Validates and completes optional data for chromosomal inversions.
        
        Args:
            inv_df: Dictionary of DataFrames containing inversion data
            lineage_df: DataFrame with lineage information (optional)
            chr_df: DataFrame with theoretical chromosome sizes (optional)
        
        Returns:
            tuple: chromosome_sizes_df, completed_lineage_df
        """
        
        # Merge all inversion DataFrames and find maximum position per chromosome
        all_inversions = pd.concat(inv_df.values(), ignore_index=True)
        experimental_sizes = (all_inversions.groupby(['iso', 'chr'], as_index=False)['end']
                            .max()
                            .rename(columns={'end': 'size'}))

        if chr_df is not None:
            # Merge experimental sizes with theoretical chromosome data
            merged_data = experimental_sizes.merge(
                chr_df, 
                on=["iso", "chr"], 
                how="left", 
                suffixes=("_exp", "_ref")
            )

            # Identify chromosome pairs missing from reference data
            missing_chromosomes = merged_data[merged_data["size_ref"].isna()][["iso", "chr"]]
            if not missing_chromosomes.empty:
                print("\nWarning: The following chromosomes are missing from the reference file:")
                print(missing_chromosomes.to_string(index=False))
                print("Their sizes will be estimated from the furthest inversion position.\n")

            # Use reference size when available, otherwise fall back to experimental
            merged_data["size"] = merged_data["size_ref"].fillna(merged_data["size_exp"])
            chromosome_sizes = merged_data[["iso", "chr", "size"]]
        else:
            print("\nInfo: No reference chromosome sizes provided.")
            print("Chromosome sizes will be estimated from inversion positions.\n")
            chromosome_sizes = experimental_sizes

        # Handle lineage data
        unique_isolates = chromosome_sizes['iso'].unique()
        
        if lineage_df is None:
            # Create default lineage DataFrame
            print(f"Info: Creating default lineage data for {len(unique_isolates)} isolates.\n")
            lineage_df = pd.DataFrame({
                'iso': unique_isolates,
                'lineage': ['unassigned'] * len(unique_isolates),
                'color': ['#000000'] * len(unique_isolates)
            })
        else:
            # Check for missing isolates in lineage data
            existing_isolates = set(lineage_df['iso'])
            missing_isolates = [iso for iso in unique_isolates if iso not in existing_isolates]

            if missing_isolates:
                print(f"\nWarning: {len(missing_isolates)} isolate(s) missing from lineage data:")
                print(", ".join(missing_isolates))
                print("They will be assigned to 'unknown' lineage.\n")
                
                missing_lineage_data = pd.DataFrame({
                    'iso': missing_isolates,
                    'lineage': ['unknown'] * len(missing_isolates),
                    'color': ['#808080'] * len(missing_isolates)  # Grey for unknown
                })
                lineage_df = pd.concat([lineage_df, missing_lineage_data], ignore_index=True)

        return chromosome_sizes, lineage_df

    def load_all(self) -> Dict[str, Any]:
        """
        Load and validate all configuration data.
        Returns:
            Dictionary containing all loaded data:
            {
                "inversions": {method_name: DataFrame, ...},
                "lineages": DataFrame or None,
                "chr_lengths": DataFrame or None,
                "iso_ref": [iso1, iso2, ...],
                "outpath": "/path/to/output",
                "colors": {method: color, ...}
                }
            }
        """
        self.logger.info("=" * 60)
        self.logger.info("Loading configuration data")
        self.logger.info("=" * 60)
        
        # Load inversions
        inversions_dfs = self._load_inversions()
        
        # Get all available isolates
        available_isos = []
        for df in inversions_dfs.values(): available_isos.extend(df['iso'].unique().tolist())
        available_isos = list(set(available_isos))
        self.logger.info(f"Found {len(available_isos)} unique isolates in inversion data")
        
        # Get reference isolates
        iso_ref = self._get_iso_ref(available_isos)
        
        # Load optional data
        lineages_data = self._load_lineages(available_isos)
        chr_lengths_data = self._load_chr_lengths(available_isos)
        
        # Get colors
        inv_colors = self._get_inversion_colors()
        
        # Get output path
        outpath = self._get_outpath()
        
        # Clean & verif
        chr_lengths_df, lineages_df = self.validate_optional_data(inversions_dfs ,lineages_data, chr_lengths_data)

        self.logger.info("=" * 60)
        self.logger.info("Configuration loading completed successfully")
        self.logger.info("=" * 60)


        te_path = self.config["input"]["paths"].get("te_pathfile", {})
        gene_path = self.config["input"]["paths"].get("gene_pathfile", {})
        
        return {
            "inversions": inversions_dfs,
            "lineages": lineages_df,
            "chr_lengths": chr_lengths_df,
            "te_path": te_path,
            "gene_path": gene_path,
            "iso_ref": iso_ref,
            "outpath": outpath,
            "colors":  inv_colors
            }

    def summary(self) -> str:
        """
        Generate a summary of the configuration.
        Returns:
            Formatted summary string
        """
        lines = ["Configuration Summary", "=" * 60]
        
        lines.append("\nInversion files:")
        try:
            inversions = self._get_inversions_paths()
            for method, path in inversions.items():
                lines.append(f"  • {method}: {path}")
        except Exception as e: lines.append(f"  Error: {e}")
        
        lines.append("\nOptional files:")
        lineage_path = self.config['input']['paths'].get('lineages', 'Not specified')
        chr_path = self.config['input']['paths'].get('chr_lengths', 'Not specified')
        lines.append(f"  • Lineages: {lineage_path}")
        lines.append(f"  • Chromosome lengths: {chr_path}")
        
        lines.append("\nOutput:")
        try: lines.append(f"  • Output path: {self._get_outpath()}")
        except Exception as e: lines.append(f"  Error: {e}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    try:
        # Load configuration
        config = ConfigLoader(config_path, log_level=logging.INFO)
        
        # Display summary
        print("\n" + config.summary())
        
        # Load all data
        print("\n")
        data = config.load_all()
        # Display loaded data structure
        print("\n" + "=" * 60)
        print("Loaded Data Structure")
        print("=" * 60)
        print(f"\nInversions methods: {list(data['inversions'].keys())}")
        for method, df in data['inversions'].items():
            print(f"  • {method}: {len(df)} rows, {len(df.columns)} columns")
        
        print(f"\n {len(data['iso_ref'])} Reference isolate(s) : {data['iso_ref']}")
        
        print(f"\nOutput path: {data['outpath']}")
        
        df_lineage = data['lineages']
        lineage_colors = dict(df_lineage[['lineage', 'color']].drop_duplicates().values)

        print(f"\nColors:")
        print(f"  • Inversions: {data['colors']}")
        print(f"  • Lineage: {lineage_colors}")  
              
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Configuration error: {e}")
        exit(1)