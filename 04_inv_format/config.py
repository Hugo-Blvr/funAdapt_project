#!/usr/bin/env python3
"""
Configuration management for inversion processing pipeline.
"""

import json
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field, asdict

@dataclass
class OutputConfig:
    """Output files configuration."""
    save_nofilter: bool = True  # INV_*_nofilter.tsv
    save_wTE: bool = True        # INV_*_clean_wTE.tsv
    save_TE_files: bool = True   # TE_*_isTE.tsv and TE_*_inTE.tsv

@dataclass
class InputConfig:
    """Input data configuration."""
    excluded_chromosomes: List[str] = field(default_factory=lambda: ['chr2003', 'chr2016', 'chr2017'])


@dataclass
class FilteringConfig:
    """Inversion filtering parameters."""
    min_inversion_size: int = 150

@dataclass
class MergingConfig:
    """Inversion merging parameters."""
    reciprocal_overlap_ratio: float = 0.85
    fusion_gap_threshold: int = 100

@dataclass
class ClusteringConfig:
    """Clustering and spatial indexing parameters."""
    spatial_bucket_size: int = 50000
    min_overlap_ratio: float = 0.8


@dataclass
class TEFilteringConfig:
    """Transposable element filtering configuration."""
    enabled: bool = True
    overlap_threshold: float = 90.0
    te_file: Optional[str] = None


@dataclass
class LineageConfig:
    """Lineage information configuration."""
    enabled: bool = True
    lineage_file: Optional[str] = None


@dataclass
class PerformanceConfig:
    """Performance and parallelization settings."""
    n_cores: Optional[int] = None


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    output: OutputConfig = field(default_factory=OutputConfig)
    input: InputConfig = field(default_factory=InputConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    merging: MergingConfig = field(default_factory=MergingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    te_filtering: TEFilteringConfig = field(default_factory=TEFilteringConfig)
    lineage: LineageConfig = field(default_factory=LineageConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def from_json(cls, json_path: Path) -> 'PipelineConfig':
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            PipelineConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If JSON is malformed or contains invalid values
        """
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        return cls(
            output=OutputConfig(**data.get('output', {})),
            input=InputConfig(**data.get('input', {})),
            filtering=FilteringConfig(**data.get('filtering', {})),
            merging=MergingConfig(**data.get('merging', {})),
            clustering=ClusteringConfig(**data.get('clustering', {})),
            te_filtering=TEFilteringConfig(**data.get('te_filtering', {})),
            lineage=LineageConfig(**data.get('lineage', {})),
            performance=PerformanceConfig(**data.get('performance', {}))
        )
    
    def to_json(self, json_path: Path) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path where to save the configuration
        """
        config_dict = asdict(self)
        
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate ratios (0-1 range)        
        if not 0 <= self.merging.reciprocal_overlap_ratio <= 1:
            raise ValueError(f"merging.reciprocal_overlap_ratio must be between 0 and 1, got {self.merging.reciprocal_overlap_ratio}")
        
        if not 0 <= self.clustering.min_overlap_ratio <= 1:
            raise ValueError(f"clustering.min_overlap_ratio must be between 0 and 1, got {self.clustering.min_overlap_ratio}")
        
        # Validate percentages (0-100 range)
        if not 0 <= self.te_filtering.overlap_threshold <= 100:
            raise ValueError(f"te_filtering.overlap_threshold must be between 0 and 100, got {self.te_filtering.overlap_threshold}")
        
        # Validate positive integers
        if self.filtering.min_inversion_size < 0:
            raise ValueError(f"filtering.min_inversion_size must be positive, got {self.filtering.min_inversion_size}")
        
        if self.merging.fusion_gap_threshold < 0:
            raise ValueError(f"merging.fusion_gap_threshold must be positive, got {self.merging.fusion_gap_threshold}")
        
        if self.clustering.spatial_bucket_size <= 0:
            raise ValueError(f"clustering.spatial_bucket_size must be positive, got {self.clustering.spatial_bucket_size}")
        
        # Validate file paths if provided
        if self.te_filtering.enabled and self.te_filtering.te_file:
            te_path = Path(self.te_filtering.te_file)
            if not te_path.exists():
                raise FileNotFoundError(f"TE file not found: {te_path}")
        
        if self.lineage.enabled and self.lineage.lineage_file:
            lineage_path = Path(self.lineage.lineage_file)
            if not lineage_path.exists():
                raise FileNotFoundError(f"Lineage file not found: {lineage_path}")
    
    @staticmethod
    def create_default_config(output_path: Path) -> None:
        """
        Create a default configuration file.
        
        Args:
            output_path: Path where to save the default config
        """
        default_config = PipelineConfig()
        default_config.to_json(output_path)
        print(f"Default configuration created: {output_path}")