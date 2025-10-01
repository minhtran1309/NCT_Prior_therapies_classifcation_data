"""
Utility functions for clinical trial prior therapy classification.
"""

import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Setup logging configuration."""
    if config and 'logging' in config:
        log_config = config['logging']
        level = getattr(logging, log_config.get('level', 'INFO'))
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file')
        
        # Create logs directory if it doesn't exist
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=level,
            format=format_str,
            filename=log_file,
            filemode='a'
        )
    else:
        logging.basicConfig(level=logging.INFO)
    
    return logging.getLogger(__name__)


def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories based on configuration."""
    if 'paths' in config:
        paths = config['paths']
        for path_name, path_value in paths.items():
            Path(path_value).mkdir(exist_ok=True)
    
    # Create evaluation output directory
    if 'evaluation' in config and 'output_dir' in config['evaluation']:
        Path(config['evaluation']['output_dir']).mkdir(exist_ok=True)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def validate_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return Path(file_path).exists()


def ensure_directory_exists(directory_path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)