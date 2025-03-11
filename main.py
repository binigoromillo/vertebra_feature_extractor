# main.py

import sys
import os
import yaml
from omegaconf import OmegaConf, DictConfig
from vertebra_feature_extractor.src.features import extract_features, extract_features_from_df
from vertebra_feature_extractor.src.visualize import vis_features, vis_inspection
from vertebra_feature_extractor.src.predict import predict_quality_seg
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger(__name__)

def load_config(config_path):
    # Using OmegaConf to load the YAML file
    return OmegaConf.load(config_path)

def main():
    # Ensure the config file is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file.yaml>")
        sys.exit(1)

    config_file = sys.argv[1]
    if not os.path.isfile(config_file):
        print(f"Config file {config_file} not found.")
        sys.exit(1)

    # Load configuration from YAML file
    config = load_config(config_file)
    run = config.run

    # Call the extract_features function with the loaded config
    if run == "extract_features":
        extract_features(config.extract_features)
        # extract_features_from_df(config.extract_features)
    elif run == "predict":
        predict_quality_seg(config.predict)
    elif run == "vis_features":
        vis_features(config.visualize.vis_features)
    elif run == "vis_inspection":
        vis_inspection(config.visualize.vis_inspection)

if __name__ == "__main__":
    main()
