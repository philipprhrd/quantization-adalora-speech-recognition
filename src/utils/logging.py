import logging
from transformers import set_seed as hf_set_heed

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

def set_seed(seed):
    hf_set_heed(seed)