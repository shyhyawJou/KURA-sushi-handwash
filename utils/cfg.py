import yaml
from loguru import logger



CFG = None

with open('utils/config.yaml') as f:
    CFG = yaml.safe_load(f)
    logger.info(f'config: {CFG}')
