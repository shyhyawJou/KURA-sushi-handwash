import yaml
import json
from loguru import logger



CFG = None
SYS_CFG = None


with open('utils/config.yaml') as f:
    CFG = yaml.safe_load(f)
logger.info(f'config: {CFG}')


with open(CFG['logic']['system_parameter']) as f:
    SYS_CFG = json.load(f)
    for info in SYS_CFG['stages']:
        info['washcountmax'] = int(info['washcountmax'])
        info['washtimemax'] = float(info['washtimemax']) * CFG['logic']['parameter']['time_scale']
        info['timeoutmax'] = float(info['timeoutmax']) * CFG['logic']['parameter']['time_scale']       
logger.info(f'system config: {SYS_CFG}')
