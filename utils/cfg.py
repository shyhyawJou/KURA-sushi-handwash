import yaml
import json
from loguru import logger



CFG = None
SYS_CFG = None


with open('utils/config.yaml') as f:
    CFG = yaml.safe_load(f)


with open(CFG['logic']['system_parameter']) as f:
    SYS_CFG = json.load(f)
    time_param = CFG['logic']['time_parameter']
    time_scale = time_param['time_scale']
    for info in SYS_CFG['stages']:
        info['washcountmax'] = int(info['washcountmax'])
        info['washtimemax'] = float(info['washtimemax']) * time_scale
        info['timeoutmax'] = float(info['timeoutmax']) * time_scale
    for name in time_param:
        if name == 'time_scale':
            continue
        CFG['logic']['time_parameter'][name] *= time_scale


if len(SYS_CFG['stages']) != CFG['logic']['handwash_parameter']['num_step']:
    logger.error(f"洗手步驟數不一致! 請檢查 {CFG['logic']['system_parameter']} 和 utils/config.yaml")
    raise
logger.info(f'config: {CFG}')
logger.info(f'system config: {SYS_CFG}')
