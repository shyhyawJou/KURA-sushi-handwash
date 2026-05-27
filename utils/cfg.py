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
        if info['washcountmax'] > 0:
            info['washcountmax'] += 1 * time_scale  # 為了讓 UI 能到達及格搓洗次數
        info['washtimemax'] = float(info['washtimemax']) * time_scale
        info['washtimemax'] += 0.5 * time_scale  # 為了讓 UI 能到達及格秒數
        info['timeoutmax'] = float(info['timeoutmax']) * time_scale
    for name in time_param:
        if name == 'time_scale':
            continue
        CFG['logic']['time_parameter'][name] *= time_scale


logger.info(f'config: {CFG}')
logger.info(f'system config: {SYS_CFG}')
