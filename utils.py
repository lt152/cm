import logging
import time


class Utils:

    @staticmethod
    def get_time():
        str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        return str_time


str_time = Utils.get_time()

# 创建日志
logger = logging.getLogger("1")
logger.setLevel(logging.INFO)
# 创建两个handler，用于写入日志文件和控制台展示
fh = logging.FileHandler('logs/' + str_time + '.log', encoding="utf-8", mode="a")
ch = logging.StreamHandler()

# 创建一个formatter并设置其格式
formatter = logging.Formatter('%(asctime)s %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
