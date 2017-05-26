# -*- coding: utf-8 -*-
import logging, sys
sys.path.append('../../')

from conf import configs

class Logger():
    def __init__(self, name=configs.DEFAULT_LOGGER_NAME, log_file='', level=configs.DEFAULT_LOGGER_LEVEL):
        # 检查参数合法性
        if name is None or name == '':
            self._name = configs.DEFAULT_LOGGER_NAME
        else:
            self._name = name
            
        if level is None:
            self._level = configs.DEFAULT_LOGGER_LEVEL
        else:
            self._level = level
        
        # 创建logger实例
        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(self._level)
        self._formatter = logging.Formatter(configs.DEFAULT_LOGGER_FORMAT)
        
        # 输出到日志文件
        if log_file is not None and log_file != '':
            self._file_handler = logging.FileHandler(log_file)
            self._file_handler.setLevel(self._level)
            self._file_handler.setFormatter(self._formatter)
            self._logger.addHandler(self._file_handler)
        
        # 输出到控制台
        self._console_handler = logging.StreamHandler()
        self._console_handler.setLevel(self._level)
        self._console_handler.setFormatter(self._formatter)
        self._logger.addHandler(self._console_handler)
        
        #logging.basicConfig(level=self._level, 
        #            format='[%(asctime)s] [%(levelname)s] [%(filename)s@line:%(lineno)d] %(message)s', 
        #            datefmt='%Y-%m-%d %H:%M:%S', 
        #            filename='../../logs/remoteai_server.log',
        #            filemode='a')
    
    def debug(self, msg):
        self._logger.debug(msg)
    
    def info(self, msg):
        self._logger.info(msg)
        
    def warn(self, msg):
        self._logger.warn(msg)
        
    def error(self, msg):
        self._logger.error(msg)
        
    def critical(self, msg):
        self._logger.critical(msg)
        
    def exception(self, msg):
        self._logger.exception(msg)
        
    def fatal(self, msg):
        self._logger.fatal(msg)
        
    def log(self, level, msg):
        self._logger.log(level, msg)
        
    def setLevel(self, level):
        if level is None:
            return
        self._logger.setLevel(level)
        self._level = level
    
    def warning(self, msg):
        self._logger.warning(msg)
        