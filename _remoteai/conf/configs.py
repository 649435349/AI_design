# -*- coding: utf-8 -*-
import logging

RPC_SERVER_HOST = '127.0.0.1'
RPC_SERVER_PORT = 29898

TRAIN_CLIENT_TYPE = 'train'

# logger
DEFAULT_LOGGER_NAME = 'remoteai'
DEFAULT_LOGGER_LEVEL = logging.INFO
DEFAULT_LOGGER_FILE = '../../logs/remoteai.log'
DEFAULT_LOGGER_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s'

DEFAULT_MQ_FILE = '../../resources/need_to_start_pvp.json'

# model
DEFAULT_PG_MODEL = '../../resources/pg.model'
DEFAULT_MC_MODEL = '../../resources/mc_zhanshi.model'

WPCT_FILE = '../../resources/wpct.csv'