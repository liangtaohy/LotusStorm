# -*- coding: utf-8 -*-
import logging
import os
import logging.config

APP_NAME = 'LotusStorm'

logging.config.fileConfig(os.path.join(os.path.dirname(__file__), APP_NAME + '.conf'))

# create logger
logger = logging.getLogger(APP_NAME)