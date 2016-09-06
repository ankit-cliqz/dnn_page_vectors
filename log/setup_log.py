#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import logging.config
import yaml


def setup_logging(default_path=str(os.path.dirname(os.path.realpath(__file__)))+'/logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)

    if value:
        path = value
    if os.path.exists(path):
        # print "Using Yaml log Config."
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        # print "Using Basic Log Config."
        logging.basicConfig(level=default_level)

