#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging

def is_Chinese(ch):
    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False



def get_logger(file_name):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(file_name)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

