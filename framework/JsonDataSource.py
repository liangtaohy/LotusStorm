# -*- coding: utf-8 -*-
import json
import MLog


class DocJson:
    type = 'json'

    def __init__(self, filename, encoding='UTF-8'):
        self.file = filename
        self.load(encoding)
        MLog.logger.debug('empty')

    def load(self, encode):
        """
        load json string to python object
        :param encode:
        :return:
        """
        file_ = open(self.file)
        try:
            text = file_.read()
            obj = json.loads(text, encoding=encode)
        finally:
            file_.close()