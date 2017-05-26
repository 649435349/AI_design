# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
sys.path.append('../../')

class Singleton(object):
    @classmethod
    def instance(cls):
        inst = getattr(cls, "_instance", None)
        if inst is None:
            inst = cls._instance = cls()
        return inst