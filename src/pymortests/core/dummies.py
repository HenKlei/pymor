# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import (BasicObject, abstractmethod)
from pymor.core.cache import CacheableObject, cached
from pymor.core import interfaces


class BoundaryInterface(interfaces.BasicObject):
    pass


class AllDirichletBoundaryInfo(BoundaryInterface):
    pass


class UnknownInterface(BasicObject):
    pass


class StupidInterface(BasicObject):
    """I am a stupid Interface"""

    @abstractmethod
    def shout(self, phrase, repeat):
        """ I repeatedly print a phrase.

        :param phrase: what I'm supposed to shout
        :param repeat: how often I'm shouting phrase
        :type phrase: str
        :type repeat: int

        .. seealso:: blabla
        .. warning:: blabla
        .. note:: blabla
        """
        pass


class BrilliantInterface(BasicObject):
    """I am a brilliant Interface"""

    @abstractmethod
    def whisper(self, phrase, repeat):
        """
        :type phrase: str
        :type repeat: int,=1
        """
        pass


class StupidImplementer(StupidInterface):

    def shout(self, phrase, repeat):
        print(phrase * repeat)


class AverageImplementer(StupidInterface, BrilliantInterface):

    def shout(self, phrase, repeat):
        # cannot change docstring here or else
        print(phrase * repeat)

    def whisper(self, phrase, repeat):
        print(phrase * repeat)

    some_attribute = 2


class CacheImplementer(CacheableObject):

    some_attribute = 2

    @cached
    def run(self):
        pass


class DocImplementer(AverageImplementer):
    """I got my own docstring"""

    def whisper(self, phrase, repeat):
        """my interface is stupid, I can whisper a lot more
        """
        self.logger.critical(phrase * repeat)


class FailImplementer(StupidInterface):
    pass


class BoringTestInterface(BasicObject):
    pass


class BoringTestClass(BasicObject):

    def validate_interface(self, cls, other):
        """If you want to contract check on a type defined in the same module you CANNOT use the absolute path
        notation. For classes defined elsewhere you MUST use it. Only builtins and classes with
        UberMeta as their metaclass can be checked w/o manually defining a new contract type.

        :type cls: pymortests.core.dummies.BoringTestInterface
        :type other: pymor.discretizers.builtin.grids.boundaryinfos.AllDirichletBoundaryInfo
        """
        pass

    def dirichletTest(self, dirichletA, dirichletB):
        """I'm used in testing whether contracts can distinguish
        between equally named classes in different modules

        :type dirichletA: pymor.discretizers.builtin.grids.interfaces.BoundaryInfo
        :type dirichletB: pymortests.core.dummies.AllDirichletBoundaryInfo
        """
        return dirichletA != dirichletB
