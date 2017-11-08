#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division

from nltk import compat

import re, pprint

from nltk.compat import Counter

_NINF = float('-1e300')

@compat.python_2_unicode_compatible
class DrawerMatplot(Counter):
    def __init__(self, samples=None):
        """
        Construct a new frequency distribution.  If ``samples`` is
        given, then the frequency distribution will be initialized
        with the count of each object in ``samples``; otherwise, it
        will be initialized to be empty.

        In particular, ``FreqDist()`` returns an empty frequency
        distribution; and ``FreqDist(samples)`` first creates an empty
        frequency distribution, and then calls ``update`` with the
        list ``samples``.

        :param samples: The samples to initialize the frequency
            distribution with.
        :type samples: Sequence
        """
        Counter.__init__(self, samples)

    def _cumulative_frequencies(self, samples):
        """
        Return the cumulative frequencies of the specified samples.
        If no samples are specified, all counts are returned, starting
        with the largest.

        :param samples: the samples whose frequencies should be returned.
        :type samples: any
        :rtype: list(float)
        """
        cf = 0.0
        for sample in samples:
            cf += self[sample]
            yield cf


    def plot(self, most_common):
        """
        Plot samples from the frequency distribution
        displaying the most frequent sample first.  If an integer
        parameter is supplied, stop after this many samples have been
        plotted.  For a cumulative plot, specify cumulative=True.
        (Requires Matplotlib to be installed.)

        :param title: The title for the graph
        :type title: str
        :param cumulative: A flag to specify whether the plot is cumulative (default = False)
        :type title: bool
        """
        try:
            from matplotlib import pylab
            from matplotlib.font_manager import FontProperties
            font_set = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc")
            samples = [item for item, _ in most_common]

            freqs = list(self._cumulative_frequencies(samples))
            ylabel = "Cumulative Counts"
            # percents = [f * 100 for f in freqs]  only in ProbDist?
            print(freqs)
            pylab.grid(True, color="silver")
            pylab.plot(freqs)
            pylab.xticks(range(len(samples)), [compat.text_type(s) for s in samples], rotation=90)
            pylab.xlabel(u"采样", fontproperties=font_set)
            pylab.ylabel(ylabel, fontproperties=font_set)
            pylab.show()
        except ImportError:
            raise ValueError('The plot function requires matplotlib to be installed.'
                         'See http://matplotlib.org/')