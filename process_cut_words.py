# -*- coding: utf-8 -*-
from optparse import OptionParser
import os
import jieba


def get_opt_parser():
    parser = OptionParser(usage=__doc__)
    parser.add_option("-s", "--source", action="store", default=False,type="string",
                      dest="path", help="Source Data Dir")

    parser.add_option("-d", "--dest",
                      action="store", default=False, dest="dest",type="string",
                      help="Dest File Path")

    parser.add_option("-u", "--userdict",
                      action="store", default='', dest="userdict", type="string",
                      help="User Defined Dict")

    parser.add_option("-b", "--treebuilder", action="store", type="string",
                      dest="treebuilder", default="etree")
    return parser


if __name__ == "__main__":
    opt_p = get_opt_parser()
    opts, args = opt_p.parse_args()

    if os.path.exists(opts.userdict):
        jieba.load_userdict(opts.userdict)

    f = open(opts.path, 'r')

    if f:
        buf = f.read()
        f.close()
        seg_list = jieba.cut(buf)
        buf = " ".join(seg_list)
        dest = open(opts.dest, "w")
        dest.write(buf)
        dest.close()
