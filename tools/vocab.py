"""
    :author Liang Tao (liangtaohy@163.com)

    从管道读入数据，按空格分隔，一行一个词，并输出到stdout
"""
import sys


def main():
    data = sys.argv[1:]
    if not sys.stdin.isatty():
        data.append(sys.stdin.read())
    return data


if __name__ == '__main__':
    data = main()
    t = []
    for d in data:
        t.extend(d.split(" "))
    t = [x.strip() for x in t]
    t = list(set(t))
    print('\n'.join(t))
