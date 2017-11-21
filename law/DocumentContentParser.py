#coding=utf-8
import re
from law.NumberDict import *
from law.JiebaTokenizer import JiebaTokenizer
from law import GovDict

debug = False


class DocContentParser:
    def __init__(self, content=''):
        self.v = '0.0.1'
        self.content = content

    def parse_content(self):
        tokens = JiebaTokenizer().tokenize(self.content)
        return tokens

    def find_flags(self, tokens, flags, start):
        i = 0
        total = len(flags)
        j = start
        match = False
        for (_, flag) in tokens[start:]:
            if flag == flags[i] and i < total:
                i += 1
            else:
                i = 0

            if i == total:
                match = True
                break
            j += 1
        if match:
            return j - total + 1
        return -1

    def is_number(self, str):
        try:
            return int(str)
        except ValueError:
            pass
        return False

    @staticmethod
    def title_normalize(title):
        result, number = re.subn(r'\(.*\)|（.*）', '', title)
        return result

    @staticmethod
    def is_skip_title(title):
        p = re.compile("（摘录）|（已失效）")
        match = p.search(title.strip())
        if match:
            return True
        return False

    def guowuyuanling_normalize(self):
        p = re.compile('第　([0-9]+)　号')
        match = p.search(self.content)
        if match:
            return '国令第%s号' % (match.group(1))
        return ''

    def date_normalization(self):
        result, number = re.subn('０', '0', self.content)
        result, number = re.subn('１', '1', result)
        result, number = re.subn('２', '2', result)
        result, number = re.subn('３', '3', result)
        result, number = re.subn('４', '4', result)
        result, number = re.subn('５', '5', result)
        result, number = re.subn('６', '6', result)
        result, number = re.subn('７', '7', result)
        result, number = re.subn('８', '8', result)
        self.content, number = re.subn('９', '9', result)

    def parse_time_1(self, tokens):
        """v t [x] m{1,6} x | n t [x] m{1,6} x"""
        total = len(tokens)
        state = 'T_start'

        i = 0

        verb = ''
        date = ''
        times = []
        while i < total:
            w, flag = tokens[i]
            if state == 'T_start':
                verb = ''
                date = ''
                if flag == 'n':
                    state = 'T_n'
                    verb = w
                elif flag == 'v':
                    state = 'T_v'
                    verb = w
            elif state == 'T_n':
                if flag == 't' or w == '时间':
                    state = 'T_nt'
                else:
                    state = 'T_start'
            elif state == 'T_v':
                if flag == 't' or w == '时间':
                    state = 'T_vt'
                else:
                    state = 'T_start'
            elif state == 'T_nt':
                if flag == 'm':
                    date += w
                    state = 'T_m'
                elif flag == 'x':
                    state = 'T_nt'
                else:
                    state = 'T_start'
            elif state == 'T_vt':
                if flag == 'm':
                    date += w
                    state = 'T_m'
                elif flag == 'x':
                    state = 'T_vt'
                else:
                    state = 'T_start'
            elif state == 'T_m':
                if flag == 'm':
                    date += w
                    state = 'T_m'
                elif flag == 'x':
                    state = 'T_end'
            elif state == 'T_end':
                state = 'T_start'
                if len(verb) > 0 and len(date) > 0:
                    p = re.compile(r'([0-9]{4}).([0-9]{1,2}).([0-9]{1,2})')
                    match = p.search(date)
                    if match and len(match.groups()) == 3:
                        try:
                            Y = int(match.groups()[0])
                            M = int(match.groups()[1])
                            D = int(match.groups()[2])
                            times.append({'verb': verb, 'date': '%d%02d%02d' % (Y, M, D)})
                        except ValueError:
                            pass
            i += 1
        return times

    def parse_time(self):
        tokens = None
        self.date_normalization()
        pattern1 = re.compile(r"自([0-9一二三四五六七八九十○零１２３４５６７８９０]{4})年([0-9一二三四五六七八九十１２３４５６７８９０]{1,2})月([0-9一二三四五六七八九十１２３４５６７８９０]{1,3})日起施行")
        pattern2 = re.compile(r"[公发]布，自(.*)?起施行")
        pattern2_1 = re.compile(r"现公布.*?，自(.*)?起施行")
        pattern3 = re.compile(r"([0-9一二三四五六七八九十○零１２３４５６７８９０]{4})年([0-9一二三四五六七八九十１２３４５６７８９０]{1,2})月([0-9一二三四五六七八九十１２３４５６７８９０]{1,3})日") #二○○九年七月二十日

        match = pattern1.search(self.content)

        valid_time = ''
        publish_time = ''
        author = ''

        if match and len(match.groups()) >= 3:
            if debug:
                print(match.groups())
            year = han_to_num(match.groups()[0])
            month = han_to_num(match.groups()[1])
            day = han_to_num(match.groups()[2])
            valid_time = "%s%02d%02d" % (year, int(month), int(day))

        match2 = pattern2.search(self.content)

        if debug:
            print(match2)
        if not match2:
            match2 = pattern2_1.search(self.content)
        if match2:
            t, d = match2.span()
            match3 = pattern3.search(self.content[d:d+150])
            if debug:
                print(match3)
            if match3 and len(match3.groups()) >= 3:
                if match3.groups()[0][0] in NumberDict:
                    if debug:
                        print(match3.groups())
                    year = han_to_num(match3.groups()[0])
                    month = han_to_num(match3.groups()[1])
                    day = han_to_num(match3.groups()[2])
                    publish_time = "%s%02d%02d" % (year, int(month), int(day))
                else:
                    publish_time = "%s%02d%02d" % (match3.groups()[0], int(match3.groups()[1]), int(match3.groups()[2]))
            if match2.group(1) == "公布之日" or match2.group(1) == "发布之日" or match2.group(1) == "颁布之日":
                valid_time = publish_time

        if len(publish_time) == 0:
            tokens = self.parse_content()
            tokens = tokens[:300]

            pos = 0
            p_times = []
            nts = []

            total_tokens = len(tokens)

            if debug:
                print(tokens)
            while pos < total_tokens:
                if debug:
                    print("pos: %d" % pos)
                date_pos = self.find_flags(tokens, ['m', 'm', 'm', 'm', 'm'], pos)
                if debug:
                    print("date_pos: %d" % date_pos)
                if date_pos == -1:
                    break

                nt_pos = self.find_flags(tokens, ['nt'], date_pos + 6)

                if nt_pos != -1 and nt_pos - date_pos - 6 <= 6:
                    wnt, _ = tokens[nt_pos]
                    nts.append(wnt)

                i = 1
                end_pos = pos
                offset = nt_pos
                if offset == -1:
                    offset = pos

                while i <= 20 and offset + i < total_tokens:
                    w, f = tokens[offset + i]
                    if w == ' ' or w == '\u3000' or w == '。' or w == ';' or w == '；' or w == '.' or w == '\n' or w == '，' or w == '\xa0':
                        end_pos = offset + i
                        break
                    i += 1

                while end_pos < total_tokens:
                    w, f = tokens[end_pos]
                    if w == '\n' or w == '\xa0':
                        end_pos += 1
                    else:
                        break

                if debug:
                    print("end_pos: %d, nt_pos: %d, pos: %d" % (end_pos, nt_pos, pos))
                    print(tokens[date_pos:])
                if end_pos:
                    y, _ = tokens[date_pos]
                    m, _ = tokens[date_pos + 2]
                    d, _ = tokens[date_pos + 4]
                    m = self.is_number(m)
                    d = self.is_number(d)

                    if m is False or d is False:
                        break

                    p_times.append("%s%02d%02d" % (y, int(m), int(d)))
                    pos = end_pos + 1
                else:
                    break

            if debug:
                print(p_times)
            if len(p_times):
                publish_time = p_times[0]
            if len(nts):
                author = nts[0]

        if len(valid_time) == 0:
            pattern4 = re.compile(r"本.*?自[公|发]布之日起施行。")
            match = pattern4.search(self.content)
            if match:
                valid_time = publish_time

        chengwen = ''

        if len(publish_time) == 0 or len(valid_time) == 0:
            if tokens is None:
                tokens = self.parse_content()

            times = self.parse_time_1(tokens)
            if len(times) > 0:
                for t in times:
                    if t['verb'] in ['成文']:
                        chengwen = t['date']
                    elif t['verb'] in ['发布', '颁布']:
                        publish_time = t['date']
                    if t['verb'] in ['实施', '施行', '生效']:
                        valid_time = t['date']

        if len(chengwen) > 0:
            publish_time = chengwen

        return {
            'publish_time': publish_time,
            'valid_time': valid_time,
            'author': author
        }

    def parse_author(self):
        tokens = JiebaTokenizer().tokenize(self.content)
        max = 100
        num = 1
        for (word, flag) in tokens:
            if flag != 'x':
                num += 1
            if num > max:
                return ''
            if flag == 'nt':
                if GovDict.is_gov(word):
                    return word
        return ''

if __name__ == '__main__':
    content = open("./test.txt").read()
    parser = DocContentParser(content)
    print(parser.parse_time())
    print(parser.parse_author())
    parser.parse_content()

    x = parser.is_number('号')
    print(x)
    x = parser.is_number('10')
    print(x)
    print(parser.guowuyuanling_normalize())
