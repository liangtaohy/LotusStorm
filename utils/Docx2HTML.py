import mammoth


def to_html(filename):
    with open(filename, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file)
        html = result.value  # The generated HTML
        return html


def to_text(filename):
    with open(filename, "rb") as docx_file:
        result = mammoth.extract_raw_text(docx_file)
        text = result.value
        return text


if __name__ == "__main__":
    f = "/Users/xlegal/PycharmProjects/LotusStorm/uploads/02_TS-_template-2018.docx"

    content = to_text(f)
    fp = open("1.txt", "w", encoding="utf-8")
    fp.write(content)
    fp.close()

    html = to_html(f)
    fp = open("2.html", "w", encoding="utf-8")
    fp.write(html)
    fp.close()