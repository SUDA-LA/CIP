from bs4 import BeautifulSoup


def parse_html(html_file_path, encoding=None):
    html_file = open(html_file_path, encoding=encoding)
    soup = BeautifulSoup(html_file, 'html.parser')
    title = soup.title.string
    raw_contents = [content for content in soup.body.strings]
    contents = []
    rep = True
    for c in raw_contents:
        if c == '\n':
            if not rep:
                contents.append(c)
                rep = True
        else:
            rep = False
            contents.append(c.strip())
            if c[-1] == '\n':
                contents.append('\n')
                rep = True
    if contents[-1] == '\n':
        contents.pop()
    anchors = [(a.string, a['href']) for a in soup.find_all('a')]
    return title, contents, anchors


def formal_output(title, contents, anchors):
    out = "title:\n"
    out += str(title) + '\n'
    out += "body:" + '\n'
    out += ''.join(contents) + '\n'
    out += "link:" + '\n'
    out += '\n'.join([f'{a[0]} {a[1]}' for a in anchors])
    return out


if __name__ == '__main__':
    title, contents, anchors = parse_html('./data/1.html', encoding='UTF-8')
    print(formal_output(title, contents, anchors))
    print()
    title, contents, anchors = parse_html('./data/2.html', encoding='UTF-8')
    print(formal_output(title, contents, anchors))
