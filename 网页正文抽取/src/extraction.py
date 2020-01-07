from lxml import etree
import sys
import re


def get_content(infile, outfile):
    f = open(infile)
    fout = open(outfile, 'w', encoding='utf-8')
    content = f.read()
    root = etree.HTML(content)

    title = root.xpath('//title')
    fout.write('title:\n')
    fout.write(title[0].text + '\n')

    body = root.xpath('//body')
    fout.write('body:\n')
    global body_content
    body_content = ''
    get_body(body[0])
    body_content = delete_enter(body_content)
    fout.write(body_content + '\n')

    href = root.xpath('//a')
    fout.write('link:\n')
    for node in href:
        fout.write(node.text + "\t" + node.attrib['href'] + '\n')


def get_body(elemtree):
    global body_content
    if elemtree.tag != 'script':
        if elemtree.text != None:
            body_content += elemtree.text
        if elemtree.tail != None:
            body_content += elemtree.tail

    for i in range(len(elemtree)):
        get_body(elemtree[i])


def delete_enter(string):
    return re.sub('\n+\s*\n+\s*', '\n', string)


if __name__ == '__main__':
    get_content('./data/1.html', './data/out1.txt')
    get_content('./data/2.html', './data/out2.txt')
