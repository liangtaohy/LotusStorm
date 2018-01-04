from wand.image import Image
import os


def convert_pdf_to_jpg(file_name):
    with Image(filename=file_name) as img:
        print('pages = ', len(img.sequence))

    with img.convert('jpeg') as converted:
        converted.save(filename=file_name + '.jpeg')

convert_pdf_to_jpg('/Users/xlegal/Desktop/DOC180104-20180104141457.pdf')