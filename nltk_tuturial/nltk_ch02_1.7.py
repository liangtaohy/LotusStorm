# coding=utf-8
import nltk
from nltk.book import *

#ws = nltk.corpus.cess_esp.words()
#print(ws[:5])
#ws = nltk.corpus.floresta.words()
#print(ws[:5])
#ws = nltk.corpus.indian.words('hindi.pos')
#print(ws[:5])
#print(nltk.corpus.unhr.fileids())
#print(nltk.corpus.udhr.words('Javanese-Latin1')[:11])

from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1')
)
cfd.plot(cumulative=True)