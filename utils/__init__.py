from .CorpusTfIDF import tfidf
def build_word_cloud(text):
    """
    build WordCloud Image
    :return:
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wordcloud = WordCloud(font_path='/System/Library/Fonts/PingFang.ttc').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


__all__ = ["build_word_cloud", "tfidf"]