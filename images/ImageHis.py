from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class ImageHis:
    def __init__(self, image_file):
        self.image_file = image_file
        self.img = None
        self.nparr = None

    def image_L(self):
        self.img = Image.open(self.image_file).convert("L")
        self.nparr = np.array(self.img)

        print(self.nparr)

        print(self.nparr.flatten())

        one_arr = list(self.nparr.flatten())

        median = self.nparr.mean()

        threshold = median

        print(threshold)

        table = []

        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(1)

        bimg = self.img.point(table, "1")
        bimg.save(self.image_file + "_mean.jpeg", "JPEG")

        "求出图象的最大灰度值和最小灰度值，分别记为gl和gu，初始阈值"
        g1 = self.nparr.min()
        gu = self.nparr.max()
        t0 = int((g1 + gu) / 2)

        K = 1
        Tk = 0
        while K <= 256:
            Ab = 0
            Af = 0
            Cb = 0
            Cf = 0

            for g in one_arr:
                if g >= g1 and g <= t0:
                    Ab = Ab + g
                    Cb += 1
                elif g >= t0 + 1 and g <= gu:
                    Af = Af + g
                    Cf += 1

            Ab = Ab / Cb
            Af = Af / Cf
            Tk = int(Ab + Af) / 2

            if Tk - t0 > 1:
                t0 = Tk
                K += 1
            else:
                break
        print("The best threshold is: %d" % Tk)
        self.binary_by_threshold(Tk)

    def binary_by_threshold(self, threshold):
        table = []

        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(1)

        bimg = self.img.point(table, "1")
        bimg.save(self.image_file + "_bin.jpeg", "JPEG")

    def ostu(self):
        """
        1979年 日本大津提出
        思想：取某个预置，例得前景和背景两类的类间方差最大
        :return:
        """
    def binaries(self):
        img = Image.open(self.image_file).convert("1")
        img.save(self.image_file + "_bin.jpeg", "JPEG")

    def to_L(self):
        img = Image.open(self.image_file).convert('L')
        img.save(self.image_file + "_L.jpeg", "JPEG")

    def show_hist_L(self):
        img = np.array(Image.open(self.image_file).convert('L'))

        plt.figure("直方图")
        arr = img.flatten()

        n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor='green', alpha=0.75)

        plt.show()

if __name__ == '__main__':
    his = ImageHis(image_file = '/Users/xlegal/Desktop/test.jpeg')
    his.show_hist_L()
    his.to_L()
    #his.binaries()
    his.image_L()