from PIL import Image
from skimage import io, filters, feature
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage.transform as transform


class ImageHis:
    def __init__(self, image_file):
        self.image_file = image_file
        self.img = None
        self.nparr = None

    def filters(self, filter='sobel'):
        self.img = Image.open(self.image_file).convert("L")
        self.nparr = np.array(self.img)
        edges = filters.sobel_h(self.nparr)
        plt.imshow(edges, plt.cm.gray)
        plt.show()

    def hough(self):
        # hough线变换
        self.img = Image.open(self.image_file).convert("L")
        self.nparr = np.array(self.img)

        edges = feature.canny(self.nparr, sigma=4)

        h, theta, d = transform.hough_line(edges)
        # 生成一个一行两列的窗口（可显示两张图片）.
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 6))
        plt.tight_layout()

        # 显示原始图片
        ax0.imshow(self.img, plt.cm.gray)
        ax0.set_title('原始图像')
        ax0.set_axis_off()

        # 显示hough变换所得数据
        ax1.imshow(np.log(1 + h))
        ax1.set_title('Hough变换')
        ax1.set_xlabel('Angles (degrees)')
        ax1.set_ylabel('Distance (pixels)')
        ax1.axis('image')

        ax2.imshow(self.img, plt.cm.gray)
        row1, col1 = self.nparr.shape
        for _, angle, dist in zip(*transform.hough_line_peaks(h, theta, d)):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
            print(0, col1, y0,y1)

            ax2.annotate("%f%f"%(y0,y1), xy=(y0,y1))
            ax2.plot((0, col1), (y0, y1), '-r')
        ax2.axis((0, col1, row1, 0))
        ax2.set_title('Detected lines')
        ax2.set_axis_off()

        plt.show()

    def edge_canny(self):
        self.img = Image.open(self.image_file).convert("L")
        self.nparr = np.array(self.img)

        edges1 = feature.canny(self.nparr)

        edges2 = feature.canny(self.nparr, sigma=4)

        io.imshow(edges2)
        io.show()
        #plt.imshow(edges2, plt.cm.gray)
        #plt.show()

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
        img = np.array(Image.open(self.image_file))

        plt.figure("直方图")
        arr = img.flatten()

        plt.subplot(221)
        plt.imshow(img, plt.cm.gray)  # 原始图像
        plt.subplot(222)
        n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor='green', alpha=0.75)

        plt.show()

if __name__ == '__main__':
    his = ImageHis(image_file = '/Users/xlegal/Desktop/book/最新企业人力资源速查速用全书第二版/page_88.jpg')
    #his.edge_canny()
    #his.filters()
    #his.show_hist_L()
    #his.to_L()
    #his.binaries()
    #his.image_L()
    his.hough()