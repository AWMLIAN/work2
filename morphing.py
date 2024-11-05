import numpy as np
import cv2
import dlib


# Read points from text file 从text文件中读取点
def readPoints(path):
    # Create an array of points.
    points = [];
    # Read points
    with open(path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


########################################################################
def interpolation(image):
    '''
    定义插值函数
    :return:
    '''
    image_h = image.shape[0]
    image_w = image.shape[1]

    for j in range(image_h):
        for i in range(image_w):
            if image[j][i].all() == 0 and (j + 2) < image_h and (i + 2) < image_w:
                image[j][i] = image[j + 2][i + 2]

    return image


############################################################################################
# dlib人脸特征点检测调用,返回值是68个检测点构成的一个列表
def character_point(image, save_path):
    detector = dlib.get_frontal_face_detector()  # 使用dlib库提供的人脸提取器
    predictor = dlib.shape_predictor(save_path)  # 构建特征提取器
    rects = detector(image, 1)  # rects表示人脸数 人脸检测矩形框4点坐标:左上角（x1,y1)、右下角（x2,y2）
    points_lst = []
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(image, rects[i]).parts()])  # 人脸关键点识别 landmarks:获取68个关键点的坐标
        # shape=predictor(img,box) 功能：定位人脸关键点 img:一个numpy ndarray,包含8位灰度或RGB图像  box:开始内部形状预测的边界框
        # 返回值:68个关键点的位置
        for idx, point in enumerate(landmarks):  # enumerate函数遍历序列中的元素及它们的下标
            # 68点的坐标
            pos = (int(point[0, 0]), int(point[0, 1]))
            # print(idx,pos)
            points_lst.append(pos)

    points_lst.append((0, 0))  # 图像左上角的点
    points_lst.append((0, image.shape[0] // 2))  # 左边界中间的点
    points_lst.append((0, image.shape[0] - 1))  # 左下角的点
    points_lst.append((image.shape[1] // 2, 0))  # 上边界中间的点
    points_lst.append((image.shape[1] // 2, image.shape[0] - 1))  # 下边界中间的点
    points_lst.append((image.shape[1] - 1, 0))  # 右上角的点
    points_lst.append((image.shape[1] - 1, image.shape[0] // 2))  # 右边界中间的点
    points_lst.append((image.shape[1] - 1, image.shape[0] - 1))  # 右下角的点

    return points_lst


########################################################################
# delaunay三角划分
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def get_delaunary(img, point_list):
    size = img.shape
    rect = (0, 0, size[1], size[0])  # 使用矩形定义要区分的空间
    subdiv = cv2.Subdiv2D(rect)  # 创建 Subdiv2D 的实例
    for p in point_list:
        subdiv.insert(p)
    lst = []  # 创建列表，存入三角形三个顶点的索引

    triangleList = subdiv.getTriangleList()
    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        # print(pt1)
        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            # cv2.line(image, pt1, pt2, delaunary_color, 1, cv2.LINE_AA, 0)
            # cv2.line(image, pt2, pt3, delaunary_color, 1, cv2.LINE_AA, 0)
            # cv2.line(image, pt3, pt1, delaunary_color, 1, cv2.LINE_AA, 0)

            lst_1 = []
            for index, item in enumerate(point_list):

                if pt1 == item:
                    lst_1.append(index)
                elif pt2 == item:
                    lst_1.append(index)
                elif pt3 == item:
                    lst_1.append(index)
                # print(item)
                # print(lst_1)
            lst.append(lst_1)

    return lst


########################################################################
# 获得仿射变换矩阵
def getAffineTransform(srctri, dsttri):
    srctri = np.float32(srctri)
    dsttri = np.float32(dsttri)

    srctri1 = []
    for i in range(len(srctri)):
        src = srctri[i]
        src = list(src)
        srctri1.append(src)
    A = np.reshape(np.array(srctri1), (3, 2))  # 求出基于原始多边形顶点的坐标所组成的矩阵

    dsttri1 = []
    for i in range(len(dsttri)):
        dst = dsttri[i]
        dst = list(dst)
        dst.append(1)  # 得出[x,y,1]的形式
        dsttri1.append(dst)
    B = np.reshape(np.array(dsttri1), (3, 3))  # 求出基于目标多边形顶点的坐标所组成的矩阵

    MT = np.linalg.pinv(B).dot(A)
    M = MT.T

    return M


# 计算三角形面积
def is_trangle_area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))


# 将仿射变换矩阵应用到图像中
def warpAffine(src, warpMat, size, dstTri):
    dst = np.zeros((size[1], size[0], 3), dtype=src.dtype)
    x1 = dstTri[0][0]
    y1 = dstTri[0][1]
    x2 = dstTri[1][0]
    y2 = dstTri[1][1]
    x3 = dstTri[2][0]
    y3 = dstTri[2][1]
    abc = is_trangle_area(x1, y1, x2, y2, x3, y3)
    # k=0
    for j in range(dst.shape[0]):
        for i in range(dst.shape[1]):
            abp = is_trangle_area(x1, y1, x2, y2, i, j)
            acp = is_trangle_area(x1, y1, x3, y3, i, j)
            bcp = is_trangle_area(x2, y2, x3, y3, i, j)
            if abc == abp + acp + bcp:
                lst = []
                lst.append(i)
                lst.append(j)
                lst.append(1)
                A = np.reshape(np.array(lst), (3, 1))  # 将列表转换为矩阵（3*1）
                B = np.dot(warpMat, A)  # 应用仿射变换矩阵，得到仿射后的坐标

                lst_dst = []
                lst1 = list(B)
                for index in range(len(lst1)):
                    lst_dst.append(lst1[index][0])

                # print(dst.shape)
                # print(lst_dst)
                # dst[j][i]=src[(int(lst_dst[1])-1)][int(lst_dst[0])-1]
                # '''
                x = lst_dst[0] - 1
                y = lst_dst[1] - 1
                m = int(y)
                n = int(x)
                u = y - m
                v = x - n
                dst[j][i] = (1 - u) * (1 - v) * src[m][n] + (1 - u) * v * src[m][n + 1] + u * (1 - v) * src[m + 1][
                    n] + u * v * src[m + 1][n + 1]
                # '''

    # dst=cv2.resize(dst,(size[0],size[1]))
    return dst


# def applyAffineTransform(src, srcTri, dstTri, size):
#     warpMat = getAffineTransform(srcTri, dstTri)
#     # warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
#     # M = cv2.GetAffineTransform(src, dst)  src:原始图像中的三个点的坐标 dst:变换后的这三个点对应的坐标；M:根据三个对应点求出的仿射变换矩阵（2*3）
#
#     dst = warpAffine(src, warpMat, size, dstTri)
#     # dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
#     # cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
#     # src:输入图像 M：变换矩阵 dsize：输出图像的大小 flags：插值方法的组合（int类型） borderMode：边界像素模式（int类型） borderValue：边界填充值；默认情况下，它为0
#     return dst
def applyAffineTransform(src, srcTri, dstTri, size):
    # 使用cv2的getAffineTransform直接获取仿射变换矩阵
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # 使用warpAffine进行仿射变换
    dst = cv2.warpAffine(src, warpMat, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


########################################################################################
def morphTriangle(img1, img2, img_1, img_2, t1, t2, t, alpha):
    # 为每个三角形寻找边界矩形
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    # r=cv2.boundingRect(cnt) cnt:一个轮廓点集合；r:返回值,x,y(矩阵左上点的坐标)；w(矩阵的宽),h(矩阵的高)

    # 各个矩形左上角点的偏移点
    t1Rect = []
    t2Rect = []
    tRect = []  # [(),(),()]

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))  # 三角形每个顶点与相应矩形左上角顶点的偏移
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # 通过填充三角形来获取掩膜
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)  # 3是3个通道的意思
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);
    # cv2.fillConvexPoly( image , 多边形顶点array , RGB color)

    # 把扭曲图像应用到小的矩形块
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]  # 裁剪矩形块

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)  # 对裁剪的矩形块进行扭曲操作
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha 混合矩形块
    # imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # 复制矩形块的三角形区域到到输出图像
    img_1[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img_1[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (
                1 - mask) + warpImage1 * mask
    img_2[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img_2[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (
                1 - mask) + warpImage2 * mask


############################################################################################
def morphing(alpha, img1, img2, points1, points2, lst, imgMorph1, imgMorph2):
    # 将图像转换为浮点数据类型
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    #  读取对应点的数组
    points = [];
    # 计算加权平均点坐标
    for i in range(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x, y))

    # 从列表中读取三角形，列表中存放的是delaunay三角划分后的每个三角形的顶点的索引
    for i in range(len(lst)):
        a = lst[i][0]
        b = lst[i][1]
        c = lst[i][2]
        # 获得的t为三角形的三个顶点坐标
        t1 = [points1[a], points1[b], points1[c]]  # x,y,z分别为delaunay三角划分后对应的每个三角形的点的索引（顺序）
        t2 = [points2[a], points2[b], points2[c]]
        t = [points[a], points[b], points[c]]

        # 一次变形一个三角形
        morphTriangle(img1, img2, imgMorph1, imgMorph2, t1, t2, t, alpha)

    return imgMorph1, imgMorph2


#######################################################################################
def morphing_save(save_path_index):
    lst = np.linspace(0, 1, num=11)
    for i, item in enumerate(lst):
        alpha = item
        print("进入save函数")
        print(alpha)
        # beta=lst[i+1]
        imgM1, imgM2 = morphing(alpha, img1, img2, point_76_lst_ori, point_76_lst_refer, save_path_index, imgMorph1, imgMorph2)
        imgM1 = interpolation(np.uint8(imgM1))
        imgM2 = interpolation(np.uint8(imgM2))  # 对图像进行插值处理，去除黑线
        imgMorph = (1.0 - alpha) * imgM1 + alpha * imgM2
        # save_path= f"D:\\software\\pyCharm\\pythonProject\\pythonProject1\\img\\{i + 1}.jpg"
        cv2.imwrite("D:\\software\\pyCharm\\pythonProject\\pythonProject1\\img\\img7\\{}.jpg".format(str(i + 1)), imgMorph)
        # cv2.imwrite(save_path, imgMorph)
        print("成功推出save函数")
        # cv2.waitKey(50)


###########################################################################################################
if __name__ == '__main__':
    # Read images 读取图像
    img1 = cv2.imread('menglei.jpg');
    img2 = cv2.imread('jcelian.jpg');
    # 读取坐标点
    # points1 = readPoints('F:\\BaiduNetdiskDownload\\pictures\\renlian\\telangpu1.txt')    #points1为列表，里面存放的是获取的关键点的坐标[(),()...]
    # points2 = readPoints('F:\\BaiduNetdiskDownload\\pictures\\renlian\\clinton1.txt')
    save_path_68 = 'shape_predictor_68_face_landmarks.dat'
    point_76_lst_ori = character_point(img1, save_path_68)
    point_76_lst_refer = character_point(img2, save_path_68)  # 获取76个坐标点
    # save_path_index='tri2.txt'   #从tri.txt文件中读取三角形，tri.txt中存放的是delaunay三角划分后的每个三角形的顶点的索引
    lst_index = get_delaunary(img1, point_76_lst_ori)  # 获取每个三角形顶点在point_76_lst_ori中的索引
    # 为最终输出分配空间
    imgMorph1 = np.zeros(img1.shape, dtype=np.float32)
    imgMorph2 = np.zeros(img2.shape, dtype=np.float32)

    morphing_save(lst_index)  # 保存每个alptha值对应的morphing后的图片，前面有定义
