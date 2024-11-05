import cv2
import numpy as np
# import matplotlib.pyplot as plt
import dlib


#################################################################
# 定义读取坐标点函数
def readpoints(path):
    points = []
    with open(path, 'r') as f:
        for line in f:
            x, y = line.split()
            x = int(x)
            y = int(y)
            points.append((x, y))
    return points


#######################################################################
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


######################################################################
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

            lst.append(lst_1)

    return lst


#######################################################################################
# 仿射变换函数求取变换矩阵以及变换矩阵的应用
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

                x = lst_dst[0] - 1
                y = lst_dst[1] - 1
                m = int(y)
                n = int(x)
                u = y - m
                v = x - n
                dst[j][i] = (1 - u) * (1 - v) * src[m][n] + (1 - u) * v * src[m][n + 1] + u * (1 - v) * src[m + 1][
                    n] + u * v * src[m + 1][n + 1]
                # '''

    return dst


def applyAffineTransform(src, srcTri, dstTri, size):
    # 获取仿射变换矩阵M
    M = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    # M=getAffineTransform(srcTri,dstTri)
    # 将仿射变换应用到图像块中
    dst = cv2.warpAffine(src, M, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # dst=warpAffine(src,M,size,dstTri)

    return dst


#####################################################################
# 定义warp函数来对图像块进行形变处理
def warpImage(srcTri, dstTri, img1, img):
    # 为每个三角形寻找边界矩形
    r1 = cv2.boundingRect(np.float32(srcTri))
    r = cv2.boundingRect(np.float32(dstTri))

    # 每个三角形三个顶点与矩形左上角顶点的偏移
    t1Rect = []
    tRect = []

    for i in range(0, 3):
        t1Rect.append(((srcTri[i][0] - r1[0]), (srcTri[i][1] - r1[1])))
        tRect.append(((dstTri[i][0] - r[0]), (dstTri[i][1] - r[1])))

    # 为r设置掩膜
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # 裁剪矩形块
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # imgRect=img[r[1]:r[1]+r[3],r[0]:r[0]+r[2]]

    # 对img1Rect进行warpping操作
    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)

    # dim=(size)
    # img2Rect=cv2.resize(img1Rect,dim)
    # imgmix=(1.0 - alpha) * img2Rect + alpha * warpImage1

    # 把warpping后的warpImage1复制到输出图像
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + warpImage1 * mask
    # imgRect=imgRect * (1-mask) + warpImage1

    # return imgRect


######################################################################################
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


##########################################################################################

if __name__ == '__main__':

    # 设置alpha值
    alpha = 0.5

    # 读取图像
    img_ori = cv2.imread('zhang1.jpg')
    img_refer = cv2.imread('zhang3.jpg')
    # 将img图像转为浮点数据类型
    img = np.float32(img_ori)

    # 读取坐标点
    save_path_68 = 'shape_predictor_68_face_landmarks.dat'
    point_76_lst_ori = character_point(img_ori, save_path_68)
    point_76_lst_refer = character_point(img_refer, save_path_68)  # 获取76个坐标点

    # points1=readpoints('telangpu1.txt')  #readpoints函数在前面有定义
    # points2=readpoints('clinton1.txt')
    points = []
    # '''
    for i in range(0, len(point_76_lst_ori)):
        x = alpha * point_76_lst_ori[i][0] + (1 - alpha) * point_76_lst_refer[i][0]
        y = alpha * point_76_lst_ori[i][1] + (1 - alpha) * point_76_lst_refer[i][1]
        points.append((int(x), int(y)))  # 获取目标图像的76个坐标点
    # '''
    lst = get_delaunary(img_ori, point_76_lst_ori)  # 获取每个三角形顶点在point_76_lst_ori中的索引

    # 为最终输出分配空间
    warppingImage = np.zeros(img.shape, dtype=img.dtype)

    # '''
    for item in lst:
        x = int(item[0])
        y = int(item[1])
        z = int(item[2])
        t1 = [point_76_lst_ori[x], point_76_lst_ori[y], point_76_lst_ori[z]]
        t = [points[x], points[y], points[z]]
        warpImage(t1, t, img, warppingImage)  # 对每个小块进行warp操作
    # '''

    # warppingImage=interpolation(np.uint8(warppingImage))   #对图像进行插值，去除黑线，函数在上面有定义
    cv2.imshow('after warpping \'s image', np.uint8(warppingImage))
    cv2.imshow('img_ori', img_ori)
    k = cv2.waitKey(0)
    if k == ord('s'):
        # cv2.imwrite('final.jpg', np.uint8(warppingImage))
        cv2.destroyAllWindows()