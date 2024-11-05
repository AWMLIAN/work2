import dlib
import cv2 as cv
import numpy as np


# dlib人脸特征点检测调用,返回值是68个检测点构成的一个列表
def character_point(image, save_path):
    detector = dlib.get_frontal_face_detector()  # 使用dlib库提供的人脸提取器
    predictor = dlib.shape_predictor(
        'shape_predictor_68_face_landmarks.dat')  # 构建特征提取器
    # 注意，这个dat文件要放在和你代码相同的路径下
    rects = detector(image, 1)  # rects表示人脸数 人脸检测矩形框4点坐标:左上角（x1,y1)、右下角（x2,y2）
    f = open(save_path, 'w+')
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
            # 将特征点写入txt文件中，方便后面使用
            f.write(str(point[0, 0]))
            f.write('\t')
            f.write(str(point[0, 1]))
            f.write('\n')

    f.write(str(0))
    f.write('\t')
    f.write(str(0))
    f.write('\n')  # 图像左上角的点

    f.write(str(0))
    f.write('\t')
    f.write(str(image.shape[0] // 2))
    f.write('\n')  # 左边界中间的点

    f.write(str(0))
    f.write('\t')
    f.write(str(image.shape[0] - 1))
    f.write('\n')  # 左下角的点

    f.write(str(image.shape[1] // 2))
    f.write('\t')
    f.write(str(0))
    f.write('\n')  # 上边界中间的点

    f.write(str(image.shape[1] // 2))
    f.write('\t')
    f.write(str(image.shape[0] - 1))
    f.write('\n')  # 下边界中间的点

    f.write(str(image.shape[1] - 1))
    f.write('\t')
    f.write(str(0))
    f.write('\n')  # 右上角的点

    f.write(str(image.shape[1] - 1))
    f.write('\t')
    f.write(str(image.shape[0] // 2))
    f.write('\n')  # 右边界中间的点

    f.write(str(image.shape[1] - 1))
    f.write('\t')
    f.write(str(image.shape[0] - 1))
    f.write('\n')  # 右下角的点

    points_lst.append((0, 0))  # 图像左上角的点
    points_lst.append((0, image.shape[0] // 2))  # 左边界中间的点
    points_lst.append((0, image.shape[0] - 1))  # 左下角的点
    points_lst.append((image.shape[1] // 2, 0))  # 上边界中间的点
    points_lst.append((image.shape[1] // 2, image.shape[0] - 1))  # 下边界中间的点
    points_lst.append((image.shape[1] - 1, 0))  # 右上角的点
    points_lst.append((image.shape[1] - 1, image.shape[0] // 2))  # 右边界中间的点
    points_lst.append((image.shape[1] - 1, image.shape[0] - 1))  # 右下角的点

    for i, item in enumerate(points_lst):
        pos = item
        # '''
        # 利用cv2.circle给每个特征点画一个圈，共68个
        cv.circle(image, pos, 2, color=(0, 255, 0))
        # 利用cv.putText输出1-68
        # 各参数依次是：图片，添加的文字，坐标，字体，字体大小，颜色，字体粗细
        cv.putText(image, str(i), pos, cv.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1, cv.LINE_AA)
        # cv.namedWindow("img", point_detect)
        cv.imshow("img", image)  # 显示图像
        cv.waitKey(100)  # 等待按键，随后退出
        # '''
    # cv.imwrite('telangpu_point_detect',image)

    f.close()
    return points_lst, image


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


# Draw delaunay triangles
def draw_delaunay(image, subdiv, delaunary_color, save_path, save_path_index):
    f1 = open(save_path_index, 'w')  # 创建text文本，存入三角形三个顶点的索引

    save_path_lst = []
    with open(save_path, 'r') as f:  # 打开存有76个关键点的文件，放入列表中
        for line in f:
            x, y = line.split()
            save_path_lst.append((int(x), int(y)))

    triangleList = subdiv.getTriangleList()
    for t in triangleList:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            cv.line(image, pt1, pt2, delaunary_color, 1, cv.LINE_AA, 0)
            cv.line(image, pt2, pt3, delaunary_color, 1, cv.LINE_AA, 0)
            cv.line(image, pt3, pt1, delaunary_color, 1, cv.LINE_AA, 0)

            for index, item in enumerate(save_path_lst):
                if pt1 == item:
                    f1.write(str(index))
                    f1.write('\t')
                elif pt2 == item:
                    f1.write(str(index))
                    f1.write('\t')
                elif pt3 == item:
                    f1.write(str(index))
                    f1.write('\t')
            f1.write('\n')
    f1.close()


#####################################
#           以下是主程序             #
#####################################
image = cv.imread("001.jpg")  # 读取图像
save_path1 = '001.txt'  # 将特征点保存入text文件中
point_list1, image1 = character_point(image, save_path1)  # dlib人脸特征点检测调用,返回值是68个检测点构成的一个列表

save_path2 = 'tri3_dilireba.txt'  # 创建text文本，存入三角形三个顶点的索引

size = image.shape  # 使用矩形定义要区分的空间
rect = (0, 0, size[1], size[0])

subdiv = cv.Subdiv2D(rect)  # 创建 Subdiv2D 的实例
delaunary_color = (0, 0, 255)  # 使用红色画三角形
image_origin = image.copy()  # 拷贝一份图像
animate = True  # 当画三角形的时候打开动画
for p in point_list1:
    subdiv.insert(p)

    # 显示动画
    if animate:
        img_copy = image_origin.copy()
        # Draw delaunay triangles
        draw_delaunay(img_copy, subdiv, (0, 0, 255), save_path1, save_path2);
        cv.imshow('win_delaunay', img_copy)
        k = cv.waitKey(100)

# if k==ord('s'):
# cv.destroyAllWindows()

draw_delaunay(image_origin, subdiv, delaunary_color, save_path1, save_path2);

# dim=(550,700)
# image_origin=cv.resize(image_origin,dim)  #更改图像尺寸

cv.imshow('win_delaunay', image_origin)
# cv.imshow('win_0',image)
k = cv.waitKey(0)
if k == ord('s'):
    cv.imwrite("after delaunary/'s image.jpg", image_origin)
    cv.imwrite("clinton point detect.jpg", image1)
    cv.destroyAllWindows()