# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:M兴M
@Blog(个人博客地址): https://blog.csdn.net/MbingxingM?spm=1000.2115.3001.5343

@File:creategif.py
@Time:2022/4/29 22:21

@Motto:不积跬步无以至千里，不积小流无以成江海，程序人生的精彩需要坚持不懈地积累！
"""
import imageio
from pathlib import Path


def imgs2gif(imgPaths, saveName, duration=None, loop=0, fps=None):
    """
    生成动态图片 格式为 gif
    :param imgPaths: 一系列图片路径
    :param saveName: 保存gif的名字
    :param duration: gif每帧间隔， 单位 秒
    :param fps: 帧率
    :param loop: 播放次数（在不同的播放器上有所区别）， 0代表循环播放
    :return:
    """
    if fps:
        duration = 1 / fps
    images = [imageio.v2.imread(str(img_path)) for img_path in imgPaths]
    imageio.mimsave(saveName, images, "gif", duration=duration, loop=loop)

pathlist = Path(r"D:\\software\\pyCharm\\pythonProject\\pythonProject1\\img\\img7\\").glob("*.jpg")

p_lis = []
for n, p in enumerate(pathlist):
    if n % 1 == 0:  # 间隔几张图片显示
        p_lis.append(p)

imgs2gif(p_lis, "005.gif", 0.033 * 11, 0)
