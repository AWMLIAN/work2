"""
import cv2
#获取一张图片的宽高作为视频的宽高
image=cv2.imread('E:/face_morphing/mo/0.jpg')
image_info=image.shape
height=image_info[0]
width=image_info[1]
size=(height,width)
fps=10
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter('E:\face_morphing\mo\001.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height)) #创建视频流对象-格式一
for i in range(0,101,5):
    file_name = "E:/face_morphing/mo/" + str(i) +".jpg "
    image=cv2.imread(file_name)
    video.write(image)  # 向视频文件写入一帧--只有图像，没有声音
cv2.waitKey()
#video = cv2.VideoWriter('E:\face_morphing\morphing_video\001.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width,height)) #创建视频流对象-格式二
参数1 即将保存的文件路径
参数2 VideoWriter_fourcc为视频编解码器
    fourcc意为四字符代码（Four-Character Codes），顾名思义，该编码由四个字符组成,下面是VideoWriter_fourcc对象一些常用的参数,注意：字符顺序不能弄混
    cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi
    cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi
    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi
    cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv
    cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
    cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    文件名后缀为.mp4
参数3 为帧播放速率
参数4 (width,height)为视频帧大小
"""
import cv2

if __name__ == '__main__':
    # 保存视频的FPS，可以适当调整, 帧率过低，视频会有卡顿
    fps = 5
    photo_size = (600, 800)
    # 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # video: 要保存的视频地址
    video = 'D:/software/pyCharm/pythonProject/pythonProject1/video'
    # video = 'F:/BaiduNetdiskDownload/pictures/renlian/Morphing/video_fps1.mp4'
    videoWriter = cv2.VideoWriter(video, fourcc, fps, photo_size)

    for i in range(1, 12):
        # image: 图片地址
        image = "    D:/software/pyCharm/pythonProject/pythonProject1/img" + str(i) + ".jpg"
        frame = cv2.imread(image)
        videoWriter.write(frame)
    # videoWriter.release()