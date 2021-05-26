# show detected frame and store
import cv2
import ipywidgets as widgets 
from IPython import display 
from torchvision.transforms import functional as fc
import numpy as np
import copy 
def show(vidpath,store=False,obj=False,store_root='ssd_video'):
        
    wImg = widgets.Image( layout = widgets.Layout(border="solid") ) 
    display.display( wImg)


    capture = cv2.VideoCapture(vidpath) # 카메라 장치 연결
    
    if not capture.isOpened(): # 초기화 확인
        raise IOError("Can't open webcam")

    if store:
        objroot = store_root
        vidname = os.path.split(vidpath)[-1]
        objvid = os.path.join(vidroot,vidname.replace(os.path.splitext(vidname)[-1],'.avi' ) )
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        out = cv2.VideoWriter(objvid ,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    while True:

        ret, frame = capture.read() # 프레임 읽기
        if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            break

        # detection
        if obj:
            frame = fc.to_pil_image(frame)
            frame = detect(model,frame)
            frame = np.array(frame)

        # show
        tmpStream = cv2.imencode(".jpeg", frame)[1].tostring() 
        wImg.value = tmpStream

        # stroe
        if store:
            out.write(frame)


    capture.release() # 캡처 자원 반납
    cv2.destroyAllWindows()