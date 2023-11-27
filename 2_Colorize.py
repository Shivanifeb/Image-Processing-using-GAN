
import streamlit as st
from PIL import Image
import numpy as np

import cv2
from io import BytesIO
import cv2


from deoldify import device
from deoldify.device_id import DeviceId
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)

from deoldify.visualize import *
plt.style.use('dark_background')
torch.backends.cudnn.benchmark=True



colorizer = get_image_colorizer(artistic=True)


st.title("IMAGE COLORIZER")

img1=st.file_uploader("")

col1,col2=st.columns(2)

    

def color():
    with col2:
        with st.spinner():
            #NOTE:  Max is 45 with 11GB video cards. 35 is a good default
            render_factor=20
            #NOTE:  Make source_url None to just read from file at ./video/source/[file_name] directly without modification
            source_url=None
            source_path = 'saved/image.jpg'
            result_path = None

            if source_url is not None:
                result_path = colorizer.plot_transformed_image_from_url(url=source_url, path=source_path, render_factor=render_factor, compare=True)
            else:
                result_path = colorizer.plot_transformed_image(path=source_path, render_factor=render_factor, compare=True)
            



if img1:

    col1.header("Original Image") 
    col2.header("Colorized Image")
    col1.image(img1)

    bytes_data = img1.getvalue()
    img1 = np.array(Image.open(BytesIO(bytes_data)))
    cv2.imwrite("saved/image.jpg",img1)
    
    color()

    col2.image("result_images/image.jpg")
