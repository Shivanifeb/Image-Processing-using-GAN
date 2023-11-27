import streamlit as st
import numpy as np
from numpy import asarray
from PIL import Image
from io import BytesIO

import cv2
import os
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer


def upscale():

        model_path='models/customESRGAN.pth'
        
        my_model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

        upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=my_model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0)


        
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            output, _ = upsampler.enhance(img, outscale=4)
        except RuntimeError as error:
            print('Error', error)
        else:
         
            
            save_path = os.path.join('result_images/upscaled.jpg')
            output= cv2.GaussianBlur(output, (5, 5), 0)
            
            blurred_img = cv2.GaussianBlur(output, (0, 0), 7)
            output = cv2.addWeighted(output, 1.7, blurred_img, -0.7, 0)
            
            cv2.imwrite(save_path, output)







st.title("IMAGE UPSCALER")


img=st.file_uploader("")

col1,col2=st.columns(2)

if img:
    col1.header("Your Image") 
    col2.header("Upscaled Image")
    col1.image(img)

    

    test_img= img

    

   


# read images
    with col2:
            with st.spinner():
                bytes_data = test_img.getvalue()
                img = np.array(Image.open(BytesIO(bytes_data)))
                img=np.flip(img, axis=-1) 

                upscale()

                #img = img * 1.0 / 255
                

            
            
            col2.image("result_images/upscaled.jpg", clamp=True, channels='RGB')
            st.balloons()
            
            with open("result_images/upscaled.jpg") as file:
                btn = col2.download_button(
                        label="Download image",
                        data=file,
                        file_name="upscaled.jpg",
                        mime="image/png"
                    )
                