import time
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from PIL import Image
st.set_page_config(layout="wide")

st.title("Nhận diện bệnh Viêm phổi, Covid sử dụng CNN")

model = tf.keras.models.load_model('mv2.h5')

uploaded_file = st.file_uploader("Choose a X-ray image file", type=["jpg","jpeg","png"])

k=[[0,1,0],[1,-4,1],[0,1,0]]

def histogram_equalization(image):
    # Chuyển đổi ảnh sang ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tính toán histogram của ảnh xám
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    # Tính toán histogram tích lũy
    cum_hist = hist.cumsum()
    
    # Chuẩn hóa histogram tích lũy
    cum_hist_normalized = cum_hist / cum_hist.max()
    
    # Ánh xạ lại các giá trị cường độ từ histogram tích lũy chuẩn hóa
    mapping = (cum_hist_normalized * 255).astype(np.uint8)
    
    # Áp dụng phép ánh xạ lại lên ảnh gốc
    equalized_image = mapping[gray_image.astype(np.uint8)]
    
    return equalized_image

def Sacnet(imgPIL):
    sacnet = Image.new(imgPIL.mode, imgPIL.size)
    width = sacnet.size[0]
    height = sacnet.size[1]
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            rs = 0
            gs = 0
            bs = 0
            rs1 = 0
            gs1 = 0
            bs1 = 0
            a = 0
            b = 0
            for i in range(x - 1, x + 1 + 1):
                for j in range(y - 1, y + 1 + 1):
                    color = imgPIL.getpixel((i, j))
                    R = color[0]
                    G = color[1]
                    B = color[2]
                    rs += R * k[a][b]
                    gs += G * k[a][b]
                    bs += B * k[a][b]
                    b += 1
                    if b == 3:
                        b = 0
                a += 1
                if a == 3:
                    a = 0
            R1, G1, B1 = imgPIL.getpixel((x, y))
            rs1 = R1 - rs
            gs1 = G1 - gs
            bs1 = B1 - bs

            if rs1 > 255:
                rs1 = 255
            elif rs1 < 0:
                rs1 = 0
            if gs1 > 255:
                gs1 = 255
            elif gs1 < 0:
                gs1 = 0
            if bs1 > 255:
                bs1 = 255
            elif bs1 < 0:
                bs1 = 0

            sacnet.putpixel((x, y), (bs1, gs1, rs1))
    return sacnet
    
if uploaded_file is not None:
    img = image.load_img(uploaded_file,target_size=(300,300))
    
    col1, col2 = st.columns(2) 
    with col1:
        st.write('**X-RAY IMAGE NON-PROCESS**')
        st.image(img, channels="RGB")
        Process = st.button("**Pre-process & Predict**")

    if Process:
        img_array = Sacnet(img)
        img_array = img_to_array(img)
        img_array = histogram_equalization(img_array)
        img_array = np.expand_dims(img_array, axis=-1)  # Thêm kích thước kênh màu cuối cùng
        img_array = np.repeat(img_array, 3, axis=-1)  # Lặp lại giá trị kênh màu để có kích thước (150, 150, 3)
        img = image.array_to_img(img_array)
        
        with col2:
            st.write('**X-RAY IMAGE IS PROCESSED**')
            st.image(img, channels="RGB")
            img = img.resize((150,150))
            img = img_to_array(img)
            img = img.reshape(1,150,150,3)
            img = img.astype('float32')
            img = img / 255

            with st.spinner("Waiting !!!"):
                time.sleep(2)

            result = int(np.argmax(model.predict(img),axis =1))
            percent = model.predict(img)

            if result == 0:
                st.write("**Based on the x-ray image it is COVID19**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ",percent,"%")
            elif result == 1 :
                st.write("**Based on the x-ray image it is HEALTHY**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ", percent,"%")
            else :
                st.write("**Based on the x-ray image it is PNEUMOIA**")
                percent = (percent.max())*100
                st.write("**Accuracy:** ", percent,"%")

