# For start this app, run on tertmital:
# streamlit run path/to/Comparison_img.py --server.port 4180


from threading import Thread
from PIL import Image
import cv2
import numpy as np
import streamlit as st
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
import os
from streamlit.ReportThread import add_report_ctx

# U can put it in a separate file
# This class can return value from thread
# class ThreadWithReturnValue(Thread):
#     def __init__(self, group=None, target=None, name=None, args=[], kwargs=None, *, daemon=None):
#         Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

#         self._return = None

#     def run(self):
#         if self._target is not None:
#             self._return = self._target(*self._args, **self._kwargs)

#     def join(self):
#         Thread.join(self)
#         return self._return
# This part of code responsible for processing and displaying images of a specific thread
# (requires refracting)
def comparison_img(before, after):
    def second_operation(before, after):
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
        y = st.sidebar.slider('Пороговое значение экспозиции изображений:', 0, 255, 125)
        b, before_gray = cv2.threshold(before_gray, y, 255, cv2.THRESH_BINARY)
        a, after_gray = cv2.threshold(after_gray, y, 255, cv2.THRESH_BINARY)
        (score, diff) = compare_ssim(before_gray, after_gray, full=True)
        st.sidebar.subheader("Сходство изображений:")
        st.sidebar.subheader(score)
        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        mask = np.zeros(before.shape, dtype='uint8')
        filled_after = after.copy()
        x = st.sidebar.slider('Пороговое значение площади несоответствия изображений, в относительных еденицах:',0, 10000, 40, 10)
        for c in contours:
            area = cv2.contourArea(c)
            if area > x: # 40
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(before, (x, y), (x + w, y + h), (36, 150, 12), 3)
                cv2.rectangle(after, (x, y), (x + w, y + h), (36, 150, 12), 3)
                cv2.drawContours(mask, [c], 0, (255, 255, 255), -4)
                cv2.drawContours(filled_after, [c], 0, (36, 150, 12), -4)
        st.title('Результат')
        st.subheader('Первое изображение')
        st.image(before, channels="BGR", use_column_width=True)
        st.subheader('Второе изображение')
        st.image(after, channels="BGR", use_column_width=True)
        st.subheader('Места изменений')
        st.image(filled_after, channels="BGR", use_column_width=True)

    if before.shape[0] == after.shape[0] and before.shape[1] == after.shape[1]:
        second_operation(before, after)
    elif before.shape[0]*before.shape[1] > after.shape[0]*after.shape[1]:
        first_size = (before.shape[1], before.shape[0])
        after = cv2.resize(after, first_size, interpolation=cv2.INTER_CUBIC)
        second_operation(before, after)
    elif before.shape[0]*before.shape[1] < after.shape[0]*after.shape[1]:
        first_size = (after.shape[1], after.shape[0])
        before = cv2.resize(before, first_size, interpolation=cv2.INTER_CUBIC)
        second_operation(before, after)


# Upload image and convert to opencv
def uploaded_file(uploaded_file):
    opencv_image = cv2.imdecode(0,0)
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # show image:
        st.image(opencv_image, channels="BGR", use_column_width=True)
    return opencv_image


# This main body streamlit app
# Open u logo:
image = Image.open(r"/Users/denis.murataev/PycharmProjects/WebSearchElastic/Comparison_img/LOGO.png")
st.sidebar.image(image, use_column_width=True)
st.sidebar.title('Сравнение изображений')
# Load u image
uploaded_file_one = st.sidebar.file_uploader("Выберите первое изображение", type=["jpg","png"]) # добавить форматы файлов
uploaded_file_two = st.sidebar.file_uploader("Выберите второе изображение", type=["jpg","png"])
if uploaded_file_one:
    st.title('Первое изображение')
    before = uploaded_file(uploaded_file_one)
if uploaded_file_two:
    st.title('Второе изображение')
    after = uploaded_file(uploaded_file_two)
# Create thread for modification image
# Due to the fact that it is a server and performs constant reading (loop),
# this design excludes the creation of a thread before the images are loaded
# (there may be a more elegant solution, but I don’t know it yet)
# try:
#     thread = ThreadWithReturnValue(target=comparison_img, args=(before, after))
#     add_report_ctx(thread)
#     thread.start()
#     thread.join()
# except:
#     pass
