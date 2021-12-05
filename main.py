import sys
from streamlit import cli as stcli

from folium.plugins import HeatMap
import folium
from streamlit_folium import folium_static

import tempfile

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import os
import json

import torch
import cv2

from stqdm import stqdm

import tkinter as tk
from tkinter import filedialog

@st.cache()
def load_model(path='models/best.pt'):
    detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
    return detection_model


def detect_image(image, model):
    pred = model(image)
    pred_df = pred.pandas().xyxy[0].sort_values('confidence', ascending=False)
    pred_image = pred.render()[0]
    if pred_df.shape[0] > 0:
        if pred_df.confidence.iloc[0] > 0.5:
            return pred_image, pred_df.name.iloc[0]
        else:
            return pred_image, 'Неизвестное животное'
    else:
        return image, 'Нет животного'


def plot_map():
    map_df = pd.read_csv('lisa_tracking.csv')
    map_heatmap = folium.Map(location=[map_df.latitude.mean(), map_df.longitude.mean()], zoom_start=11)
    # Filter the DF for columns, then remove NaNs
    heat_df = map_df[["latitude", "longitude"]]
    heat_df = heat_df.dropna(axis=0, subset=["latitude", "longitude"])
    # List comprehension to make list of lists
    heat_data = [
        [row["latitude"], row["longitude"]] for index, row in heat_df.iterrows()
    ]
    # Plot it on the map
    HeatMap(heat_data).add_to(map_heatmap)
    # Display the map using the community component
    folium_static(map_heatmap)


def process_video(cap, model, save=True, path_to_save='temp.mp4'):
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    stframe = st.empty()
    preds = []
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path_to_save, fourcc, 25.0, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame, pred = detect_image(frame, model)
        #st.write(pred)
        if pred != 'Неизвестное животное' and pred != 'Нет животного':
            if save:
                out.write(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            preds.append(pred)
        elif pred == 'Неизвестное животное':
            preds.append(pred)
    cap.release()
    if save:
        out.release()
    if len(preds) > 0:
        return pd.DataFrame(preds).reset_index().groupby(0).count().sort_values('index').index[-1]
    else:
        return 'Нет животного'

def main():

    st.title('Обработка данных из фотоловушек')

    model = load_model('best.pt')

    data_type = st.radio(
        "Выберите тип данных",
        ('Директория с фото', 'Директория с видео',
         'Фото', 'Видео'))


    if data_type == 'Директория с фото':
        
        st.header('Обработка директории с фотографиями')
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)


        st.write('Выберите папку с фотографиями:')
        clicked = st.button('Выбор папки')

        if clicked:
        # Создать папки, в которых будут лежать обработанные фотографии

            dirname = st.text_input('Выбранная папка:', filedialog.askdirectory(master=root))
            #directory = "ProcessedImages"
            path = './ProcessedImages' #os.path.join(dirname, directory)
            try:
                os.mkdir(path)
            except:
                st.write('WARNING: Папка ProcessedImages уже существует.')


            for i in ['Bear', 'Deer', 'Lynx', 'Fox', 'Tiger', 'Leopard', 'Wolf', 'Eagle', 'Boar', 'Saiga', 'Squirrel', 'Raccoon', 'Man','Weasel', 'Unknown_Animal', 'No_Animal']:
                os.mkdir(os.path.join(path, i))
            # Загрузка модели
            labels = {}
            for img_path in stqdm([dirname + '\\' + x for x in os.listdir(dirname)]):
                try:
                    image = np.array(Image.open(img_path))
                    image, prediction = detect_image(image, model)
                except :
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                labels[img_path.split('\\')[-1]]=[prediction]

                with open('labels.json', 'w') as outfile:
                    json.dump(labels, outfile)

                for i, p in zip(['Медведь', 'Олень', 'Рысь', 'Лиса', 'Тигр', 'Леопард', 'Волк', 'Орёл', 'Кабан', 'Сайгак', 'Белка', 'Енот', 'Человек', 'Ласка', 'Неизвестное животное', 'Нет животного'],
                ['Bear', 'Deer', 'Lynx', 'Fox', 'Tiger', 'Leopard', 'Wolf', 'Eagle', 'Boar', 'Saiga', 'Squirrel', 'Raccoon', 'Man','Weasel', 'Unknown_Animal', 'No_Animal']):
                    if prediction == i:
                        path_init = os.path.join(path, p)
                        path_to_save = os.path.join(path_init, img_path.split('\\')[-1])
                        cv2.imwrite(path_to_save, np.array(image))
                        break

            st.write('Фотографии обработаны')





    elif data_type == 'Директория с видео':
        st.header('Обработка директории с видео')
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)


        st.write('Выберите папку с видео:')
        clicked = st.button('Выбор папки')

        if clicked:
        # Создать папки, в которых будут лежать обработанные фотографии

            dirname = st.text_input('Выбранная папка:', filedialog.askdirectory(master=root))
            #directory = "ProcessedImages"
            path = './ProcessedVideos' #os.path.join(dirname, directory)
            try:
                os.mkdir(path)
            except:
                st.write('WARNING: Папка ProcessedVideos уже существует.')

            labels_video = {}
            for video_path in stqdm([dirname + '\\' + x for x in os.listdir(dirname)]):
                cap = cv2.VideoCapture(video_path)

                prediction = process_video(cap, model, save=True, path_to_save='./ProcessedVideos/'+video_path.split('\\')[-1].split('.')[0] + '.mp4')
                st.write(prediction)


                labels_video[video_path.split('\\')[-1]]=[prediction]

                with open('labels_video.json', 'w') as outfile:
                    json.dump(labels_video, outfile)

            st.write('Видео обработаны')


    elif data_type == 'Фото':
        st.header('Обработка фото')
        file = st.file_uploader('Загрузите изображение')
        if file:
            image = np.array(Image.open(file))
            image, pred = detect_image(image, model)
            st.header('Результаты распознавания')
            st.metric('Вид', pred)
            st.image(image)
            if pred == 'Лиса':
                plot_or_not = st.checkbox('Показать карту активности вида - {}'.format(pred))
                if plot_or_not:
                    plot_map()


    elif data_type == 'Видео':
        st.header('Обработка видео')
        file = st.file_uploader('Загрузите изображение')
        if file:
            st.header('Результаты распознавания')
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            cap = cv2.VideoCapture(tfile.name)
            pred = process_video(cap, model)
            st.text('Видео обработано')
            st.metric('Вид', pred)
            if pred == 'Лиса':
                plot_or_not = st.checkbox('Показать карту активности вида - {}'.format(pred))
                if plot_or_not:
                    plot_map()


if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
