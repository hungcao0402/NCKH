import streamlit as st
import torch
from PIL import Image
import numpy as np
import random
import requests

def add_border(image,color, border_width=3.5):
    # Add a red border to the image
    image_with_border = Image.new(image.mode, (image.width + 2*border_width, image.height + 2*border_width), color=color)
    image_with_border.paste(image, (border_width, border_width))
    return image_with_border


def search(dataset: str='market15101',  topk: int = 20):
    '''
    Make a request to the API endpoint
    return: outputs = [{'img_path', 'g_pid', 'score'}, ...]
    '''

    api_url = 'http://localhost:8000/search'
    payload = {
        'dataset': dataset,
        'topk': topk
    }
    response = requests.get(api_url, params=payload)
    outputs = response.json()

    return outputs
    

def app():
    st.set_page_config(layout="wide")
    st.title("Person Re-identification Demo")

    num_images = st.number_input("Enter the number of top-k images to display:", min_value=1, max_value=200, value=30)

    agree = st.checkbox('Use label ID')

    if agree:
        person_id = st.number_input("Enter the person ID:", min_value=1, max_value=1500)


    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:       
        st.image(uploaded_file, caption='Uploaded Image.')

        save_path = f'database/test.jpg'
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    search_query = st.button("search")

    if search_query:
        # Make a request to the API endpoint
        outputs = search('market1501', num_images)
        
        # Display the images
        st.write(f"Showing {num_images} images for query person")
        images = []
        captions=[]
        for output in outputs[:num_images]:
            image = Image.open(output['img_path'])

            if agree:
                if int(output['g_pid']) != int(person_id):
                    image= add_border(image,'red', border_width=3)
                else:
                    image= add_border(image,'green', border_width=3)
            image = add_border(image,'white', border_width=3)
            image = image.resize((200,350))
            images.append(image)
            caption = str(int(output['g_pid']))
            captions.append(caption)
        st.image(images)



if __name__ == '__main__':
    app()
