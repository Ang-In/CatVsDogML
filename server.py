import streamlit as st
from fastai.vision.all import *

def cat_vs_dog_func(file_name):
    if file_name[0].isupper():
        return True
    else:
        return False

cat_vs_dog_model = load_learner("cat_vs_dog_model.pkl")

def predict(image):
    img = PILImage.create(image)
    img = img.resize((224, 224))
    predClass, predIdx, outputs = cat_vs_dog_model.predict(img)
    likelihoodIsCat = outputs[1].item()
    if likelihoodIsCat > 0.9:
        return "Cat!"
    elif likelihoodIsCat < 0.001:
        return "Dog!"
    else:
        return "Not sure... try a different picture!"

st.title("Crystal's Cat vs. Dog ML")
st.text("Built by Crystal")

uploadedFile = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploadedFile is not None:
    st.image(uploadedFile, caption="Your Image!", use_column_width=True)

    prediction = predict(uploadedFile)
    st.write(prediction)

st.text("Built with Streamlit and Fastai.")