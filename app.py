import streamlit as st
from PIL import Image
import numpy as np
import cv2 as cv

# Page Configuration:
st.set_page_config(
    page_title="Bollywood Face Recognition",
    page_icon="🦈",
)    

# Title of project :
st.title('Bollywood Face Recognition')

# Celebrity People :
people = [
    'Esha Gupta', 'Alia Bhatt','Bipasha Basu','Huma Qureshi',
    'Aishwarya Rai Bachchan','Anushka Sharma','Deepika Padukone',
    'Genelia D_Souza','Bhumi Pednekar','Disha Patani',
    'Aftab Shivdasani', 'Ajay Devgn', 'Abhishek Bachchan',
    'Aditya Roy Kapur', 'Aamir Khan'
]

# Image Uploading :
uploaded_files = st.file_uploader(
    "Upload multiple images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Load Haar Cascade properly
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Load features and labels
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

# Load trained recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

st.subheader('Result of Face Recognition...')

# Process each uploaded image
for file in uploaded_files:
    image = Image.open(file)
    st.image(image, caption=file.name)

    # Convert PIL to OpenCV (RGB → BGR)
    img = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    # Convert BGR TO GRAY
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    if len(faces_rect) == 0:
        st.warning(f'No Face Detected in {file.name}')
    else:
        for (x,y,w,h) in faces_rect:
            faces_roi = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(faces_roi)

            # Get name
            name = people[label]

            cv.putText(img, name, (x,y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
            cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

            st.success(f'Person: {name}')
            st.info(f'Confidence: {confidence:.2f}')

        st.subheader('Detected Face:')
        result = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.image(result, width=600)


# Discription :
with st.container():
    st.markdown("### Bollywood Dataset Box")

    # Description
    st.write("Hey buddy, I just limited the Bollywood dataset. "
             "Here are the celebrities included:")

    
    st.table({"Celebrity Name": people})

    st.metric(label="Total Celebrities", value=len(people), delta="+15")


footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: #333;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
.footer a {
    color: #333;
    margin: 0 10px;
    text-decoration: none;
}
.footer a:hover {
    color: #0073e6;
}
</style>
<div style='text-align: center;'>
<p>Developed by Azahar Vajir Patel</p>
</div>
<div class="footer">
    Connect with me:
    <a href="https://leetcode.com/u/Azahar094/" target="_blank"> leetcode </a> |
    <a href="linkedin.com/in/azhar-vajir-patel" target="_blank"> LinkedIn</a> |
    <a href="github.com/Azhar9076" target="_blank"> GitHub</a>
</div>
"""

# Render footer
st.markdown(footer_html, unsafe_allow_html=True)