import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import io
from datetime import datetime
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Tải mô hình đã huấn luyện
model_path = 'model/rice_disease_model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error("Không tìm thấy mô hình phân loại!")

# Các nhãn bệnh
disease_labels = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Narrow Brown Leaf Spot",
    "Rice Hispa",
    "Sheath Blight"
]

# Hàm tiền xử lý ảnh
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Hàm lưu ảnh vào thư mục bệnh
def save_image(image_data, disease_name):
    disease_folder = os.path.join("images", disease_name)
    os.makedirs(disease_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(disease_folder, f"{timestamp}.jpg")
    
    # Mở ảnh từ dữ liệu bytes
    image = Image.open(io.BytesIO(image_data))
    image.save(image_path)
    return image_path

# Nạp CSS từ tệp style.css
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Tiêu đề ứng dụng
st.title("Phân Loại Bệnh Trên Lá Lúa")

# Tạo menu với streamlit-option-menu
with st.sidebar:
    option = option_menu(
        "MENU",
        ["Tải lên ảnh", "Chụp ảnh"],
        icons=["upload", "camera"],
        menu_icon="cast",
        default_index=0,
    )

if option == "Tải lên ảnh":
    uploaded_image = st.file_uploader("Chọn ảnh lá lúa để phân loại:", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        # Tiền xử lý và dự đoán
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = model.predict(processed_image)[0]

        # Hiển thị kết quả dự đoán
        prediction_dict = {disease_labels[i]: prediction[i] for i in range(len(disease_labels))}
        sorted_predictions = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)
        
        st.write("### Kết quả dự đoán:")
        for disease, probability in sorted_predictions:
            st.write(f"{disease}: {probability * 100:.2f}%")

        # Lưu ảnh vào thư mục tương ứng với bệnh
        disease_name = disease_labels[np.argmax(prediction)]
        save_image(uploaded_image.getvalue(), disease_name)

elif option == "Chụp ảnh":
    st.write("Chụp ảnh từ camera")

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Tiền xử lý ảnh
            image = Image.fromarray(img)
            processed_image = preprocess_image(image, target_size=(224, 224))
            prediction = model.predict(processed_image)[0]

            # Hiển thị kết quả dự đoán lên ảnh
            disease_name = disease_labels[np.argmax(prediction)]
            return cv2.putText(img, f"Dự đoán: {disease_name}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
