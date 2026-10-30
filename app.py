import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Cache the AI so it does not reload from scratch on every upload
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("Dental AI Diagnostic Tool")
st.write("Upload a PA film. I only output radiographic findings.")

uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    
    with st.spinner("Scanning X-Ray..."):
        results = model(image)
        detected_classes = []
        
        if hasattr(results[0], 'probs') and results[0].probs is not None:
            probs = results[0].probs
            for i, conf in enumerate(probs.data):
                if conf > 0.5:
                    detected_classes.append(model.names[i])
        elif hasattr(results[0], 'boxes') and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name not in detected_classes:
                    detected_classes.append(class_name)
                    
        if len(detected_classes) == 0:
            st.success("Findings: No pathology detected above the 50% confidence threshold.")
        else:
            st.warning(f"AI Findings Detected: {', '.join(detected_classes)}")