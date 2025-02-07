import streamlit as st
import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv
from PIL import Image


st.set_page_config(
    page_title="Ocean Guardian",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        }
        .stButton>button {
            background: #4CAF50 !important;
            color: white !important;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stFileUploader>section>div>div>div>button {
            background: #2196F3 !important;
            color: white !important;
        }
        .result-box {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stats {
            font-size: 1.2rem;
            color: #2c3e50;
            margin: 0.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    rf = Roboflow(api_key="UA0QHQjexJi80OXMSbAD")
    project = rf.workspace().project("ocean-waste")
    return project.version(2).model

model = load_model()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Detection Confidence", 10, 90, 40, step=5)
    overlap = st.slider("Overlap Threshold", 10, 90, 30, step=5)
    st.markdown("---")
    st.markdown("🔍 **Detection Classes**")
    st.write("- Plastic Waste")
    st.write("- Oil Spill")
    st.write("- Fishing Gear")
    st.write("- Other Debris")
    st.markdown("---")

# --- Main Interface ---
st.title("🌊 Ocean Guardian AI")
st.markdown("### Protect Our Oceans with AI-Powered Pollution Detection")

# --- File Upload Section ---
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader(
        "Upload Ocean Image",
        type=["jpg", "png", "jpeg"],
        help="Upload a clear image of ocean area for pollution detection"
    )


if uploaded_file:
    with st.spinner("🔍 Analyzing ocean environment..."):
        # Process image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        # Run detection
        result = model.predict(image_np, confidence=confidence, overlap=overlap).json()
        detections = sv.Detections.from_inference(result)
        
        # Annotate image
        label_annotator = sv.LabelAnnotator()
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(
            scene=cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
            detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections,
            labels=[f"{item['class']} {item['confidence']:.0%}" 
                   for item in result["predictions"]]
        )
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

   
    with col2:
        with st.container():
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            
            # Image display
            st.image(annotated_image, use_column_width=True, 
                    caption="Detected Pollution Overview")
       
            detection_count = len(result["predictions"])
            if detection_count > 0:
                st.error(f"🚨 Alert: {detection_count} pollution items detected!")
            else:
                st.success("✅ Clean Waters Detected!")
                
            st.markdown(f"""
                <div class='stats'>
                    📊 Detection Statistics:<br>
                    - Total Items Found: {detection_count}<br>
                    - Most Common Pollutant: {
                        max(set([p['class'] for p in result["predictions"]]), 
                            default="None")
                    }<br>
                    - Average Confidence: {
                        np.mean([p['confidence'] for p in result["predictions"]])*100:.1f}%
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    # Additional Actions
    with st.expander("📥 Download Results"):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="Download Annotated Image",
                data=cv2.imencode('.jpg', annotated_image)[1].tobytes(),
                file_name="pollution_analysis.jpg",
                mime="image/jpeg"
            )
        with col_d2:
            st.download_button(
                label="Download Report",
                data=f"Pollution Detection Report\n\nTotal Items: {detection_count}",
                file_name="pollution_report.txt"
            )

else:
    with col2:
        st.markdown("""
            <div class='result-box' style='min-height: 500px;'>
                <h3 style='color: #2c3e50; text-align: center; margin-top: 2rem;'>
                    🐋 Welcome to Ocean Guardian!
                </h3>
                <p style='text-align: center;'>
                    Upload an image to start analyzing ocean pollution<br>
                    Supported formats: JPG, PNG, JPEG
                </p>
                <div style='text-align: center; font-size: 8rem; margin: 2rem;'>🌍</div>
            </div>
        """, unsafe_allow_html=True)
