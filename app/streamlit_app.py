
from pathlib import Path
import sys
import json
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import torch

from PIL import Image
from torchvision import transforms

# --------------------------------------------------
# Project setup
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.device import get_device

from src.data.preprocessing import eval_transform
from src.deep_learning.models import build_resnet18

from src.detection.inference import load_yolo, predict_image
from src.severity.severity import estimate_severity

from src.explainability.gradcam import GradCAM, overlay_gradcam
from src.anomaly.autoencoder import ConvAutoencoder


st.set_page_config(
    page_title="RoadVision",
    page_icon="🛣️",
    layout="wide",
)

cfg = load_config()
device = get_device()

CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
]

IMAGE_SIZE = cfg["data"]["image_size"]


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@st.cache_resource
def load_classifier():
    classes_path = resolve_path(
        "data/processed/classes.json"
    )

    checkpoint_path = resolve_path(
        "checkpoints/resnet18/best.pt"
    )

    if not classes_path.exists() or not checkpoint_path.exists():
        return None, [], None

    classes = list(
        json.loads(
            classes_path.read_text(encoding="utf-8")
        ).keys()
    )

    model = build_resnet18(
        num_classes=len(classes),
        pretrained=False,
    ).to(device)

    checkpoint_data = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state = checkpoint_data.get(
        "model_state",
        checkpoint_data,
    )

    model.load_state_dict(state)
    model.eval()

    gradcam = GradCAM(
        model,
        model.layer4[-1],
    )

    return model, classes, gradcam


@st.cache_resource
def load_detector():
    checkpoint = resolve_path(
        "models/detection/roadvision_clean_split/weights/best.pt"
    )

    if not checkpoint.exists():
        return None

    return load_yolo(checkpoint)


@st.cache_resource
def load_anomaly_model():
    checkpoint = resolve_path(
        "checkpoints/anomaly/autoencoder.pt"
    )

    threshold_path = resolve_path(
        "checkpoints/anomaly/threshold.txt"
    )

    if not checkpoint.exists() or not threshold_path.exists():
        return None, None

    model = ConvAutoencoder(
        cfg["anomaly"]["latent_dim"]
    ).to(device)

    state = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(state)
    model.eval()

    threshold = float(
        threshold_path.read_text(
            encoding="utf-8"
        ).strip()
    )

    return model, threshold


# --------------------------------------------------
# Image analysis functions
# --------------------------------------------------

def classify_image(model, classes, image):
    x = eval_transform(IMAGE_SIZE)(image)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())

    return (
        classes[predicted_index],
        float(probabilities[predicted_index].item()),
        probabilities.cpu().numpy(),
        x,
    )


def generate_gradcam(gradcam, x, image):
    # Grad-CAM requires gradients, so do not wrap this in no_grad().
    cam, class_idx = gradcam(x)

    resized_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_rgb = np.asarray(resized_image)

    overlay = overlay_gradcam(
        image_rgb,
        cam,
        alpha=0.4,
    )

    return overlay, class_idx


def detect_damage(detector, image):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as temp:
            temp_path = Path(temp.name)

        image.save(temp_path, format="JPEG")

        detections, annotated = predict_image(
            detector,
            temp_path,
            cfg.get("detection", {}).get(
                "confidence", 0.25
            ),
            cfg.get("detection", {}).get(
                "iou", 0.50
            ),
        )

        return detections, annotated

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def calculate_anomaly(model, threshold, image):
    # Autoencoder preprocessing must remain in [0, 1].
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstruction = model(x)

        error = (
            (x - reconstruction) ** 2
        ).mean().item()

    return error, error > threshold


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("RoadVision")
st.caption(
    "Multi-paradigm Road Damage Analysis System"
)

st.markdown(
    """
    Analyze road images using classical machine learning,
    deep learning, object detection, anomaly scoring,
    and explainable AI.
    """
)

classifier, classes, gradcam = load_classifier()
detector = load_detector()
anomaly_model, anomaly_threshold = load_anomaly_model()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("RoadVision Control Panel")

st.sidebar.write(
    "**Execution device:**",
    str(device),
)

st.sidebar.divider()

st.sidebar.subheader("Model status")

model_status = {
    "ResNet18 classifier": classifier is not None,
    "YOLO detector": detector is not None,
    "Grad-CAM": gradcam is not None,
    "Autoencoder": anomaly_model is not None,
}

for name, available in model_status.items():
    if available:
        st.sidebar.success(f"{name}: Available")
    else:
        st.sidebar.warning(f"{name}: Unavailable")

st.sidebar.divider()

st.sidebar.caption(
    "RoadVision | Road Damage Analysis"
)


# --------------------------------------------------
# Image upload
# --------------------------------------------------

st.subheader("Upload Road Image")

uploaded = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.info(
        "Upload an image to begin road damage analysis."
    )

    st.markdown("### Available analysis modules")

    col1, col2, col3 = st.columns(3)

    col1.metric("Classification", "ResNet18")
    col2.metric("Detection", "YOLO")
    col3.metric("Explainability", "Grad-CAM")

    st.stop()


# --------------------------------------------------
# Load image
# --------------------------------------------------

try:
    image = Image.open(uploaded).convert("RGB")
except Exception as error:
    st.error(f"Unable to read image: {error}")
    st.stop()


st.divider()

st.subheader("Input Image")

st.image(
    image,
    caption="Uploaded road image",
    use_container_width=True,
)

# Store results for report generation.
report = {
    "image_name": uploaded.name,
}


# --------------------------------------------------
# Run analysis
# --------------------------------------------------

st.subheader("Analysis Results")

with st.spinner("Analyzing road image..."):

    detections = []
    annotated = None

    classification = None
    confidence = None
    probabilities = None
    input_tensor = None

    cam_overlay = None

    severity = None
    anomaly_error = None
    anomaly_flagged = None

    # ----------------------------------------------
    # Classification
    # ----------------------------------------------

    if classifier is not None:
        try:
            (
                classification,
                confidence,
                probabilities,
                input_tensor,
            ) = classify_image(
                classifier,
                classes,
                image,
            )

            report["predicted_class"] = classification
            report["classification_confidence"] = confidence

        except Exception as error:
            st.error(f"Classification failed: {error}")

    # ----------------------------------------------
    # Object detection
    # ----------------------------------------------

    if detector is not None:
        try:
            detections, annotated = detect_damage(
                detector,
                image,
            )

            report["detection_count"] = len(detections)

        except Exception as error:
            st.error(f"Object detection failed: {error}")

    # ----------------------------------------------
    # Severity estimation
    # ----------------------------------------------

    if detector is not None:
        severity = estimate_severity(
            detections,
            image.width,
            image.height,
        )

        report["visual_severity_level"] = severity["level"]
        report["visual_severity_score"] = severity["score"]
        report["damage_area_ratio"] = severity["damage_area_ratio"]

    # ----------------------------------------------
    # Grad-CAM
    # ----------------------------------------------

    if gradcam is not None and input_tensor is not None:
        try:
            cam_overlay, cam_class_idx = generate_gradcam(
                gradcam,
                input_tensor,
                image,
            )

            report["gradcam_class"] = classes[cam_class_idx]

        except Exception as error:
            st.error(f"Grad-CAM failed: {error}")

    # ----------------------------------------------
    # Anomaly scoring
    # ----------------------------------------------

    if anomaly_model is not None:
        try:
            anomaly_error, anomaly_flagged = calculate_anomaly(
                anomaly_model,
                anomaly_threshold,
                image,
            )

            report["reconstruction_error"] = anomaly_error
            report["anomaly_threshold"] = anomaly_threshold
            report["above_anomaly_threshold"] = anomaly_flagged

        except Exception as error:
            st.error(f"Anomaly scoring failed: {error}")


# --------------------------------------------------
# Dashboard metrics
# --------------------------------------------------

st.divider()

st.subheader("Key Performance Indicators")

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    if classification is not None:
        st.metric(
            "Predicted Damage",
            classification,
        )
    else:
        st.metric("Classification", "Unavailable")

with metric2:
    if confidence is not None:
        st.metric(
            "Classification Confidence",
            f"{confidence * 100:.2f}%",
        )
    else:
        st.metric("Confidence", "N/A")

with metric3:
    if severity is not None:
        st.metric(
            "Visual Severity",
            severity["level"],
        )
    else:
        st.metric("Visual Severity", "N/A")

with metric4:
    if detections is not None and detector is not None:
        st.metric(
            "Damage Detections",
            len(detections),
        )
    else:
        st.metric("Detections", "N/A")


# --------------------------------------------------
# Detailed analysis tabs
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Classification",
    "Detection",
    "Grad-CAM",
    "Anomaly",
    "Severity",
])


# --------------------------------------------------
# Tab 1: Classification
# --------------------------------------------------

with tab1:
    st.subheader("ResNet18 Classification")

    if classification is not None:

        st.success(
            f"Predicted class: {classification}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%",
        )

        probability_df = pd.DataFrame({
            "Damage Class": classes,
            "Probability": probabilities,
        }).sort_values(
            "Probability",
            ascending=False,
        )

        st.bar_chart(
            probability_df.set_index("Damage Class")
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
        )

    else:
        st.warning(
            "ResNet18 classifier is unavailable."
        )


# --------------------------------------------------
# Tab 2: Detection
# --------------------------------------------------

with tab2:
    st.subheader("YOLO Object Detection")

    if annotated is not None:

        st.image(
            annotated,
            channels="BGR",
            caption="Detected road damage",
            use_container_width=True,
        )

        if detections:
            detection_df = pd.DataFrame([
                {
                    "Class": item["class_name"],
                    "Confidence": round(
                        item["confidence"], 4
                    ),
                    "Bounding Box": str(
                        [round(v, 2) for v in item["box"]]
                    ),
                }
                for item in detections
            ])

            st.dataframe(
                detection_df,
                use_container_width=True,
            )

        else:
            st.info(
                "No damage objects passed the detection threshold. "
                "This does not prove that the road is damage-free."
            )

    else:
        st.warning("YOLO detector is unavailable.")


# --------------------------------------------------
# Tab 3: Grad-CAM
# --------------------------------------------------

with tab3:
    st.subheader("Explainable AI — Grad-CAM")

    if cam_overlay is not None:

        st.image(
            cam_overlay,
            caption="ResNet18 Grad-CAM heatmap overlay",
            use_container_width=True,
        )

        st.info(
            "Highlighted regions show image areas that contributed "
            "to the selected ResNet18 class prediction. "
            "They are not guaranteed to represent the exact physical "
            "boundaries of road damage."
        )

    else:
        st.warning(
            "Grad-CAM is unavailable."
        )


# --------------------------------------------------
# Tab 4: Anomaly detection
# --------------------------------------------------

with tab4:
    st.subheader("Autoencoder Anomaly Scoring")

    if anomaly_error is not None:

        st.metric(
            "Reconstruction Error",
            f"{anomaly_error:.6f}",
        )

        st.metric(
            "Training-Derived Threshold",
            f"{anomaly_threshold:.6f}",
        )

        if anomaly_flagged:
            st.warning(
                "Reconstruction error exceeds the training-derived threshold."
            )
        else:
            st.success(
                "Reconstruction error is at or below the threshold."
            )

        st.caption(
            "This is a reconstruction-error flag, not a validated "
            "normal-road versus damaged-road classification. "
            "The autoencoder was trained on damaged-road images."
        )

    else:
        st.warning(
            "Autoencoder is unavailable."
        )


# --------------------------------------------------
# Tab 5: Severity
# --------------------------------------------------

with tab5:
    st.subheader("Visual Damage Assessment")

    if severity is not None:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Visual Severity Score",
            severity["score"],
        )

        col2.metric(
            "Visual Severity Level",
            severity["level"],
        )

        col3.metric(
            "Detected Objects",
            severity["count"],
        )

        st.metric(
            "Bounding-Box Area Ratio",
            f"{severity['damage_area_ratio'] * 100:.2f}%",
        )

        st.info(
            "The severity score is a heuristic based on detected "
            "bounding-box area, detection count, and confidence. "
            "It is not a calibrated measure of structural road safety."
        )

    else:
        st.warning(
            "Severity estimation is unavailable."
        )


# --------------------------------------------------
# Download analysis report
# --------------------------------------------------

st.divider()

st.subheader("Export Analysis")

if report:
    report_df = pd.DataFrame([report])

    st.download_button(
        label="Download Analysis Report (CSV)",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="roadvision_analysis.csv",
        mime="text/csv",
    )

if detections:
    detection_report = pd.DataFrame(detections)

    st.download_button(
        label="Download Detection Details (CSV)",
        data=detection_report.to_csv(index=False).encode("utf-8"),
        file_name="roadvision_detections.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "RoadVision | Multi-Paradigm Road Damage Analysis | "
    "Research and educational prototype"
)
