import base64
import io
import os
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as ReportImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="CliniScan | Lung Abnormality Detection",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

DET_URL = (
    "https://huggingface.co/vasuchakravarthi/cliniscan-models/"
    "resolve/main/best1.pt?download=true"
)
CLF_URL = (
    "https://huggingface.co/vasuchakravarthi/cliniscan-models/"
    "resolve/main/best_clf_model.pth?download=true"
)
CLASS_NAMES = ["Abnormal", "Normal"]

if "page" not in st.session_state:
    st.session_state.page = "home"
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# -----------------------------------------------------------------------------
# STYLE
# -----------------------------------------------------------------------------

def apply_css(dark_mode: bool):
    if dark_mode:
        background = "#0b1220"
        card = "#162033"
        card_alt = "#1d2a40"
        text = "#e5eefb"
        muted = "#9fb0c8"
        border = "#2b3b55"
    else:
        background = "#f4f9fc"
        card = "#ffffff"
        card_alt = "#ecf7fb"
        text = "#102a43"
        muted = "#526d82"
        border = "#d8e6ef"

    st.markdown(
        f"""
        <style>
            .stApp {{
                background: {background};
                color: {text};
            }}
            .block-container {{
                max-width: 1400px;
                padding-top: 1.5rem;
                padding-bottom: 2.5rem;
            }}
            .hero {{
                background: linear-gradient(125deg, #0a7891, #125e9d 55%, #2e9eae);
                padding: 2rem 2.2rem;
                border-radius: 22px;
                color: white;
                margin-bottom: 1.25rem;
                box-shadow: 0 12px 28px rgba(14, 89, 132, 0.22);
            }}
            .hero h1 {{
                margin: 0;
                font-size: 2.3rem;
                letter-spacing: -0.5px;
            }}
            .hero p {{
                margin: 0.55rem 0 0;
                font-size: 1.05rem;
                opacity: 0.95;
            }}
            .medical-card {{
                background: {card};
                border: 1px solid {border};
                border-radius: 18px;
                padding: 1rem 1.1rem;
                min-height: 108px;
                box-shadow: 0 5px 16px rgba(30, 70, 100, 0.07);
                animation: fadeUp 0.45s ease-in;
            }}
            .medical-card .label {{
                color: {muted};
                font-size: 0.82rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }}
            .medical-card .value {{
                color: {text};
                margin-top: 0.35rem;
                font-size: 1.55rem;
                font-weight: 750;
                overflow-wrap: anywhere;
            }}
            .medical-card .small {{
                color: {muted};
                margin-top: 0.25rem;
                font-size: 0.85rem;
            }}
            .section-title {{
                color: {text};
                font-weight: 750;
                font-size: 1.28rem;
                margin: 0.6rem 0;
            }}
            .footer {{
                text-align: center;
                color: {muted};
                border-top: 1px solid {border};
                margin-top: 2rem;
                padding-top: 1rem;
                font-size: 0.85rem;
            }}
            @keyframes fadeUp {{
                from {{ opacity: 0; transform: translateY(8px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @media (max-width: 760px) {{
                .hero {{ padding: 1.2rem; border-radius: 16px; }}
                .hero h1 {{ font-size: 1.65rem; }}
                .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="medical-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="small">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_header():
    st.markdown(
        """
        <div class="hero">
            <h1>🩻 CliniScan</h1>
            <p>AI-assisted chest X-ray screening with abnormality classification, YOLO detection, and visual model attention.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("CliniScan Controls")
        st.toggle("🌙 Dark mode", key="dark_mode")
        st.caption("Theme changes apply after the next app rerun.")
        st.divider()
        st.subheader("About")
        st.markdown(
            """
            **Classification**
            - Abnormal (Class 0)
            - Normal (Class 1)

            **Detection**
            - 14 chest X-ray abnormality categories

            **Important:** This project is for education and research only. It is not a clinical diagnostic device.
            """
        )
        st.markdown("[GitHub Repository](https://github.com/vasuchakravarthi/cliniscan-lung-detection)")


def show_footer():
    st.markdown(
        """
        <div class="footer">
            <strong>⚠️ Educational and research use only.</strong><br>
            Do not use this application for clinical diagnosis or medical decision-making.<br>
            Vasu Chakravarthi | SRKR Engineering College | BTech AIML 2025
        </div>
        """,
        unsafe_allow_html=True,
    )


def go_to(page: str):
    st.session_state.page = page
    st.rerun()


# -----------------------------------------------------------------------------
# MODEL DOWNLOAD AND LOAD
# -----------------------------------------------------------------------------

@st.cache_resource
def download_models():
    os.makedirs("models/detection", exist_ok=True)
    os.makedirs("models/classification", exist_ok=True)

    det_path = "models/detection/best.pt"
    clf_path = "models/classification/best_clf_model.pth"

    try:
        if not os.path.exists(det_path) or os.path.getsize(det_path) == 0:
            urllib.request.urlretrieve(DET_URL, det_path)
        if not os.path.exists(clf_path) or os.path.getsize(clf_path) == 0:
            urllib.request.urlretrieve(CLF_URL, clf_path)
    except Exception as exc:
        raise RuntimeError(
            "Model download failed. Verify that both Hugging Face model URLs are public. "
            f"Details: {exc}"
        ) from exc

    if not os.path.exists(det_path) or os.path.getsize(det_path) == 0:
        raise RuntimeError("Detection model file is missing or empty.")
    if not os.path.exists(clf_path) or os.path.getsize(clf_path) == 0:
        raise RuntimeError("Classification model file is missing or empty.")

    return det_path, clf_path


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def load_models():
    det_path, clf_path = download_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clf_model = EfficientNetClassifier(num_classes=2, dropout=0.3).to(device)
    checkpoint = torch.load(clf_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    clf_model.load_state_dict(state_dict, strict=True)
    clf_model.eval()

    det_model = YOLO(det_path)
    return clf_model, det_model


TRANSFORM = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# -----------------------------------------------------------------------------
# EXPLAINABILITY, PLOTS, REPORT
# -----------------------------------------------------------------------------

def generate_activation_map(model, img_tensor):
    """Activation-map visualization based on EfficientNet-B3 conv_head features."""
    try:
        device = next(model.parameters()).device
        feature_extractor = create_feature_extractor(model.model, {"conv_head": "feat"})

        with torch.no_grad():
            x = img_tensor.unsqueeze(0).to(device)
            features = feature_extractor(x)

        feature_map = features["feat"].squeeze(0).mean(dim=0).cpu().numpy()
        heatmap = cv2.resize(feature_map, (512, 512))
        heatmap = np.maximum(heatmap, 0)
        max_value = float(heatmap.max())
        if max_value > 0:
            heatmap = heatmap / max_value
        return heatmap
    except Exception as exc:
        raise RuntimeError(f"Unable to create model-attention heatmap: {exc}") from exc


def create_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float):
    original = np.array(image.resize((512, 512)))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original, 1 - alpha, colored, alpha, 0)


def probability_chart(probs):
    colors_list = ["#e75a5a", "#1cae85"]
    figure = go.Figure(
        go.Bar(
            x=CLASS_NAMES,
            y=[float(probs[0]), float(probs[1])],
            marker_color=colors_list,
            text=[f"{float(probs[0]):.1%}", f"{float(probs[1]):.1%}"],
            textposition="auto",
        )
    )
    figure.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=25, b=10),
        title="Classification probabilities",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        xaxis_title=None,
        yaxis_title="Probability",
    )
    return figure


def confidence_gauge(confidence: float, predicted_label: str):
    bar_color = "#e75a5a" if predicted_label == "Abnormal" else "#1cae85"
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": f"{predicted_label} confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 50], "color": "#e8eef3"},
                    {"range": [50, 75], "color": "#cbe9e0"},
                    {"range": [75, 100], "color": "#a8dcca"},
                ],
            },
        )
    )
    figure.update_layout(height=260, margin=dict(l=20, r=20, t=45, b=10))
    return figure


def image_to_buffer(image_array):
    image = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def create_pdf_report(result):
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("CliniScan Lung Abnormality Analysis Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Educational and research use only. Not for clinical diagnosis.", styles["Italic"]))
    story.append(Spacer(1, 14))

    table_data = [
        ["Field", "Result"],
        ["Analysis time", result["timestamp"]],
        ["Classification", result["prediction"]],
        ["Classification confidence", f"{result['confidence']:.2%}"],
        ["Total detections", str(result["total_detections"])],
        ["Top detection", result["top_detection"]],
        ["Top detection confidence", f"{result['top_detection_confidence']:.2%}"],
        ["Average detection confidence", f"{result['average_detection_confidence']:.2%}"],
    ]
    table = Table(table_data, colWidths=[2.25 * inch, 4.4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0a7891")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c9d4")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f9fc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 16))

    if result.get("overlay") is not None:
        story.append(Paragraph("Model-attention visualization", styles["Heading2"]))
        story.append(Spacer(1, 8))
        story.append(ReportImage(image_to_buffer(result["overlay"]), width=5.5 * inch, height=5.5 * inch))
        story.append(Spacer(1, 12))

    if result.get("detection_image") is not None:
        story.append(Paragraph("YOLO detection output", styles["Heading2"]))
        story.append(Spacer(1, 8))
        story.append(ReportImage(image_to_buffer(result["detection_image"]), width=5.5 * inch, height=4.3 * inch))

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# PAGES
# -----------------------------------------------------------------------------

def home_page():
    show_header()
    st.markdown("<div class='section-title'>Welcome to CliniScan</div>", unsafe_allow_html=True)
    st.write(
        "CliniScan combines an EfficientNet-B3 classifier, YOLOv8 abnormality detector, "
        "and model-attention visualization for chest X-ray analysis."
    )

    a, b, c = st.columns(3)
    with a:
        card("Classification", "Abnormal / Normal", "EfficientNet-B3")
    with b:
        card("Detection", "14 Categories", "YOLOv8 bounding boxes")
    with c:
        card("Explainability", "Attention Overlay", "Adjustable heatmap intensity")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔐 Login", use_container_width=True):
            go_to("login")
    with c2:
        if st.button("🧪 Continue to Free Trial", type="primary", use_container_width=True):
            go_to("trial")
    show_footer()


def login_page():
    show_header()
    st.markdown("<div class='section-title'>Secure access</div>", unsafe_allow_html=True)
    st.info("Demo credentials: username `admin`, password `cliniscan`.")

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Login", type="primary", use_container_width=True):
            if username == "admin" and password == "cliniscan":
                go_to("dashboard")
            else:
                st.error("Invalid credentials.")
    with c2:
        if st.button("← Back", use_container_width=True):
            go_to("home")
    show_footer()


def trial_page():
    show_header()
    st.markdown("<div class='section-title'>Free Trial</div>", unsafe_allow_html=True)
    st.info("Upload a chest X-ray image to test the analysis workflow.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start analysis", type="primary", use_container_width=True):
            go_to("dashboard")
    with c2:
        if st.button("← Back", use_container_width=True):
            go_to("home")
    show_footer()


def add_history(result):
    history_item = {
        "timestamp": result["timestamp"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "detections": result["total_detections"],
        "top_detection": result["top_detection"],
        "top_detection_confidence": result["top_detection_confidence"],
        "average_detection_confidence": result["average_detection_confidence"],
        "overlay": result.get("overlay"),
        "detection_image": result.get("detection_image"),
    }
    st.session_state.history.insert(0, history_item)
    st.session_state.history = st.session_state.history[:20]


def dashboard_page():
    show_header()

    tab_analysis, tab_analytics, tab_history = st.tabs(
        ["🩻 Analysis", "📈 Analytics", "📚 History & Compare"]
    )

    with tab_analysis:
        uploaded_file = st.file_uploader(
            "Upload chest X-ray (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            key="xray_uploader",
        )

        if uploaded_file is None:
            st.info("Upload a chest X-ray to start classification, detection, and visual analysis.")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded chest X-ray", use_container_width=True)

            try:
                with st.spinner("Loading AI models. The first request may take a minute..."):
                    clf_model, det_model = load_models()
            except Exception as exc:
                st.error(str(exc))
                st.stop()

            image_tensor = TRANSFORM(image)
            device = next(clf_model.parameters()).device

            with st.spinner("Running classification and abnormality detection..."):
                with torch.no_grad():
                    logits = clf_model(image_tensor.unsqueeze(0).to(device))
                    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    predicted_index = int(np.argmax(probabilities))

                results = det_model.predict(
                    source=np.array(image),
                    conf=0.25,
                    verbose=False,
                )

            yolo_result = results[0]
            detection_image = yolo_result.plot()
            boxes = yolo_result.boxes

            total_detections = 0
            top_detection = "No finding detected"
            top_detection_confidence = 0.0
            average_detection_confidence = 0.0
            findings = []

            if boxes is not None and len(boxes) > 0:
                confidence_values = boxes.conf.detach().cpu().numpy()
                class_ids = boxes.cls.detach().cpu().numpy().astype(int)
                total_detections = len(boxes)
                top_index = int(np.argmax(confidence_values))
                top_detection = str(det_model.names[class_ids[top_index]])
                top_detection_confidence = float(confidence_values[top_index])
                average_detection_confidence = float(np.mean(confidence_values))
                findings = [
                    {
                        "Abnormality": str(det_model.names[class_id]),
                        "Confidence": float(confidence),
                    }
                    for class_id, confidence in zip(class_ids, confidence_values)
                ]

            predicted_label = CLASS_NAMES[predicted_index]
            predicted_confidence = float(probabilities[predicted_index])

            st.markdown("<div class='section-title'>Analysis summary</div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                card("Classification", predicted_label, "EfficientNet-B3 output")
            with m2:
                card("Prediction confidence", f"{predicted_confidence:.1%}", "Probability of selected class")
            with m3:
                card("Detections", str(total_detections), "YOLOv8 findings")
            with m4:
                card("Top finding", top_detection, f"Confidence: {top_detection_confidence:.1%}")

            left_col, right_col = st.columns(2)
            with left_col:
                st.plotly_chart(
                    confidence_gauge(predicted_confidence, predicted_label),
                    use_container_width=True,
                    key="classification_gauge",
                )
                st.plotly_chart(
                    probability_chart(probabilities),
                    use_container_width=True,
                    key="probability_chart",
                )

            with right_col:
                st.markdown("<div class='section-title'>Detection output</div>", unsafe_allow_html=True)
                st.image(
                    detection_image,
                    caption="Detected abnormalities with bounding boxes",
                    use_container_width=True,
                )

                if findings:
                    st.dataframe(
                        pd.DataFrame(findings).assign(
                            Confidence=lambda df: df["Confidence"].map(lambda x: f"{x:.2%}")
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success("No YOLO detections were returned at the selected confidence threshold.")

            st.markdown("<div class='section-title'>🫁 Model-attention overlay</div>", unsafe_allow_html=True)
            st.caption("This visualization uses the EfficientNet-B3 `conv_head` activation map. It is an explainability aid, not a clinical heatmap.")
            alpha = st.slider(
                "Heatmap overlay intensity",
                min_value=0.0,
                max_value=1.0,
                value=0.40,
                step=0.05,
                key="gradcam_alpha",
            )

            overlay = None
            try:
                heatmap = generate_activation_map(clf_model, image_tensor)
                overlay = create_overlay(image, heatmap, alpha)
                original_col, overlay_col = st.columns(2)
                with original_col:
                    st.image(image.resize((512, 512)), caption="Original X-ray", use_container_width=True)
                with overlay_col:
                    st.image(
                        overlay,
                        caption=f"Model-attention overlay — {alpha:.0%} intensity",
                        use_container_width=True,
                    )
            except Exception as exc:
                st.warning(str(exc))

            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": predicted_label,
                "confidence": predicted_confidence,
                "probabilities": probabilities.tolist(),
                "total_detections": total_detections,
                "top_detection": top_detection,
                "top_detection_confidence": top_detection_confidence,
                "average_detection_confidence": average_detection_confidence,
                "findings": findings,
                "overlay": overlay,
                "detection_image": detection_image,
            }
            st.session_state.last_result = result

            action1, action2 = st.columns(2)
            with action1:
                if st.button("➕ Save this analysis to history", use_container_width=True):
                    add_history(result)
                    st.success("Analysis saved in this browser session.")
            with action2:
                pdf_data = create_pdf_report(result)
                st.download_button(
                    "📄 Download PDF report",
                    data=pdf_data,
                    file_name=f"cliniscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    with tab_analytics:
        st.markdown("<div class='section-title'>Analytics dashboard</div>", unsafe_allow_html=True)
        if not st.session_state.history:
            st.info("Save one or more analyses to populate session analytics.")
        else:
            history_df = pd.DataFrame(
                [
                    {
                        "Timestamp": item["timestamp"],
                        "Classification": item["prediction"],
                        "Confidence": item["confidence"],
                        "Detections": item["detections"],
                        "Top finding": item["top_detection"],
                    }
                    for item in st.session_state.history
                ]
            )

            a1, a2, a3 = st.columns(3)
            with a1:
                card("Saved analyses", str(len(history_df)), "Current browser session")
            with a2:
                abnormal_rate = float((history_df["Classification"] == "Abnormal").mean())
                card("Abnormal predictions", f"{abnormal_rate:.1%}", "Among saved analyses")
            with a3:
                card("Mean confidence", f"{history_df['Confidence'].mean():.1%}", "Classifier confidence")

            class_counts = history_df["Classification"].value_counts().reindex(CLASS_NAMES, fill_value=0)
            classification_figure = go.Figure(
                go.Bar(
                    x=class_counts.index,
                    y=class_counts.values,
                    marker_color=["#e75a5a", "#1cae85"],
                    text=class_counts.values,
                    textposition="auto",
                )
            )
            classification_figure.update_layout(
                title="Saved classification outcomes",
                height=310,
                margin=dict(l=15, r=15, t=45, b=15),
                yaxis_title="Number of analyses",
            )

            trend_figure = go.Figure(
                go.Scatter(
                    x=list(range(1, len(history_df) + 1)),
                    y=history_df.iloc[::-1]["Confidence"].tolist(),
                    mode="lines+markers",
                    line=dict(color="#0a7891", width=3),
                )
            )
            trend_figure.update_layout(
                title="Classification confidence trend",
                height=310,
                margin=dict(l=15, r=15, t=45, b=15),
                xaxis_title="Analysis order",
                yaxis=dict(title="Confidence", tickformat=".0%", range=[0, 1]),
            )

            p1, p2 = st.columns(2)
            with p1:
                st.plotly_chart(classification_figure, use_container_width=True, key="history_class_chart")
            with p2:
                st.plotly_chart(trend_figure, use_container_width=True, key="history_trend_chart")

    with tab_history:
        st.markdown("<div class='section-title'>Detection history and comparison</div>", unsafe_allow_html=True)
        if not st.session_state.history:
            st.info("Save analyses from the Analysis tab to view history and comparisons.")
        else:
            history_df = pd.DataFrame(
                [
                    {
                        "ID": index + 1,
                        "Timestamp": item["timestamp"],
                        "Classification": item["prediction"],
                        "Confidence": f"{item['confidence']:.2%}",
                        "Detections": item["detections"],
                        "Top finding": item["top_detection"],
                    }
                    for index, item in enumerate(st.session_state.history)
                ]
            )
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            csv = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download history as CSV",
                data=csv,
                file_name="cliniscan_history.csv",
                mime="text/csv",
            )

            if len(st.session_state.history) >= 2:
                labels = [
                    f"#{i + 1} — {item['timestamp']} — {item['prediction']}"
                    for i, item in enumerate(st.session_state.history)
                ]
                comparison_choices = st.multiselect(
                    "Select exactly two analyses to compare",
                    options=labels,
                    default=labels[:2],
                    max_selections=2,
                )
                if len(comparison_choices) == 2:
                    selected = [
                        st.session_state.history[labels.index(choice)]
                        for choice in comparison_choices
                    ]
                    c1, c2 = st.columns(2)
                    for column, item, title in zip((c1, c2), selected, ("Analysis A", "Analysis B")):
                        with column:
                            st.subheader(title)
                            card("Classification", item["prediction"], f"Confidence: {item['confidence']:.1%}")
                            st.write(f"**Detections:** {item['detections']}")
                            st.write(f"**Top finding:** {item['top_detection']}")
                            if item.get("overlay") is not None:
                                st.image(item["overlay"], caption="Attention overlay", use_container_width=True)
                            if item.get("detection_image") is not None:
                                st.image(item["detection_image"], caption="Detection output", use_container_width=True)
            else:
                st.caption("Save one more analysis to enable side-by-side comparison.")

            if st.button("🗑 Clear session history"):
                st.session_state.history = []
                st.rerun()

    if st.button("Logout", use_container_width=False):
        go_to("home")
    show_footer()


# -----------------------------------------------------------------------------
# ROUTER
# -----------------------------------------------------------------------------

apply_css(st.session_state.dark_mode)

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "login":
    login_page()
elif st.session_state.page == "trial":
    trial_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
