import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from ultralytics import YOLO
import timm
import os
import urllib.request

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="🩻 CliniScan - Lung Detection",
    layout="wide"
)

# --------------------------------------------------
# NAVIGATION STATE
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to(page):
    st.session_state.page = page
    st.rerun()


# --------------------------------------------------
# HEADER (TITLE + SIDEBAR)
# --------------------------------------------------

def show_header():

    st.title("🩻 CliniScan: AI-Powered Lung Abnormality Detection")

    st.markdown("""
    Upload a **Chest X-ray** image to:

    - 🎯 Detect **14 lung abnormalities** with bounding boxes (YOLOv8-M, mAP: 0.4305)
    - 📊 Get **overall classification**: Abnormal vs Normal (EfficientNet-B3, Acc: 95.20%)
    - 🧠 View **Grad-CAM heatmap** showing model focus areas

    **Note**: Classification trained on 512×512 images, optimized for chest X-ray analysis.
    """)

    with st.sidebar:

        st.header("ℹ️ About CliniScan")

        st.markdown("""
        **14 Detectable Abnormalities**

        1. Aortic enlargement  
        2. Atelectasis  
        3. Calcification  
        4. Cardiomegaly  
        5. Consolidation  
        6. ILD  
        7. Infiltration  
        8. Lung Opacity  
        9. Nodule/Mass  
        10. Other lesion  
        11. Pleural effusion  
        12. Pleural thickening  
        13. Pneumothorax  
        14. Pulmonary fibrosis  

        **Classification Classes**

        - Abnormal (Class 0)
        - Normal (Class 1)

        **⚠️ Disclaimer**: Educational purposes only.
        """)

        st.markdown("---")
        st.markdown("**Developer**: Vasu Chakravarthi")
        st.markdown("**Institution**: SRKR Engineering College")

        st.markdown(
            "[GitHub Repository](https://github.com/vasuchakravarthi/cliniscan-lung-detection)"
        )


# --------------------------------------------------
# FOOTER (DISCLAIMER)
# --------------------------------------------------

def show_footer():

    st.markdown("---")

    st.markdown("""
    <div style='text-align: center; color: gray;'>

    <p><strong>⚠️ DISCLAIMER</strong></p>

    <p>This system is for <strong>educational and research purposes only</strong>.</p>

    <p>It should NOT be used for clinical diagnosis or medical decision-making.</p>

    <p>Always consult a qualified radiologist for medical interpretation of chest X-rays.</p>

    <hr>

    <p><strong>Vasu Chakravarthi</strong> | SRKR Engineering College | BTech AIML 2025</p>

    <p>
    <a href='https://github.com/vasuchakravarthi/cliniscan-lung-detection'>
    GitHub Repository
    </a>
    </p>

    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# MODEL URLS
# --------------------------------------------------

DET_URL = (
    "https://huggingface.co/vasuchakravarthi/cliniscan-models/"
    "resolve/main/best1.pt?download=true"
)

CLF_URL = (
    "https://huggingface.co/vasuchakravarthi/cliniscan-models/"
    "resolve/main/best_clf_model.pth?download=true"
)


# --------------------------------------------------
# DOWNLOAD MODELS
# --------------------------------------------------

@st.cache_resource
def download_models():
    os.makedirs("models", exist_ok=True)

    det_path = "models/best.pt"
    clf_path = "models/best_clf_model.pth"

    try:
        if not os.path.exists(det_path):
            urllib.request.urlretrieve(DET_URL, det_path)

        if not os.path.exists(clf_path):
            urllib.request.urlretrieve(CLF_URL, clf_path)
    except Exception as exc:
        raise RuntimeError(
            f"Model download failed. Check the Hugging Face URLs. Details: {exc}"
        ) from exc

    if not os.path.exists(det_path) or os.path.getsize(det_path) == 0:
        raise RuntimeError(f"Detection model is missing or empty: {det_path}")

    if not os.path.exists(clf_path) or os.path.getsize(clf_path) == 0:
        raise RuntimeError(f"Classification model is missing or empty: {clf_path}")

    return det_path, clf_path


# --------------------------------------------------
# CLASSIFICATION MODEL
# --------------------------------------------------

class EfficientNetClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=2
        )

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def load_models():
    det_path, clf_path = download_models()

    # Classification model
    classifier = EfficientNetClassifier()

    checkpoint = torch.load(
        clf_path,
        map_location="cpu",
        weights_only=False
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove possible 'module.' prefix from keys
    state_dict = {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
    }

    classifier.load_state_dict(state_dict, strict=True)
    classifier.eval()

    # Detection model
    detector = YOLO(det_path)

    return classifier, detector


# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# GRAD-CAM IMPLEMENTATION
# --------------------------------------------------

def find_last_conv_layer(model):
    """
    Find the last convolutional layer in a timm EfficientNet-like model.
    """
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d)):
            last_conv = (name, module)
    return last_conv


def compute_gradcam(model, img_tensor, target_class):
    """
    Compute Grad-CAM for a given image tensor and target class.
    model: EfficientNetClassifier (eval mode)
    img_tensor: (C, H, W) tensor, already normalized
    target_class: 0 or 1
    """
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).requires_grad_(True)  # (1, C, H, W)

    last_conv_name, last_conv_module = find_last_conv_layer(model.model)
    if last_conv_name is None:
        raise RuntimeError("No convolutional layer found for Grad-CAM.")

    fmap = None

    def hook_fn(module, inp, out):
        nonlocal fmap
        fmap = out.detach()

    hook = last_conv_module.register_forward_hook(hook_fn)

    try:
        # Forward pass
        logits = model(img_tensor)  # (1, 2)
        probs = torch.nn.functional.softmax(logits, dim=1)

        # Backward pass for target class
        model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward()

        # Get gradients w.r.t. feature maps
        grads = img_tensor.grad  # not used directly
        # We need gradients of the feature maps, not input
        # So we access grads via last_conv_module's output
        # Instead, we recompute using a custom hook on gradients
    finally:
        hook.remove()

    # Better approach: use a hook that captures both output and gradient
    fmap = None
    grad = None

    def forward_hook(module, inp, out):
        nonlocal fmap
        fmap = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal grad
        grad = grad_out[0].detach()

    hook_f = last_conv_module.register_forward_hook(forward_hook)
    hook_b = last_conv_module.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()
        img_tensor = img_tensor.detach().requires_grad_(True)
        logits = model(img_tensor)
        target_score = logits[0, target_class]
        target_score.backward()
    finally:
        hook_f.remove()
        hook_b.remove()

    if fmap is None or grad is None:
        raise RuntimeError("Grad-CAM feature map or gradient is None.")

    # fmap: (1, C, H_f, W_f)
    # grad: (1, C, H_f, W_f)
    pooled_grad = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    fmap = fmap[0]  # (C, H_f, W_f)
    pooled_grad = pooled_grad[0]  # (C, 1, 1)

    weighted_map = torch.sum(fmap * pooled_grad.squeeze(), dim=0)  # (H_f, W_f)

    heatmap = weighted_map.cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    # Resize to 512x512
    heatmap = cv2.resize(heatmap, (512, 512))

    return heatmap


def apply_gradcam_overlay(image_pil, heatmap):
    """
    image_pil: original PIL image
    heatmap: 2D numpy array (512, 512), values in [0, 1]
    """
    original = np.array(image_pil.resize((512, 512)))

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        original.astype("float32"),
        0.6,
        heatmap_color.astype("float32"),
        0.4,
        0
    )
    overlay = np.clip(overlay, 0, 255).astype("uint8")
    return overlay


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

def home_page():

    show_header()

    st.subheader("🏠 Welcome to CliniScan")

    st.write("""
    **CliniScan** is an AI-powered chest X-ray analysis system designed to assist
    in detecting lung abnormalities.

    The system integrates:

    • YOLOv8 detection for lung abnormalities  
    • EfficientNet classification  
    • Grad-CAM explainability
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔐 Login"):
            go_to("login")

    with col2:
        if st.button("🧪 Free Trial"):
            go_to("trial")

    show_footer()


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

def login_page():

    show_header()

    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "cliniscan":
            go_to("dashboard")
        else:
            st.error("Invalid credentials")

    if st.button("⬅ Back"):
        go_to("home")

    show_footer()


# --------------------------------------------------
# TRIAL PAGE
# --------------------------------------------------

def trial_page():

    show_header()

    st.subheader("🧪 Free Trial")

    st.info("Upload a chest X-ray to test the AI system.")

    if st.button("Start Trial"):
        go_to("dashboard")

    if st.button("⬅ Back"):
        go_to("home")

    show_footer()


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

def dashboard_page():

    show_header()

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        img_tensor = transform(image)

        with st.spinner("Loading AI models (first time may take a minute)..."):
            clf_model, det_model = load_models()

        col1, col2 = st.columns(2)

        # Classification

        with col1:

            st.subheader("Classification")

            with torch.no_grad():

                preds = clf_model(img_tensor.unsqueeze(0))
                probs = torch.nn.functional.softmax(preds, dim=1)

                pred_class = torch.argmax(probs).item()

            classes = ["Abnormal", "Normal"]

            st.write("Prediction:", classes[pred_class])
            st.write("Confidence:", f"{probs[0][pred_class]:.2%}")

            st.subheader("Grad-CAM")

            with st.spinner("Computing Grad-CAM..."):
                try:
                    heatmap = compute_gradcam(
                        clf_model,
                        img_tensor,
                        target_class=pred_class
                    )
                    overlay = apply_gradcam_overlay(image, heatmap)
                    st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")
                    st.info(
                        "Classification and detection still work; Grad-CAM is optional."
                    )

        # Detection

        with col2:

            st.subheader("Detection")

            results = det_model.predict(
                source=np.array(image),
                conf=0.25,
                verbose=False
            )

            res_img = results[0].plot()

            st.image(res_img, use_container_width=True)

            if results[0].boxes is not None:

                boxes = results[0].boxes

                st.write("Total detections:", len(boxes))

                for i in range(len(boxes)):

                    cls = int(boxes.cls[i])
                    conf = float(boxes.conf[i])

                    st.write(
                        f"{det_model.names[cls]} - {conf:.2%}"
                    )

    if st.button("Logout"):
        go_to("home")

    show_footer()


# --------------------------------------------------
# ROUTER
# --------------------------------------------------

if st.session_state.page == "home":
    home_page()

elif st.session_state.page == "login":
    login_page()


elif st.session_state.page == "trial":
    trial_page()

elif st.session_state.page == "dashboard":
    dashboard_page()
