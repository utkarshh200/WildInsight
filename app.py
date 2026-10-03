import streamlit as st
import torch
import numpy as np
import cv2

from torchvision import models, transforms
from PIL import Image

# Import camouflage detector
from camouflage_detector import detect_and_crop


# -----------------------------------
# Page setup
# -----------------------------------

st.set_page_config(
    page_title="WildInsight",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 WildInsight")
st.write("Animal Species Recognition using Deep Learning")

st.divider()


# -----------------------------------
# Classes
# -----------------------------------

classes = ["leopard", "lion", "tiger", "wolf"]


# -----------------------------------
# Load model
# -----------------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        4
    )

    model.load_state_dict(
        torch.load(
            "wildinsight_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


model = load_model()


# -----------------------------------
# Image transformation
# -----------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -----------------------------------
# Upload image
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload an animal image",
    type=["jpg", "jpeg", "png", "webp"]
)


# -----------------------------------
# Grad-CAM function
# -----------------------------------

def generate_gradcam(model, image_tensor, prediction):

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    # Forward pass
    output = model(image_tensor)

    # Clear previous gradients
    model.zero_grad()

    # Backpropagate prediction
    output[0, prediction].backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    activation = activations[0]
    gradient = gradients[0]

    # Calculate importance weights
    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (weights * activation).sum(dim=1)

    cam = torch.relu(cam)

    cam = cam.squeeze()

    # Normalize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam.detach().numpy()


# -----------------------------------
# Prediction
# -----------------------------------

if uploaded_file is not None:

    # -----------------------------------
    # Load original image
    # -----------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # -----------------------------------
    # Animal localization
    # -----------------------------------

    with st.spinner("🔍 Locating animal..."):

        analysis_image, animal_detected, detection_label, detection_confidence = (
            detect_and_crop(image)
        )


    # -----------------------------------
    # Localization status
    # -----------------------------------

    if animal_detected:

        st.success(
            f"🎯 Animal localized "
            f"(localization confidence: "
            f"{detection_confidence * 100:.2f}%)"
    )

        st.image(
            analysis_image,
            caption="Localized animal region used for classification",
            use_container_width=True
        )

    else:

        st.info(
            "🔎 Animal localization was unavailable. "
            "WildInsight is analyzing the full image."
        )


    # -----------------------------------
    # Prepare image for ResNet
    # -----------------------------------

    image_tensor = transform(
        analysis_image
    ).unsqueeze(0)


    # -----------------------------------
    # Prediction
    # -----------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = (
            probabilities[0, prediction].item()
            * 100
        )

    animal = classes[prediction]


    # ===================================
    # IMAGE + PREDICTION
    # ===================================

    left_col, right_col = st.columns(
        [1.2, 0.8],
        gap="large"
    )


    # -------- LEFT: Uploaded image --------

    with left_col:

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )


    # -------- RIGHT: Prediction --------

    with right_col:

        st.subheader("🔎 AI Prediction")

        st.markdown(
            f"## {animal.upper()} 🐾"
        )

        st.write(
            f"**Confidence:** {confidence:.2f}%"
        )


        # Confidence-based result
        if confidence >= 80:

            st.success(
                "✅ Strong prediction — "
                "the model is highly confident."
            )

        elif confidence >= 50:

            st.warning(
                "⚠️ Moderate confidence — "
                "the image has some visual ambiguity."
            )

        else:

            st.error(
                "⚠️ Low confidence — "
                "the model is uncertain about "
                "this prediction."
            )


        # Show alternative only below 80%
        if confidence < 80:

            sorted_indices = torch.argsort(
                probabilities[0],
                descending=True
            )

            alternative_index = (
                sorted_indices[1].item()
            )

            alternative_animal = (
                classes[alternative_index]
            )

            alternative_confidence = (
                probabilities[0][alternative_index].item()
                * 100
            )

            st.info(
                f"🔎 Possible alternative: "
                f"**{alternative_animal.upper()}** "
                f"({alternative_confidence:.2f}%)"
            )


    # ===================================
    # CLASS PROBABILITIES
    # ===================================

    st.divider()

    st.subheader("📊 Class Probabilities")

    for i, animal_name in enumerate(classes):

        probability = (
            probabilities[0, i].item()
            * 100
        )

        st.write(
            f"{animal_name.capitalize()}: "
            f"{probability:.2f}%"
        )

        st.progress(
            min(probability / 100, 1.0)
        )


    # ===================================
    # GRAD-CAM
    # ===================================

    st.divider()

    st.subheader("🔥 AI Focus — Grad-CAM")

    # Generate heatmap
    cam = generate_gradcam(
        model,
        image_tensor,
        prediction
    )


    # Resize heatmap
    cam_image = np.uint8(
        255 * cam
    )

    cam_image = cv2.resize(
        cam_image,
        (
            analysis_image.width,
            analysis_image.height
        )
    )


    # Apply color map
    heatmap = cv2.applyColorMap(
        cam_image,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    # Original analysis image
    original = np.array(
        analysis_image
    )


    # Combine
    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )


    st.image(
        overlay,
        caption="Grad-CAM: Areas influencing the prediction",
        use_container_width=True
    )


    st.info(
        "Brighter regions indicate areas that "
        "contributed more strongly to the model's prediction."
    )