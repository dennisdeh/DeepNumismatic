import streamlit as st
import torch
import torchvision
from PIL import Image
import os
from pathlib import Path
import sys

# Import from main.py
from modules.main import train_cnn, inference, device
from modules.loader import pytorch_loader

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "transformer" not in st.session_state:
    st.session_state.transformer = None
if "label_to_idx" not in st.session_state:
    st.session_state.label_to_idx = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "current_device" not in st.session_state:
    st.session_state.current_device = None

st.title("DeepNumismatic - Numismatic Inference and training app")
st.markdown(
    "This is a numismatic inference and training app using PyTorch and Streamlit "
    "for ancient Roman coins"
)

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["🗂️ Load Model", "🖥️ Train CNN", "📷 Inference"])

# ====================== TAB 1: LOAD MODEL ======================
with tab1:
    st.header("Load Existing Model")

    # Device selection
    st.subheader("Device Configuration")
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.append("cuda")

    selected_device = st.selectbox(
        "Select device for model loading:",
        options=available_devices,
        index=available_devices.index("cuda") if "cuda" in available_devices else 0,
        help="Choose CPU or CUDA. If model was saved on CUDA but you're loading on CPU, select 'cpu'.",
    )

    st.divider()

    models_dir = Path("models")
    if models_dir.exists():
        # Get all subdirectories in models/
        model_folders = [f.name for f in models_dir.iterdir() if f.is_dir()]

        if model_folders:
            selected_folder = st.selectbox(
                "Select a model folder:", options=sorted(model_folders, reverse=True)
            )

            if st.button("Load Model", type="primary"):
                try:
                    model_path = models_dir / selected_folder

                    # Map location based on selected device
                    map_location = torch.device(selected_device)

                    # Load model, transformer, and label_to_idx
                    with st.spinner(f"Loading model to {selected_device}..."):
                        st.session_state.model = torch.load(
                            model_path / "model.pth",
                            map_location=map_location,
                            weights_only=False,
                        )
                        st.session_state.transformer = torch.load(
                            model_path / "transformer.pth",
                            map_location=map_location,
                            weights_only=False,
                        )
                        st.session_state.label_to_idx = torch.load(
                            model_path / "labels_mapping.pth",
                            map_location=map_location,
                            weights_only=False,
                        )

                        # Move model to selected device
                        st.session_state.model = st.session_state.model.to(map_location)
                        st.session_state.model_loaded = True
                        st.session_state.current_device = selected_device

                    st.success(
                        f"✅ Model loaded successfully from `{selected_folder}` to `{selected_device}`!"
                    )
                    st.info(
                        f"**Classes:** {list(st.session_state.label_to_idx.keys())}"
                    )
                    st.info(f"**Device:** {selected_device}")

                except Exception as e:
                    st.error(f"❌ Error loading model: {e}")
                    st.session_state.model_loaded = False
                    import traceback

                    with st.expander("Show detailed error"):
                        st.code(traceback.format_exc())
        else:
            st.warning("No model folders found in `/models/` directory.")
    else:
        st.warning("The `/models/` directory does not exist.")

    # Display current loaded model status
    st.divider()
    if st.session_state.model_loaded:
        st.success(
            f"✅ A model is currently loaded on `{st.session_state.get('current_device', 'unknown')}` and ready for inference."
        )
    else:
        st.info("ℹ️ No model is currently loaded.")

# ====================== TAB 2: TRAIN CNN ======================
with tab2:
    st.header("Train a New CNN Model")

    st.subheader("Data Configuration")
    data_path = st.text_input(
        "Data Path:", value="data/RRC-60/Observe", help="Path to the dataset directory"
    )

    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("Batch Size:", min_value=1, value=32, step=1)
        img_width = st.number_input("Image Width:", min_value=16, value=150, step=1)
        n_channels = st.selectbox("Number of Channels:", options=[1, 3], index=1)

    with col2:
        split_ratio = st.slider(
            "Train/Val Split:", min_value=0.5, max_value=0.95, value=0.8, step=0.05
        )
        img_height = st.number_input("Image Height:", min_value=16, value=150, step=1)

    st.subheader("Training Parameters")
    col3, col4, col5 = st.columns(3)
    with col3:
        num_epochs = st.number_input("Number of Epochs:", min_value=1, value=10, step=1)
    with col4:
        learning_rate = st.number_input(
            "Learning Rate:", min_value=0.0, value=0.001, step=0.0001, format="%.4f"
        )
    with col5:
        print_every = st.number_input(
            "Print Every N Steps:", min_value=1, value=50, step=1
        )

    st.divider()

    if st.button("🚀 Start Training", type="primary"):
        try:
            with st.spinner("Preparing dataset and training model..."):
                # Create transformer
                img_size = (img_height, img_width)
                transformer = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize(size=img_size),
                        torchvision.transforms.CenterCrop(size=img_size),
                        torchvision.transforms.Grayscale(n_channels),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            n_channels * (0.5,), n_channels * (0.5,)
                        ),
                    ]
                )

                # Load dataset
                st.info(f"Loading dataset from `{data_path}`...")
                ds = pytorch_loader(
                    data_path,
                    transformer=transformer,
                    batch_size=batch_size,
                    split=split_ratio,
                )

                # Train model
                st.info("Training started...")
                progress_placeholder = st.empty()

                # Redirect print statements to streamlit
                import io
                import contextlib

                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    result = train_cnn(
                        ds=ds,
                        num_epochs=num_epochs,
                        lr=learning_rate,
                        print_every=print_every,
                    )

                # Display training output
                st.text_area("Training Log:", output.getvalue(), height=300)

                # Save model
                from datetime import datetime

                str_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                path_out = Path(f"models/{str_timestamp}")
                path_out.mkdir(parents=True, exist_ok=True)

                torch.save(result["model"], path_out / "model.pth")
                torch.save(transformer, path_out / "transformer.pth")
                torch.save(result["label_to_idx"], path_out / "labels_mapping.pth")

                import pandas as pd

                pd.DataFrame(result["training info"]).T.to_excel(
                    path_out / "training_info.xlsx"
                )

                st.success(f"✅ Training completed! Model saved to `{path_out}`")

                # Optionally load the trained model
                if st.checkbox("Load this model for inference?", value=True):
                    st.session_state.model = result["model"]
                    st.session_state.transformer = transformer
                    st.session_state.label_to_idx = result["label_to_idx"]
                    st.session_state.model_loaded = True
                    st.success("Model loaded into memory for inference!")

        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            import traceback

            st.text(traceback.format_exc())

# ====================== TAB 3: INFERENCE ======================
with tab3:
    st.header("Image Inference")

    # Check if model is loaded
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a model first from the 'Load Model' tab.")
    else:
        st.success("✅ Model is loaded and ready!")

        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image for inference:", type=["png", "jpg", "jpeg", "bmp"]
        )

        # Probability option
        proba_mode = st.checkbox("Show probabilities for all classes", value=False)

        if uploaded_file is not None:
            # Load image as PIL.Image
            img = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(img, caption="Uploaded Image", use_container_width=True)

            with col2:
                st.subheader("Transformed Image")
                # Apply transformer to show the transformed version
                if st.session_state.transformer is not None:
                    transformed_tensor = st.session_state.transformer(img)
                    # Convert tensor back to displayable image
                    # Denormalize if needed (reverse the normalization)
                    display_tensor = transformed_tensor.clone()

                    # If normalized with mean=0.5, std=0.5, denormalize
                    n_ch = display_tensor.shape[0]
                    for c in range(n_ch):
                        display_tensor[c] = display_tensor[c] * 0.5 + 0.5

                    # Convert to PIL for display
                    display_tensor = torch.clamp(display_tensor, 0, 1)
                    transform_to_pil = torchvision.transforms.ToPILImage()
                    transformed_img = transform_to_pil(display_tensor)
                    st.image(
                        transformed_img,
                        caption="Transformed Image",
                        use_container_width=True,
                    )

            st.divider()

            # Inference button
            if st.button("🔮 Run Inference", type="primary"):
                try:
                    with st.spinner("Running inference..."):
                        prediction = inference(
                            model=st.session_state.model,
                            img=img,
                            transformer=st.session_state.transformer,
                            device=device,
                            label_to_idx=st.session_state.label_to_idx,
                            proba=proba_mode,
                        )

                    st.subheader("Prediction Results")

                    if proba_mode:
                        # Display probabilities
                        st.write("**Class Probabilities:**")

                        # Create a nice table
                        import pandas as pd

                        df = pd.DataFrame(
                            list(prediction.items()), columns=["Class", "Probability"]
                        )
                        df["Probability"] = df["Probability"].apply(
                            lambda x: f"{x:.4f}"
                        )
                        st.dataframe(df, use_container_width=True, hide_index=True)

                        # Show bar chart
                        import pandas as pd

                        chart_df = pd.DataFrame(
                            list(prediction.items()), columns=["Class", "Probability"]
                        )
                        chart_df = chart_df.sort_values("Probability", ascending=False)
                        st.bar_chart(chart_df.set_index("Class"))

                        # Highlight the top prediction
                        top_class = max(prediction.items(), key=lambda x: x[1])
                        st.success(
                            f"**Top Prediction: `{top_class[0]}` with probability `{top_class[1]:.4f}`**"
                        )
                    else:
                        # Display single prediction
                        st.success(f"**Predicted Class:** `{prediction}`")
                        st.balloons()

                except Exception as e:
                    st.error(f"❌ Inference failed: {e}")
                    import traceback

                    st.text(traceback.format_exc())
