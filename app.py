import streamlit as st
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import io
import base64

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the function to load and preprocess the images
def load_image(image, max_size=512):
    image = Image.open(image).convert('RGB')
    size = max(max(image.size), max_size) if max(image.size) > max_size else max(image.size)
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    image = transform(image)[:3, :, :].unsqueeze(0)  # Ensure the image has 3 channels and add batch dimension
    return image.to(device)

# Convert tensor to image for display and download
def im_convert(tensor):
    image = tensor.cpu().clone().detach().squeeze(0)  # Remove batch dimension
    image = image.numpy().transpose(1, 2, 0)  # Convert to HWC format
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))  # Denormalize
    image = np.clip(image, 0, 1)  # Clip to valid pixel range
    return image

# Define the VGG model class
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_layers = ['0', '5', '10', '19', '28']  # Layers for content and style extraction
        self.model = models.vgg19(pretrained=True).features[:29].to(device).eval()

    def forward(self, x):
        features = []
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.chosen_layers:
                features.append(x)
        return features

# Function to calculate the Gram matrix for style loss
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Function to compute content and style loss
def get_total_loss(content_features, style_features, generated_features, content_weight, style_weight):
    content_loss = torch.mean((generated_features[3] - content_features[3]) ** 2)  # Content loss
    style_loss = 0
    for gf, sf in zip(generated_features, style_features):
        gf_gram = gram_matrix(gf)
        sf_gram = gram_matrix(sf)
        style_loss += torch.mean((gf_gram - sf_gram) ** 2)  # Style loss
    return content_weight * content_loss + style_weight * style_loss

# Function to run neural style transfer
def run_style_transfer(content_image, style_image, content_weight=1e5, style_weight=1e10, steps=300, lr=0.003):
    # Load the VGG model
    vgg = VGG().to(device)
    
    # Extract content and style features
    content_features = vgg(content_image)
    style_features = vgg(style_image)
    
    # Initialize the generated image as a clone of the content image
    generated_image = content_image.clone().requires_grad_(True)
    
    # Define the optimizer
    optimizer = optim.Adam([generated_image], lr=lr)
    
    # Training loop for neural style transfer
    for step in range(steps):
        optimizer.zero_grad()
        generated_features = vgg(generated_image)
        
        # Compute total loss
        total_loss = get_total_loss(content_features, style_features, generated_features, content_weight, style_weight)
        
        total_loss.backward()
        optimizer.step()
        
        # Print progress every 100 steps
        if (step + 1) % 100 == 0:
            st.write(f'Step [{step+1}/{steps}], Total Loss: {total_loss.item()}')

    return generated_image

# Streamlit app layout
st.title("Neural Style Transfer App")

# Sidebar for user inputs
st.sidebar.header("Upload Images")
content_image_file = st.sidebar.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])
style_image_file = st.sidebar.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])

if content_image_file is not None and style_image_file is not None:
    # Display content and style images
    st.sidebar.image(content_image_file, caption="Content Image", use_column_width=True)
    st.sidebar.image(style_image_file, caption="Style Image", use_column_width=True)
    
    # Process images and run style transfer
    content_image = load_image(content_image_file)
    style_image = load_image(style_image_file)

    st.write("Running Neural Style Transfer...")
    generated_image_tensor = run_style_transfer(content_image, style_image, steps=1000)

    # Convert tensor to image for display
    generated_image = im_convert(generated_image_tensor)
    
    # Display the generated image
    st.image(generated_image, caption="Generated Image", use_column_width=True)
    
    # Function to download the generated image
    def get_image_download_link(img, filename, text):
        buffered = io.BytesIO()
        img_pil = Image.fromarray((img * 255).astype('uint8'))
        img_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
        return href
    
    st.markdown(get_image_download_link(generated_image, "generated_image.jpg", "Download Generated Image"), unsafe_allow_html=True)
