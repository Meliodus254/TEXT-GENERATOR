import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generator model definition (same as training)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

# Load model
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("MNIST Digit Generator (Conditional GAN)")
digit = st.selectbox("Select a digit (0–9)", list(range(10)))

if st.button("Generate"):
    noise = torch.randn(5, 100)
    labels = torch.full((5,), digit, dtype=torch.long)
    with torch.no_grad():
        generated = model(noise, labels).detach().numpy()

    st.write("Generated Images:")
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(generated[i][0], cmap="gray", vmin=-1, vmax=1)
        axs[i].axis("off")
    st.pyplot(fig)
