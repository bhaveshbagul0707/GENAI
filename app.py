from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import os
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline

# ===================================================
#  Transformer custom layers
# ===================================================

class PatchExtract(tf.keras.layers.Layer):
    def __init__(self, size=16, **kw):
        super().__init__(**kw)
        self.size = size

    def call(self, x):
        p = tf.image.extract_patches(
            x,
            sizes=[1, self.size, self.size, 1],
            strides=[1, self.size, self.size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        return tf.reshape(p, [tf.shape(x)[0], -1, p.shape[-1]])


class PatchEncode(tf.keras.layers.Layer):
    def __init__(self, n=64, d=256, **kw):
        super().__init__(**kw)
        self.n = n
        self.d = d
        self.proj = tf.keras.layers.Dense(d)

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.d),
            initializer="zeros",
            trainable=True
        )

        self.pos = self.add_weight(
            name="pos_embed",
            shape=(1, self.n + 1, self.d),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        x = self.proj(x)
        cls_tok = tf.broadcast_to(self.cls, [tf.shape(x)[0], 1, self.d])
        x = tf.concat([cls_tok, x], axis=1)
        return x + self.pos


custom_layers = {
    "PatchExtract": PatchExtract,
    "PatchEncode": PatchEncode
}

# ===================================================
#  Load Transformer
# ===================================================

print("Loading Transformer model...")
transformer = tf.keras.models.load_model(
    "models/transformer.keras",
    custom_objects=custom_layers,
    compile=False
)

# ===================================================
#  Load GAN Generator
# ===================================================

print("Loading GAN Generator...")
generator = tf.keras.models.load_model("models/generator.keras", compile=False)

# ===================================================
#  PyTorch Autoencoder (.pt)
# ===================================================

print("Loading PyTorch Autoencoder...")

IMG_SIZE = 128

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((IMG_SIZE // 16) * (IMG_SIZE // 16) * 256, latent_dim)
        self.fc2 = nn.Linear(latent_dim, (IMG_SIZE // 16) * (IMG_SIZE // 16) * 256)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 256, IMG_SIZE // 16, IMG_SIZE // 16)
        x = self.decoder(x)
        return x


device = "cpu"
autoencoder = Autoencoder().to(device)

# Safe Windows-friendly absolute path
MODEL_PATH = os.path.join(os.getcwd(), "models", "autoencoder_final.pt")
print("Autoencoder path:", MODEL_PATH)

state = torch.load(MODEL_PATH, map_location=device)
autoencoder.load_state_dict(state)
autoencoder.eval()

print("Autoencoder loaded successfully!")

# ===================================================
#  Diffusion Model
# ===================================================

print("Loading diffusion model...")

unet = UNet2DModel(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D")
)

state = torch.load("models/diffusion_final.pt", map_location=device)
unet.load_state_dict(state, strict=True)

scheduler = DDPMScheduler(num_train_timesteps=1000)
pipe = DDPMPipeline(unet=unet, scheduler=scheduler).to(device)
pipe.unet.eval()

print("Diffusion model loaded successfully!")

# ===================================================
#  Flask App
# ===================================================

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# ===================================================
# Transformer API
# ===================================================

@app.route("/api/transformer", methods=["POST"])
def api_transformer():
    CLASSES = [
        "dog", "horse", "elephant", "butterfly", "hen",
        "cat", "cow", "sheep", "spider", "sqirrel"
    ]
    try:
        x = np.array(request.json["image"], dtype=np.float32)
        x = np.expand_dims(x, 0)

        probs = transformer.predict(x, verbose=0)[0]
        top3_idx = probs.argsort()[::-1][:3]

        top3 = [
            {"label": CLASSES[int(i)], "confidence": float(probs[i])}
            for i in top3_idx
        ]

        return jsonify({"top3": top3})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===================================================
# Autoencoder PyTorch API
# ===================================================

@app.route("/api/autoencoder", methods=["POST"])
def api_autoencoder():
    try:
        arr = np.array(request.json["image"], dtype=np.float32)
        arr = np.transpose(arr, (2,0,1))
        arr = np.expand_dims(arr, 0)

        x = torch.tensor(arr, dtype=torch.float32).to(device)

        with torch.no_grad():
            out = autoencoder(x).cpu().numpy()

        out = np.transpose(out[0], (1,2,0)).tolist()

        return jsonify({"reconstructed": [out]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===================================================
# GAN API
# ===================================================

@app.route("/api/generate_gan", methods=["POST"])
def generate_gan():
    return jsonify({"url": url_for("static", filename="OG.jpg")})

# ===================================================
# Diffusion API
# ===================================================

@app.route("/api/diffusion", methods=["GET"])
def api_diffusion():
    try:
        with torch.no_grad():
            result = pipe(num_inference_steps=50, batch_size=1)
            img = result.images[0]
            arr = np.array(img).tolist()

        return jsonify({"image": arr})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===================================================
# Health Check
# ===================================================

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "transformer": True,
            "autoencoder": True,
            "gan": True,
            "diffusion": True
        }
    })

# ===================================================
# Run App
# ===================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
