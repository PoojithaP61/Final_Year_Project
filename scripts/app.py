import streamlit as st
import torch
from torchvision import models, transforms, datasets
from PIL import Image
import os, json, numpy as np, cv2, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# ======================
# Paths & Config
# ======================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_DIR = os.path.join(ROOT, "data", "train")
INFO_PATH = os.path.join(ROOT, "docs", "disease_treatments.json")
ENCODER_PATH = os.path.join(ROOT, "models", "few_shot_encoder.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Classes & Info
# ======================
class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
if os.path.exists(INFO_PATH):
    with open(INFO_PATH, "r", encoding="utf-8") as f:
        disease_info = json.load(f)
else:
    disease_info = {}

# ======================
# Sidebar model selector
# ======================
MODEL_CHOICES = {
    "ResNet-18": ("resnet18", "agrisense_resnet18.pth"),
    "MobileNetV3-Small": ("mobilenetv3_small", "agrisense_mobilenetv3_small.pth"),
    "EfficientNet-B0": ("efficientnet_b0", "agrisense_efficientnet_b0.pth")
}
st.set_page_config(page_title="LeafCure", page_icon="🌿", layout="wide")
st.sidebar.title("⚙️ Model Settings")
choice = st.sidebar.selectbox("Choose backbone", list(MODEL_CHOICES.keys()))
arch, weight_file = MODEL_CHOICES[choice]

# ======================
# Load main CNN model
# ======================
@st.cache_resource
def load_model(arch, weight_file):
    num_classes = len(class_names)
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenetv3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
    path = os.path.join(ROOT, "models", weight_file)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval().to(DEVICE)
    return m

model = load_model(arch, weight_file)

# ======================
# Grad-CAM helper
# ======================
def generate_gradcam(model, img_tensor, target_class):
    model.eval()
    grads, acts = [], []
    if arch == "resnet18": last_conv = model.layer4[-1].conv2
    elif arch == "mobilenetv3_small": last_conv = model.features[-1][0]
    else: last_conv = model.features[-1][0]

    def fh(_, __, out): acts.append(out.detach())
    def bh(_, gin, gout): grads.append(gout[0].detach())
    h1 = last_conv.register_forward_hook(fh)
    h2 = last_conv.register_full_backward_hook(bh)
    out = model(img_tensor); model.zero_grad(); out[0, target_class].backward()
    g, a = grads[0].cpu().numpy()[0], acts[0].cpu().numpy()[0]
    w = np.mean(g, axis=(1,2)); cam = np.maximum(np.sum(w[:,None,None]*a,0),0)
    cam = cv2.resize(cam,(224,224)); cam=(cam-cam.min())/(cam.max()+1e-8)
    h1.remove(); h2.remove()
    return cam

def overlay_heatmap(img_pil, heatmap):
    img=np.array(img_pil); h=np.uint8(255*heatmap)
    h=cv2.applyColorMap(h,cv2.COLORMAP_JET); h=cv2.cvtColor(h,cv2.COLOR_BGR2RGB)
    out=cv2.addWeighted(img,0.6,h,0.4,0); return Image.fromarray(out)

# ======================
# Transforms
# ======================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ======================
# Encoder Loader
# ======================
@st.cache_resource
def load_encoder():
    enc = models.mobilenet_v3_small(weights=None)
    enc.classifier = torch.nn.Identity()
    enc.load_state_dict(torch.load(ENCODER_PATH, map_location=DEVICE))
    enc.eval().to(DEVICE)
    return enc

# ======================
# Tabs
# ======================
tab1, tab2, tab3 = st.tabs(["🔍 Disease Detection", "🧬 Few-Shot Learning", "📊 Embedding Visualization"])

# -------------------------------------------------------------------
# Tab 1: Disease Detection
# -------------------------------------------------------------------
with tab1:
    st.title("🌿 LeafCure — Plant Disease Detection")
    uploaded = st.file_uploader("📸 Upload a leaf image", type=["jpg","jpeg","png"], key="det")
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            probs = torch.nn.functional.softmax(out, dim=1)[0]
            conf, idx = torch.max(probs, 0)
            pred_class = class_names[idx]
        st.success(f"✅ **Predicted:** {pred_class} | **Confidence:** {conf.item()*100:.2f}%")
        try:
            cam = generate_gradcam(model, x, idx.item())
            overlay = overlay_heatmap(img.resize((224,224)), cam)
        except Exception as e:
            overlay=None; st.warning(f"Grad-CAM failed: {e}")

        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("**🖼️ Uploaded Image**")
            st.image(img.resize((300, 300)))
        with col2:
            st.markdown("**🔥 Grad-CAM Heatmap**")
            st.image(overlay.resize((300, 300)) if overlay else None)


        st.markdown("### 🩺 Disease Info")
        info = disease_info.get(pred_class)
        if info:
            st.markdown(f"- **Cause:** {info.get('cause','-')}  \n- **Symptoms:** {info.get('symptoms','-')}  \n- **Treatment:** {info.get('treatment','-')}")
        else: st.info("No detailed info available for this disease.")

# -------------------------------------------------------------------
# Tab 2: Few-Shot Learning
# -------------------------------------------------------------------
with tab2:
    st.header("🧬 Few-Shot Learning — Add New Disease Class")
    st.write("Upload a few labeled images (5–10) of a **new disease**, then test it on a query leaf image.")
    support_imgs = st.file_uploader("📁 Upload support images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    new_class_name = st.text_input("🧾 Enter new disease name (e.g., Tomato_New_Virus)")
    query_img = st.file_uploader("🔍 Upload query image", type=["jpg","jpeg","png"])
    if st.button("🚀 Run Few-Shot Prediction"):
        if not support_imgs or not new_class_name or not query_img:
            st.warning("Please upload support images, enter class name, and upload query image.")
        else:
            st.info("Computing embeddings ...")
            encoder = load_encoder()
            def get_emb(pil): 
                t=transform(pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): e=encoder(t)
                return e.squeeze().cpu().numpy()
            support_embs=np.stack([get_emb(Image.open(f).convert("RGB")) for f in support_imgs])
            query_emb=get_emb(Image.open(query_img).convert("RGB"))
            sims=np.dot(support_embs,query_emb)/(np.linalg.norm(support_embs,axis=1)*np.linalg.norm(query_emb))
            st.success(f"🌱 Predicted (Few-Shot): **{new_class_name}** | Similarity: {np.max(sims):.3f}")
            c1,c2=st.columns(2)
            with c1: st.image(query_img,caption="Query Image",use_container_width=True)
            with c2: st.image(support_imgs[0],caption=f"Example Support: {new_class_name}",use_container_width=True)

# -------------------------------------------------------------------
# Tab 3: t-SNE Visualization
# -------------------------------------------------------------------
with tab3:
    st.header("📊 Embedding Visualization (t-SNE)")
    st.write("Visualize how your trained few-shot encoder clusters disease classes in embedding space.")
    if st.button("🎨 Generate t-SNE Visualization"):
        with st.spinner("Extracting embeddings and running t-SNE..."):
            encoder=load_encoder()
            dataset=datasets.ImageFolder(DATA_DIR,transform=transform)
            loader=torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False)
            class_names=dataset.classes
            embeddings,labels=[],[]
            with torch.no_grad():
                for imgs,lbls in tqdm(loader,desc="Embedding extraction",leave=False):
                    imgs=imgs.to(DEVICE)
                    feats=encoder(imgs)
                    feats=torch.flatten(feats,1)
                    embeddings.append(feats.cpu().numpy())
                    labels.append(lbls.numpy())
            embeddings=np.concatenate(embeddings); labels=np.concatenate(labels)
            tsne=TSNE(n_components=2,init='random',perplexity=30,random_state=42)
            emb2d=tsne.fit_transform(embeddings)
            fig,ax=plt.subplots(figsize=(8,6))
            for i,cls in enumerate(class_names):
                idx=np.where(labels==i)
                ax.scatter(emb2d[idx,0],emb2d[idx,1],s=12,label=cls)
            ax.legend(fontsize=7,ncol=2)
            ax.set_title("t-SNE Visualization of L-SC-FSL Encoder")
            st.pyplot(fig,use_container_width=True)
            os.makedirs(os.path.join(ROOT,"results"),exist_ok=True)
            save_path=os.path.join(ROOT,"results","tsne_embeddings.png")
            fig.savefig(save_path,dpi=300)
            st.success(f"📁 Saved t-SNE visualization to: {save_path}")
