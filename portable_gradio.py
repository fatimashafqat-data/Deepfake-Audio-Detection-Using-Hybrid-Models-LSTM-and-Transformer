# %% COMPLETE PORTABLE GRADIO APP - DEEPFAKE DETECTOR (FIXED)

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
import os
import warnings
import sys
import importlib.util

warnings.filterwarnings("ignore")

# Configuration
SAVED_MODELS_DIR = Path("saved_models")
TARGET_SR = 16000
dev = "cuda" if torch.cuda.is_available() else "cpu"

# PyTorch 2.6+ compatibility fix
try:
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
except:
    pass

# Exact model architecture from training
class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(128, 128, 1, batch_first=True, bidirectional=True)
        self.att = nn.MultiheadAttention(256, 4, batch_first=True)
        self.fc  = nn.Linear(256, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out, _ = self.att(out, out, out)
        out = out.mean(1)
        return torch.sigmoid(self.fc(out))

def setup_vggish():
    """Setup VGGish for exact feature extraction if available"""
    try:
        vggish_dir = Path("models/vggish_code")
        if not vggish_dir.exists():
            print("VGGish code directory not found, using fallback features")
            return None, None, None, None
            
        vggish_path = vggish_dir / "research" / "audioset" / "vggish"
        if not vggish_path.exists():
            vggish_path = vggish_dir
            
        def load_vggish_module(module_name, file_path):
            if not file_path.exists():
                return None
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        
        mel_features = load_vggish_module("mel_features", vggish_path / "mel_features.py")
        vggish_params = load_vggish_module("vggish_params", vggish_path / "vggish_params.py")
        vggish_input = load_vggish_module("vggish_input", vggish_path / "vggish_input.py")
        vggish_slim = load_vggish_module("vggish_slim", vggish_path / "vggish_slim.py")
        
        if all([mel_features, vggish_params, vggish_input, vggish_slim]):
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            return vggish_input, vggish_params, vggish_slim, tf
        else:
            return None, None, None, None
            
    except Exception as e:
        print(f"VGGish setup failed: {e}")
        return None, None, None, None

def extract_vggish_features(audio_path):
    """Extract VGGish features if available, otherwise use librosa fallback"""
    vggish_input, vggish_params, vggish_slim, tf = setup_vggish()
    
    if vggish_input is not None:
        try:
            examples = vggish_input.wavfile_to_examples(str(audio_path))
            if examples.size > 0:
                ckpt_path = Path("models/vggish.ckpt")
                if ckpt_path.exists():
                    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
                        vggish_slim.define_vggish_slim(training=False)
                        tf.compat.v1.train.Saver().restore(sess, str(ckpt_path))
                        
                        feat = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
                        emb = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
                        
                        features = sess.run(emb, {feat: examples}).mean(0)
                        print("Using VGGish feature extraction")
                        return features.reshape(1, -1).astype(np.float32)
        except Exception as e:
            print(f"VGGish extraction failed: {e}, using fallback")
    
    print("Using librosa fallback feature extraction")
    return extract_librosa_features(audio_path)

def extract_librosa_features(audio_path):
    """Extract librosa features exactly as used in training fallback"""
    try:
        import librosa
        
        y, sr = librosa.load(audio_path, sr=TARGET_SR)
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        features = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(zero_crossing), np.std(zero_crossing),
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
        ])
        
        if len(features) < 128:
            features = np.pad(features, (0, 128 - len(features)))
        else:
            features = features[:128]
            
        return features.reshape(1, -1).astype(np.float32)
        
    except Exception as e:
        print(f"Librosa feature extraction failed: {e}")
        return np.zeros((1, 128), dtype=np.float32)

def load_models():
    """Load saved models with PyTorch 2.6+ compatibility"""
    try:
        with open(SAVED_MODELS_DIR / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"Loading model: {metadata['model_info']['name']}")
        print(f"Expected accuracy: {metadata['performance']['accuracy']*100:.1f}%")
        
        with open(SAVED_MODELS_DIR / "scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully")
        
        try:
            model_data = torch.load(
                SAVED_MODELS_DIR / "hybrid_model.pth", 
                map_location=dev,
                weights_only=False
            )
        except TypeError:
            model_data = torch.load(
                SAVED_MODELS_DIR / "hybrid_model.pth", 
                map_location=dev
            )
        
        model = Hybrid()
        model.load_state_dict(model_data['model_state_dict'])
        model = model.to(dev)
        model.eval()
        print(f"Model loaded successfully on {dev}")
        
        verification_passed = True
        if (SAVED_MODELS_DIR / "test_verification.npz").exists():
            test_data = np.load(SAVED_MODELS_DIR / "test_verification.npz")
            test_scaled = test_data['scaled_features'][:1]
            expected_pred = test_data['predictions'][0]
            
            features_padded = np.zeros((1, 10, 128), dtype=np.float32)
            features_padded[:, 0, :] = test_scaled
            test_tensor = torch.tensor(features_padded).to(dev)
            
            with torch.no_grad():
                test_pred = model(test_tensor).cpu().numpy()[0][0]
            
            pred_diff = abs(test_pred - expected_pred)
            print(f"Model verification: {test_pred:.6f} vs expected {expected_pred:.6f}")
            print(f"Difference: {pred_diff:.8f}")
            
            if pred_diff > 1e-3:  # Relaxed threshold for CPU vs GPU differences
                print("WARNING: Model predictions don't match training!")
                verification_passed = False
            else:
                print("Model verification: PASSED")
        
        metadata['verification_passed'] = verification_passed
        return scaler, model, metadata
        
    except Exception as e:
        raise Exception(f"Failed to load models: {e}")

def convert_audio_to_wav(input_path):
    """Convert audio to WAV format"""
    output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    
    try:
        cmd = ["ffmpeg", "-nostats", "-loglevel", "error", "-y",
               "-i", str(input_path), "-ac", "1", "-ar", str(TARGET_SR), output_path]
        subprocess.run(cmd, check=True, timeout=30)
        return output_path
    except:
        try:
            import librosa
            import soundfile as sf
            y, sr = librosa.load(input_path, sr=TARGET_SR)
            sf.write(output_path, y, TARGET_SR)
            return output_path
        except Exception as e:
            print(f"Audio conversion failed: {e}")
            return None

def predict_deepfake(audio_file):
    """Main prediction function with exact feature matching"""
    if audio_file is None:
        return "Please upload an audio file to analyze!", None, None
    
    try:
        print(f"Processing audio: {Path(audio_file).name}")
        
        wav_path = convert_audio_to_wav(audio_file)
        if wav_path is None:
            return "Failed to process audio file. Please check the format!", None, None
        
        features = extract_vggish_features(wav_path)
        os.unlink(wav_path)
        
        print(f"Features extracted: shape={features.shape}, mean={features.mean():.4f}")
        
        features_scaled = scaler.transform(features)
        print(f"Features scaled: mean={features_scaled.mean():.4f}, std={features_scaled.std():.4f}")
        
        features_padded = np.zeros((1, 10, 128), dtype=np.float32)
        features_padded[:, 0, :] = features_scaled
        X_tensor = torch.tensor(features_padded).to(dev)
        
        model.eval()
        with torch.no_grad():
            prediction_prob = model(X_tensor).cpu().numpy()[0][0]
        
        print(f"Raw prediction: {prediction_prob:.6f}")
        
        confidence_percentage = prediction_prob * 100
        is_fake = prediction_prob > 0.5
        
        if is_fake:
            status_color = "#1976d2"
            status_bg = "linear-gradient(135deg, #1976d2 0%, #2196f3 100%)"
            status_text = "DEEPFAKE DETECTED"
            risk_level = "HIGH RISK" if confidence_percentage > 80 else "MEDIUM RISK"
            recommendation = "This audio appears to be artificially generated! Exercise caution"
            emoji = "🔴"
        else:
            status_color = "#1565c0"
            status_bg = "linear-gradient(135deg, #1565c0 0%, #1976d2 100%)"
            status_text = "AUTHENTIC AUDIO"
            risk_level = "SAFE" if confidence_percentage < 20 else "LOW RISK"
            recommendation = "This audio appears to be genuine and unmodified!"
            emoji = "🟢"
        
        results_html = f"""
        <div style="background: white; border-radius: 16px; padding: 30px; 
                   box-shadow: 0 8px 32px rgba(0,0,0,0.1); border: 2px solid #e3f2fd; margin: 10px 0;">
            
            <div style="text-align: center; margin-bottom: 30px;">
                <div style="background: {status_bg}; color: white; padding: 20px 40px; border-radius: 12px; 
                           display: inline-block; font-weight: bold; font-size: 22px; margin-bottom: 15px;
                           box-shadow: 0 4px 20px rgba(25,118,210,0.3);">
                    {emoji} {status_text}
                </div>
                <div style="color: #333333; font-size: 16px; line-height: 1.6;">{recommendation}</div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%); 
                           padding: 25px; border-radius: 12px; text-align: center;
                           border: 2px solid #bbdefb; box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
                    <div style="font-size: 28px; font-weight: bold; color: {status_color}; margin-bottom: 8px;">
                        {confidence_percentage:.1f}%
                    </div>
                    <div style="color: #666666; font-size: 12px; font-weight: 500;">Fake Probability</div>
                </div>
                
                <div style="background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%); 
                           padding: 25px; border-radius: 12px; text-align: center;
                           border: 2px solid #bbdefb; box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
                    <div style="font-size: 18px; font-weight: bold; color: #333333; margin-bottom: 8px;">
                        {risk_level}
                    </div>
                    <div style="color: #666666; font-size: 12px; font-weight: 500;">Risk Assessment</div>
                </div>
                
                <div style="background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%); 
                           padding: 25px; border-radius: 12px; text-align: center;
                           border: 2px solid #bbdefb; box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
                    <div style="font-size: 18px; font-weight: bold; color: #333333; margin-bottom: 8px;">
                        {metadata['performance']['accuracy']*100:.1f}%
                    </div>
                    <div style="color: #666666; font-size: 12px; font-weight: 500;">Model Accuracy</div>
                </div>
                
                <div style="background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%); 
                           padding: 25px; border-radius: 12px; text-align: center;
                           border: 2px solid #bbdefb; box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
                    <div style="font-size: 18px; font-weight: bold; color: #333333; margin-bottom: 8px;">
                        {prediction_prob:.4f}
                    </div>
                    <div style="color: #666666; font-size: 12px; font-weight: 500;">Raw Score</div>
                </div>
            </div>
        </div>
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('white')
        
        colors = [status_color, '#e3f2fd']
        sizes = [confidence_percentage, 100-confidence_percentage]
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=['Fake', 'Real'], 
                                          colors=colors, startangle=90,
                                          autopct='%1.1f%%', pctdistance=0.85,
                                          wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
                                          textprops={'color': '#333333', 'fontweight': 'bold', 'fontsize': 12})
        
        ax1.text(0, 0, f'{confidence_percentage:.1f}%\nFake', ha='center', va='center', 
                fontsize=14, fontweight='bold', color='#333333')
        ax1.set_title('Detection Result', fontweight='bold', pad=20, color='#333333', fontsize=16)
        
        bars = ax2.barh(['Model Confidence'], [confidence_percentage], 
                       color=status_color, alpha=0.8, height=0.5,
                       edgecolor='#333333', linewidth=1)
        ax2.axvline(x=50, color='#666666', linestyle='--', alpha=0.7, linewidth=2)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Fake Probability (%)', color='#333333', fontweight='bold', fontsize=12)
        ax2.set_title('Confidence Distribution', fontweight='bold', color='#333333', fontsize=16)
        ax2.grid(axis='x', alpha=0.3, color='#cccccc')
        ax2.set_facecolor('white')
        ax2.tick_params(colors='#333333')
        
        ax2.text(bars[0].get_width() + 2, bars[0].get_y() + bars[0].get_height()/2, 
                f'{confidence_percentage:.1f}%', va='center', fontweight='bold', 
                color='#333333', fontsize=12)
        
        ax2.axvspan(0, 50, alpha=0.1, color='#1565c0', label='Authentic Range')
        ax2.axvspan(50, 100, alpha=0.1, color='#1976d2', label='Fake Range')
        
        legend = ax2.legend(loc='upper right', frameon=True, fancybox=True, 
                           shadow=True, facecolor='white', edgecolor='#cccccc')
        for text in legend.get_texts():
            text.set_color('#333333')
        
        plt.tight_layout()
        
        stats_df = pd.DataFrame({
            'Metric': ['Prediction', 'Confidence', 'Raw Score', 'Risk Level', 'Model'],
            'Value': [
                f'{emoji} {"Fake Audio" if is_fake else "Real Audio"}',
                f'{confidence_percentage:.1f}% fake probability',
                f'{prediction_prob:.6f}',
                f'{risk_level}',
                f'{metadata["model_info"]["name"]}'
            ]
        })
        
        return results_html, fig, stats_df
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_html = f"""
        <div style="background: white; border: 2px solid #f44336; color: #d32f2f; 
                   padding: 30px; border-radius: 12px; text-align: center;">
            <h3 style="margin: 0 0 12px 0;">Processing Error</h3>
            <p style="margin: 0; color: #666666;">{str(e)}</p>
        </div>
        """
        return error_html, None, None

# Load models at startup
try:
    scaler, model, metadata = load_models()
    models_loaded = True
    status_msg = f"Models loaded successfully! Accuracy: {metadata['performance']['accuracy']*100:.1f}% | Verification: {'PASSED' if metadata.get('verification_passed', False) else 'FAILED'}"
except Exception as e:
    models_loaded = False
    status_msg = f"Models not found: {str(e)}"
    scaler, model, metadata = None, None, {'model_info': {'name': 'N/A'}, 'performance': {'accuracy': 0}}

# CSS styling
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    max-width: 1200px !important; 
    margin: 0 auto !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    min-height: 100vh !important; 
    padding: 20px !important;
}

.main-header {
    background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
    color: white; 
    padding: 40px; 
    border-radius: 20px; 
    text-align: center;
    box-shadow: 0 12px 40px rgba(21,101,192,0.3); 
    margin-bottom: 30px;
}

.compact-section {
    background: white; 
    border-radius: 16px; 
    padding: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
    border: 2px solid #e3f2fd;
    margin: 25px 0;
}

.gr-button {
    background: linear-gradient(135deg, #1976d2 0%, #2196f3 100%) !important;
    border: 2px solid #1565c0 !important; 
    border-radius: 12px !important;
    padding: 16px 32px !important; 
    font-weight: 600 !important;
    color: white !important; 
    font-size: 16px !important;
    box-shadow: 0 6px 20px rgba(25,118,210,0.3) !important;
}

.gr-file-input {
    border: 3px dashed #2196f3 !important; 
    border-radius: 12px !important;
    background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%) !important;
    padding: 30px !important; 
    color: #333333 !important;
}
"""

# Create interface
with gr.Blocks(title="Portable Audio Deepfake Detector", css=css) as app:
    
    gr.HTML(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 36px; font-weight: 700;">
            Portable Audio Deepfake Detector
        </h1>
        <p style="margin: 16px 0 24px 0; font-size: 18px;">
            Exact Feature Matching - Production Ready
        </p>
        <div style="background: rgba(255,255,255,0.2); padding: 12px 24px; 
                   border-radius: 25px; display: inline-block;">
            {status_msg}
        </div>
    </div>
    """)
    
    if not models_loaded:
        gr.HTML("""
        <div style="background: #fff3cd; border: 2px solid #ffc107; color: #856404; 
                   padding: 30px; border-radius: 12px; text-align: center;">
            <h3>Setup Required</h3>
            <p><strong>Step 1:</strong> Copy the 'saved_models' folder from your training machine to this directory.</p>
            <p><strong>Step 2:</strong> Optionally copy 'models' folder for VGGish features (will use librosa fallback otherwise).</p>
            <p><strong>Step 3:</strong> Restart this application.</p>
        </div>
        """)
    else:
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<div class="compact-section">')
                gr.Markdown("### Upload Audio File")
                
                audio_input = gr.Audio(label="Select Audio File", type="filepath", sources=["upload"])
                analyze_btn = gr.Button("Analyze Audio", variant="primary", size="lg")
                
                gr.Markdown(f"""
                **Supported Formats:** WAV, MP3, FLAC, M4A, OGG, AAC
                
                **Performance:**  
                Model Accuracy: {metadata['performance']['accuracy']*100:.1f}%  
                Feature Method: Librosa + VGGish fallback  
                Processing: {dev.upper()}
                """)
                gr.HTML('</div>')
            
            with gr.Column(scale=2):
                gr.HTML('<div class="compact-section">')
                
                results_output = gr.HTML("""
                <div style="background: linear-gradient(135deg, #f8f9ff 0%, #e3f2fd 100%); 
                           padding: 50px; border-radius: 16px; text-align: center; 
                           border: 3px dashed #2196f3;">
                    <h3 style="color: #1565c0; margin: 0 0 15px 0;">Ready for Analysis</h3>
                    <p style="color: #333333; margin: 0;">Upload an audio file to detect deepfakes with exact feature matching</p>
                </div>
                """)
                
                gr.HTML('</div>')
                
                plot_output = gr.Plot(show_label=False, container=True)
                
                with gr.Accordion("Technical Details & Statistics", open=False):
                    stats_output = gr.Dataframe(headers=['Metric', 'Value'], show_label=False)
        
        analyze_btn.click(
            fn=predict_deepfake,
            inputs=[audio_input],
            outputs=[results_output, plot_output, stats_output]
        )
    
    gr.HTML(f"""
    <div style="background: white; padding: 30px; border-radius: 16px; margin: 30px 0; 
               text-align: center; border: 2px solid #e3f2fd;">
        <p style="margin: 0; color: #333333; font-size: 16px; font-weight: 500;">
            Exact Feature Matching • Production Ready • Real-time Analysis • {metadata['performance']['accuracy']*100:.1f}% Accuracy
        </p>
        <div style="margin-top: 15px;">
            <span style="font-size: 14px; color: #666666; font-weight: 500;">
                Device: {dev.upper()} | Status: {'Ready' if models_loaded else 'Setup Required'}
            </span>
        </div>
    </div>
    """)

if __name__ == "__main__":
    print("Starting Portable Audio Deepfake Detector...")
    print(f"Device: {dev}")
    print(f"Models loaded: {models_loaded}")
    
    if models_loaded:
        print(f"Model verification: {'PASSED' if metadata.get('verification_passed', False) else 'FAILED'}")
        print(f"Expected accuracy: {metadata['performance']['accuracy']*100:.1f}%")
    else:
        print("Please copy the 'saved_models' folder to this directory")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=None,
        share=False,
        show_error=True
    )