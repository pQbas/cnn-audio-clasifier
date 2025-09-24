import modal
import torch
from pydantic import BaseModel

from src.inference import AudioInferenceEngine

app = modal.App("audio-cnn-inference")

image = (
    modal.Image.debian_slim()
    .uv_sync()
    .apt_install(["ffmpeg", "libsndfile1"])
    .add_local_dir("src", remote_path="/root/src")
)

model_volume = modal.Volume.from_name("esc-model")


class InferenceRequest(BaseModel):
    audio_data: str


def create_inference_engine() -> AudioInferenceEngine:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paths = ["/models/best_model.pth", "/models/final_model.pth"]
    model_path = None

    for path in model_paths:
        try:
            torch.load(path, map_location="cpu")
            model_path = path
            break
        except:
            continue

    if model_path is None:
        raise RuntimeError("No se encontró modelo entrenado en /models/")

    engine = AudioInferenceEngine(model_path, device)
    return engine


@app.cls(image=image, gpu="T4", volumes={"/models": model_volume}, scaledown_window=15)
class AudioClassifier:
    @modal.enter()
    def initialize(self):
        print("Inicializando clasificador de audio...")
        self.inference_engine = create_inference_engine()
        self.inference_engine.initialize()
        print("Clasificador inicializado exitosamente")

    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        try:
            result = self.inference_engine.run_inference(request.audio_data)
            return result
        except Exception as e:
            return {
                "error": f"Error durante inferencia: {str(e)}",
                "predictions": [],
                "visualization": {},
                "input_spectrogram": {"shape": [], "values": []},
                "waveform": {"values": [], "sample_rate": 0, "duration": 0},
            }


@app.local_entrypoint()
def main():
    print("Iniciando servidor de inferencia...")
    classifier = AudioClassifier()
    print(f"Servidor de inferencia disponible en: {classifier.inference.get_web_url()}")
