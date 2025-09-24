import base64
import io
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Any

from .model import ResNet


class AudioProcessor:
    """Responsable del procesamiento de audio para inferencia"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=sample_rate // 2
            ),
            T.AmplitudeToDB()
        )

    def decode_audio_from_base64(self, audio_data: str) -> Tuple[np.ndarray, int]:
        """Decodifica audio desde base64"""
        audio_bytes = base64.b64decode(audio_data)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        return audio_data, sample_rate

    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocesa audio: convierte a mono y resamplea si es necesario"""
        # Convertir a mono si es estéreo
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resamplear si es necesario
        if sample_rate != self.sample_rate:
            audio_data = librosa.resample(
                y=audio_data, orig_sr=sample_rate, target_sr=self.sample_rate
            )
        
        return audio_data

    def create_spectrogram(self, audio_data: np.ndarray) -> torch.Tensor:
        """Convierte audio a espectrograma"""
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)

    def downsample_waveform(self, audio_data: np.ndarray, max_samples: int = 8000) -> np.ndarray:
        """Reduce el número de muestras para visualización"""
        if len(audio_data) > max_samples:
            step = len(audio_data) // max_samples
            return audio_data[::step]
        return audio_data


class ModelManager:
    """Responsable de la carga y manejo del modelo"""
    
    def __init__(self, model_path: str, device: torch.device):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.classes = None

    def load_model(self):
        """Carga el modelo desde checkpoint"""
        print("Cargando modelo...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.classes = checkpoint['classes']
        
        self.model = ResNet(categories=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print("Modelo cargado exitosamente")

    def predict(self, spectrogram: torch.Tensor, return_feature_maps: bool = True) -> Tuple[torch.Tensor, Dict]:
        """Realiza predicción con el modelo"""
        if self.model is None:
            raise RuntimeError("Modelo no ha sido cargado")
        
        with torch.no_grad():
            output = self.model(spectrogram)
            # El modelo actual no retorna feature maps, retornamos diccionario vacío
            return output, {}


class InferenceProcessor:
    """Responsable de procesar resultados de inferencia"""
    
    def __init__(self, classes: List[str]):
        self.classes = classes

    def process_predictions(self, output: torch.Tensor, top_k: int = 3) -> List[Dict[str, Any]]:
        """Procesa las predicciones del modelo"""
        # Limpiar NaN values
        output = torch.nan_to_num(output)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities[0], top_k)
        
        predictions = [
            {
                "class": self.classes[idx.item()], 
                "confidence": prob.item()
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return predictions

    def process_feature_maps(self, feature_maps: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        """Procesa los mapas de características para visualización"""
        viz_data = {}
        
        for name, tensor in feature_maps.items():
            if tensor.dim() == 4:  # [batch_size, channels, height, width]
                # Promediar canales y remover dimensión de batch
                aggregated_tensor = torch.mean(tensor, dim=1)
                squeezed_tensor = aggregated_tensor.squeeze(0)
                numpy_array = squeezed_tensor.cpu().numpy()
                clean_array = np.nan_to_num(numpy_array)
                
                viz_data[name] = {
                    "shape": list(clean_array.shape),
                    "values": clean_array.tolist()
                }
        
        return viz_data

    def create_response(
        self, 
        predictions: List[Dict[str, Any]], 
        viz_data: Dict[str, Dict[str, Any]],
        spectrogram: torch.Tensor,
        waveform_data: np.ndarray,
        waveform_sample_rate: int,
        original_duration: float
    ) -> Dict[str, Any]:
        """Crea la respuesta final de la inferencia"""
        # Preparar espectrograma para respuesta
        spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
        clean_spectrogram = np.nan_to_num(spectrogram_np)
        
        response = {
            "predictions": predictions,
            "visualization": viz_data,
            "input_spectrogram": {
                "shape": list(clean_spectrogram.shape),
                "values": clean_spectrogram.tolist()
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": waveform_sample_rate,
                "duration": original_duration
            }
        }
        
        return response


class AudioInferenceEngine:
    """Clase principal que orquesta todo el proceso de inferencia"""
    
    def __init__(self, model_path: str, device: torch.device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.audio_processor = AudioProcessor()
        self.model_manager = ModelManager(model_path, device)
        self.inference_processor = None
        
    def initialize(self):
        """Inicializa el motor de inferencia"""
        self.model_manager.load_model()
        self.inference_processor = InferenceProcessor(self.model_manager.classes)
    
    def run_inference(self, audio_data_b64: str) -> Dict[str, Any]:
        """Ejecuta el proceso completo de inferencia"""
        if self.inference_processor is None:
            raise RuntimeError("Motor de inferencia no ha sido inicializado")
        
        # 1. Procesar audio de entrada
        audio_data, sample_rate = self.audio_processor.decode_audio_from_base64(audio_data_b64)
        audio_data = self.audio_processor.preprocess_audio(audio_data, sample_rate)
        
        # 2. Crear espectrograma
        spectrogram = self.audio_processor.create_spectrogram(audio_data)
        spectrogram = spectrogram.to(self.device)
        
        # 3. Realizar predicción
        output, feature_maps = self.model_manager.predict(spectrogram, return_feature_maps=True)
        
        # 4. Procesar resultados
        predictions = self.inference_processor.process_predictions(output)
        viz_data = self.inference_processor.process_feature_maps(feature_maps)
        
        # 5. Preparar datos adicionales
        waveform_data = self.audio_processor.downsample_waveform(audio_data)
        original_duration = len(audio_data) / self.audio_processor.sample_rate
        
        # 6. Crear respuesta
        response = self.inference_processor.create_response(
            predictions=predictions,
            viz_data=viz_data,
            spectrogram=spectrogram,
            waveform_data=waveform_data,
            waveform_sample_rate=44100,  # Para compatibilidad con frontend
            original_duration=original_duration
        )
        
        return response