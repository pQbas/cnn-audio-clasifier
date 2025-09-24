import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Union
import time
import sys
import os

from src.dataset import ESC50Dataset
from src.model import ResNet


class Trainer:
    def __init__(self, model, data_loader, optimizer, criterion, scheduler=None):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
        # Información inicial del entrenamiento
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Inicializando entrenamiento en dispositivo: {self.device}")
        print(f"Modelo cargado con {total_params:,} parámetros ({trainable_params:,} entrenables)")
        print(f"Dataset: {len(self.data_loader.dataset)} samples, batch size: {self.data_loader.batch_size}")
        print(f"Optimizador: {type(self.optimizer).__name__}")
        print(f"Función de pérdida: {type(self.criterion).__name__}")
        if self.scheduler:
            print(f"Scheduler: {type(self.scheduler).__name__}")
        else:
            print("Scheduler: None")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.data_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Calcular norma de gradientes para debugging
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
            
            self.optimizer.step()
            
            # Actualizar scheduler si existe (OneCycleLR se actualiza por batch)
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            
            # Calcular accuracy del batch
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct += (predicted == targets).sum().item()
            current_acc = 100. * correct / total_samples
            
            # Mostrar progreso cada 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                elapsed = time.time() - start_time
                samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
                eta_seconds = (len(self.data_loader) - batch_idx - 1) * elapsed / (batch_idx + 1) if batch_idx > 0 else 0
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                
                # Obtener learning rate actual
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f"  Batch {batch_idx+1:4d}/{len(self.data_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Acc: {current_acc:6.2f}% | "
                      f"LR: {current_lr:.6f} | "
                      f"Grad: {grad_norm:.3f} | "
                      f"Speed: {samples_per_sec:5.1f} samples/s | "
                      f"ETA: {eta_min:02d}:{eta_sec:02d}")

        avg_loss = total_loss / len(self.data_loader)
        final_acc = 100. * correct / total_samples
        return avg_loss, final_acc

    def train(self, n_epochs=10):
        self.model.to(self.device)
        print(f"\nComenzando entrenamiento por {n_epochs} épocas...")
        print("=" * 80)
        
        training_start_time = time.time()
        best_loss = float('inf')
        best_acc = 0.0
        
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            print(f"\nÉpoca {epoch+1}/{n_epochs}")
            print("-" * 60)
            
            # Mostrar learning rate actual
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
            
            avg_loss, epoch_acc = self.train_epoch()
            
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - training_start_time
            
            # Actualizar mejores métricas
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"Nuevo mejor loss: {best_loss:.4f}")
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                print(f"Nuevo mejor accuracy: {best_acc:.2f}%")
            
            # Mostrar resumen de la época
            print(f"\nÉpoca {epoch+1} completada:")
            print(f"   Loss promedio: {avg_loss:.4f}")
            print(f"   Accuracy: {epoch_acc:.2f}%")
            print(f"   Tiempo de época: {epoch_time:.2f}s")
            print(f"   Tiempo total: {total_time/60:.1f}min")
            
            # Mostrar uso de memoria GPU si está disponible
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"   Memoria GPU: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
                
        total_training_time = time.time() - training_start_time
        print(f"\nEntrenamiento completado!")
        print(f"Tiempo total: {total_training_time/60:.1f} minutos")
        print(f"Mejor loss: {best_loss:.4f}")
        print(f"Mejor accuracy: {best_acc:.2f}%")
        print("=" * 80)


class TrainerSettings:
    def __init__(
        self,
        esc50_dir: Union[str, Path],
        transform=None,
        batch_size: int = 32,
        lr: float = 0.0005,
        max_lr: float = 0.002,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        use_onecycle: bool = True,
    ):
        self.esc50_dir = esc50_dir
        self.transform = transform
        self.batch_size = batch_size
        self.lr = lr
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.use_onecycle = use_onecycle

        self.model = ResNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Crear scheduler si está habilitado
        if self.use_onecycle:
            train_dataloader = self.get_dataloader(split="train")
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.max_lr,
                epochs=self.num_epochs,
                steps_per_epoch=len(train_dataloader),
                pct_start=0.1
            )
        else:
            self.scheduler = None

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_criterion(self):
        return self.criterion
    
    def get_scheduler(self):
        return self.scheduler

    def get_dataloader(self, split="train"):
        dataset = ESC50Dataset(
            data_dir=self.esc50_dir,
            metadata_file=f"{self.esc50_dir}/meta/esc50.csv",
            split=split,
            transform=self.transform,
        )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=(split == "train")
        )
    
    def get_trainer(self, split="train"):
        """Crea y retorna un objeto Trainer configurado"""
        data_loader = self.get_dataloader(split=split)
        return Trainer(
            model=self.model,
            data_loader=data_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            scheduler=self.scheduler
        )
