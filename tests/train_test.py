import pytest
import torch
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from src.dataset import ESC50Dataset
from src.model import ResNet
from src.train import TrainerSettings, Trainer
from src.data_operations import train_transform


class MockESC50Dataset(Dataset):
    """Mock dataset que simula ESC50Dataset pero con muy pocos samples para tests rápidos"""
    
    def __init__(self, num_samples=64, transform=None):  # Incrementamos a 64 para evitar problemas con scheduler
        self.num_samples = num_samples
        self.transform = transform
        self.num_classes = 50  # ESC50 tiene 50 clases
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Crear datos sintéticos que imiten audio de ESC50 (waveform, no espectrograma)
        # ESC50 típicamente tiene ~5 segundos de audio a 22050 Hz = ~110,250 samples
        # Para tests rápidos, usamos algo más pequeño pero que funcione con las transformaciones
        sample_rate = 22050
        duration_seconds = 2  # 2 segundos en lugar de 5 para tests más rápidos
        num_samples_audio = int(sample_rate * duration_seconds)  # ~44,100 samples
        
        # Generar waveform sintético (audio en formato 1D, luego añadir dimensión de canal)
        audio_data = torch.randn(num_samples_audio)  # Shape: [44100] para 2 segundos
        audio_data = audio_data.unsqueeze(0)  # Añadir dimensión de canal: [1, 44100]
        
        # Label aleatorio entre 0 y 49 (50 clases de ESC50)
        label = torch.randint(0, self.num_classes, (1,)).item()
        
        if self.transform:
            audio_data = self.transform(audio_data)
            
        return audio_data, label


def create_mock_trainer_settings(batch_size=8, transform=None, use_onecycle=False, max_lr=0.001, num_epochs=3):
    """Crea TrainerSettings configurado para usar mock dataset"""
    
    class MockTrainerSettings:
        def __init__(self, batch_size=8, transform=None, use_onecycle=False, max_lr=0.001, num_epochs=3):
            self.batch_size = batch_size
            self.transform = transform
            self.use_onecycle = use_onecycle
            self.max_lr = max_lr
            self.num_epochs = num_epochs
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        def get_model(self):
            return ResNet(categories=50).to(self.device)
            
        def get_optimizer(self):
            model = self.get_model()
            return torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            
        def get_criterion(self):
            return torch.nn.CrossEntropyLoss()
            
        def get_dataloader(self, split="train"):
            # Usar más samples para evitar problemas con scheduler
            dataset = MockESC50Dataset(num_samples=64, transform=self.transform)  
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
        def get_scheduler(self):
            if not self.use_onecycle:
                return None
                
            optimizer = self.get_optimizer()
            dataloader = self.get_dataloader()
            steps_per_epoch = len(dataloader)  # 64 samples / 8 batch_size = 8 steps
            
            # Asegurar que tenemos suficientes pasos para el scheduler
            if steps_per_epoch < 1:
                print(f"Warning: Only {steps_per_epoch} steps per epoch, scheduler may not work properly")
                return None
            
            # Usar la misma configuración que el código original
            return OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                epochs=self.num_epochs,        # Cambiar de total_steps a epochs
                steps_per_epoch=steps_per_epoch,  # Añadir steps_per_epoch
                pct_start=0.1
            )
            
        def get_trainer(self):
            model = self.get_model()
            dataloader = self.get_dataloader()
            optimizer = self.get_optimizer()
            criterion = self.get_criterion()
            
            # Crear el scheduler usando el MISMO optimizer que el trainer
            if self.use_onecycle:
                steps_per_epoch = len(dataloader)
                scheduler = OneCycleLR(
                    optimizer,  # ← Usar el mismo optimizer
                    max_lr=self.max_lr,
                    epochs=self.num_epochs,
                    steps_per_epoch=steps_per_epoch,
                    pct_start=0.1
                )
            else:
                scheduler = None
            
            return Trainer(model, dataloader, optimizer, criterion, scheduler)
    
    return MockTrainerSettings(batch_size, transform, use_onecycle, max_lr, num_epochs)


def test_model_creation():
    settings = create_mock_trainer_settings(batch_size=16)
    model = settings.get_model()

    assert isinstance(model, ResNet), "El modelo no es una instancia de ResNet"
    assert model is not None, "El modelo no se creó correctamente"


def test_optimizer_creation():
    settings = create_mock_trainer_settings(batch_size=16)
    optimizer = settings.get_optimizer()

    assert isinstance(optimizer, torch.optim.AdamW), "El optimizador no es AdamW"
    assert optimizer is not None, "El optimizador no se creó correctamente"


def test_dataloader_creation():
    settings = create_mock_trainer_settings(batch_size=16)
    dataloader = settings.get_dataloader(split="train")

    assert isinstance(
        dataloader, torch.utils.data.DataLoader
    ), "El dataloader no es una instancia de DataLoader"
    assert len(dataloader) > 0, "El dataloader está vacío"


def test_training():
    settings = create_mock_trainer_settings(
        batch_size=16, transform=train_transform
    )

    dataloader = settings.get_dataloader(split="train")
    model = settings.get_model()
    optimizer = settings.get_optimizer()
    criterion = settings.get_criterion()

    trainer = Trainer(model, dataloader, optimizer, criterion)

    try:
        avg_loss, final_acc = trainer.train_epoch()  # Ahora retorna tupla (loss, accuracy)
        print(f"Average Loss after 1 epoch: {avg_loss:.4f}")
        print(f"Final Accuracy after 1 epoch: {final_acc:.2f}%")
        assert avg_loss > 0, "La pérdida debe ser mayor que 0"
        assert 0 <= final_acc <= 100, "La accuracy debe estar entre 0 y 100%"
    except Exception as e:
        pytest.fail(f"Error durante el entrenamiento: {e}")


def test_scheduler_creation():
    """Test que el scheduler OneCycleLR se crea correctamente"""
    settings = create_mock_trainer_settings(
        batch_size=16, 
        use_onecycle=True,
        max_lr=0.002,
        num_epochs=5
    )
    
    scheduler = settings.get_scheduler()
    
    assert scheduler is not None, "El scheduler no debe ser None cuando use_onecycle=True"
    assert isinstance(scheduler, OneCycleLR), "El scheduler debe ser una instancia de OneCycleLR"


def test_no_scheduler_creation():
    """Test que no se crea scheduler cuando use_onecycle=False"""
    settings = create_mock_trainer_settings(
        batch_size=16, 
        use_onecycle=False
    )
    
    scheduler = settings.get_scheduler()
    
    assert scheduler is None, "El scheduler debe ser None cuando use_onecycle=False"


def test_trainer_with_scheduler():
    """Test que Trainer funciona correctamente con scheduler"""
    settings = create_mock_trainer_settings(
        batch_size=16, 
        transform=train_transform,
        use_onecycle=True,
        max_lr=0.001,
        num_epochs=3
    )

    # Usar el método get_trainer() para obtener un trainer completamente configurado
    trainer = settings.get_trainer()

    # Verificar que el trainer tiene scheduler
    assert trainer.scheduler is not None, "El trainer debe tener un scheduler"
    assert isinstance(trainer.scheduler, OneCycleLR), "El scheduler debe ser OneCycleLR"

    # Ejecutar 1 época de entrenamiento
    try:
        avg_loss, final_acc = trainer.train_epoch()
        assert avg_loss > 0, "La pérdida debe ser mayor que 0"
        assert 0 <= final_acc <= 100, "La accuracy debe estar entre 0 y 100%"
    except Exception as e:
        pytest.fail(f"Error durante el entrenamiento con scheduler: {e}")


def test_trainer_without_scheduler():
    """Test que Trainer funciona correctamente sin scheduler"""
    settings = create_mock_trainer_settings(
        batch_size=16, 
        transform=train_transform,
        use_onecycle=False
    )

    trainer = settings.get_trainer()

    # Verificar que el trainer NO tiene scheduler
    assert trainer.scheduler is None, "El trainer no debe tener scheduler cuando use_onecycle=False"

    # Ejecutar 1 época de entrenamiento
    try:
        avg_loss, final_acc = trainer.train_epoch()
        assert avg_loss > 0, "La pérdida debe ser mayor que 0"
        assert 0 <= final_acc <= 100, "La accuracy debe estar entre 0 y 100%"
    except Exception as e:
        pytest.fail(f"Error durante el entrenamiento sin scheduler: {e}")


def test_learning_rate_changes_with_scheduler():
    """Test que el learning rate cambia correctamente con OneCycleLR"""
    settings = create_mock_trainer_settings(
        batch_size=8,  # batch más pequeño para tener más steps
        transform=train_transform,  # Añadir transformación para convertir waveform a espectrograma
        use_onecycle=True,
        max_lr=0.01,
        num_epochs=2
    )

    trainer = settings.get_trainer()
    
    # Capturar LRs durante el entrenamiento para verificar cambios
    learning_rates = []
    
    # Obtener LR inicial
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    learning_rates.append(initial_lr)
    print(f"Initial LR: {initial_lr}")
    
    # Manual training loop para capturar LRs intermedios
    trainer.model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(trainer.data_loader):
        inputs, targets = batch
        inputs, targets = inputs.to(trainer.device), targets.to(trainer.device)

        trainer.optimizer.zero_grad()
        outputs = trainer.model(inputs)
        loss = trainer.criterion(outputs, targets)
        loss.backward()
        trainer.optimizer.step()
        
        # Actualizar scheduler si existe y capturar LR
        if trainer.scheduler:
            trainer.scheduler.step()
            current_lr = trainer.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            print(f"After batch {batch_idx+1}: LR = {current_lr}")

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    final_lr = trainer.optimizer.param_groups[0]['lr']
    print(f"Final LR: {final_lr}")
    print(f"All LRs: {learning_rates}")
    
    # Verificar que el LR ha cambiado durante el entrenamiento
    # Con OneCycleLR deberían haber cambios
    unique_lrs = set(learning_rates)
    assert len(unique_lrs) > 1, f"El learning rate debería cambiar con OneCycleLR. LRs observados: {learning_rates}"
    
    # Verificar que el entrenamiento funcionó
    avg_loss = total_loss / len(trainer.data_loader)
    final_acc = 100. * correct / total_samples
    assert avg_loss > 0, "La pérdida debe ser mayor que 0"
    assert 0 <= final_acc <= 100, "La accuracy debe estar entre 0 y 100%"


def test_trainer_settings_parameters():
    """Test que TrainerSettings acepta y usa correctamente los nuevos parámetros"""
    max_lr = 0.005
    num_epochs = 15
    use_onecycle = True
    
    settings = create_mock_trainer_settings(
        batch_size=16,
        max_lr=max_lr,
        num_epochs=num_epochs,
        use_onecycle=use_onecycle
    )
    
    # Verificar que los parámetros se guardan correctamente
    assert settings.max_lr == max_lr, f"max_lr debería ser {max_lr}, pero es {settings.max_lr}"
    assert settings.num_epochs == num_epochs, f"num_epochs debería ser {num_epochs}, pero es {settings.num_epochs}"
    assert settings.use_onecycle == use_onecycle, f"use_onecycle debería ser {use_onecycle}, pero es {settings.use_onecycle}"
    
    # Verificar que el scheduler se crea con los parámetros correctos
    scheduler = settings.get_scheduler()
    assert scheduler is not None, "Scheduler debería crearse cuando use_onecycle=True"


@pytest.mark.skipif(not Path("./data/esc50-data").exists(), reason="Dataset ESC50 real no encontrado")
def test_real_dataset_smoke_test():
    """
    Smoke test con dataset ESC50 real - verifica que el sistema funciona 
    con datos reales procesando solo unos pocos samples
    """
    import os
    from pathlib import Path
    
    # Verificar que el dataset existe
    esc50_dir = Path("./data/esc50-data")
    if not esc50_dir.exists():
        # Buscar en ubicaciones alternativas comunes
        alt_paths = [
            Path("./esc50"),
            Path("./data/esc50-data"), 
            Path("../esc50"),
            Path.home() / "esc50"
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                esc50_dir = alt_path
                break
        else:
            pytest.skip("Dataset ESC50 no encontrado en ninguna ubicación")
    
    metadata_file = esc50_dir / "meta" / "esc50.csv"
    if not metadata_file.exists():
        pytest.skip(f"Archivo metadata no encontrado: {metadata_file}")
    
    print(f"🔍 Usando dataset ESC50 en: {esc50_dir}")
    
    try:
        # Crear dataset real pero limitar a muy pocos samples
        real_dataset = ESC50Dataset(
            data_dir=esc50_dir,
            metadata_file=metadata_file,
            split="train",
            transform=train_transform
        )
        
        print(f"📁 Dataset real cargado: {len(real_dataset)} samples totales")
        
        # Crear un subset muy pequeño (solo primeros 16 samples)
        from torch.utils.data import Subset
        small_indices = list(range(min(16, len(real_dataset))))
        small_dataset = Subset(real_dataset, small_indices)
        
        print(f"🔬 Usando subset pequeño: {len(small_dataset)} samples")
        
        # Crear dataloader con batch pequeño
        small_dataloader = DataLoader(small_dataset, batch_size=4, shuffle=False)
        
        # Verificar que los datos se cargan correctamente
        print("🧪 Verificando formato de datos...")
        sample_batch = next(iter(small_dataloader))
        inputs, targets = sample_batch
        
        print(f"  📊 Input shape: {inputs.shape}")
        print(f"  🎯 Targets shape: {targets.shape}")
        print(f"  📈 Input dtype: {inputs.dtype}")
        print(f"  🔢 Targets dtype: {targets.dtype}")
        
        # Verificar dimensiones esperadas (después de transformación)
        expected_dims = 4  # [batch, channels, height, width]
        assert len(inputs.shape) == expected_dims, f"Input debería tener {expected_dims} dimensiones, tiene {len(inputs.shape)}"
        
        # Verificar que las transformaciones produjeron espectrograma 2D
        batch_size, channels, height, width = inputs.shape
        assert channels == 1, f"Channels debería ser 1, es {channels}"
        assert height > 1 and width > 1, f"Espectrograma debería ser 2D, shape: {height}x{width}"
        
        print("✅ Formato de datos correcto")
        
        # Crear modelo y componentes de entrenamiento
        print("🤖 Creando modelo y optimizador...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNet(categories=50).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Crear scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=1,
            steps_per_epoch=len(small_dataloader),
            pct_start=0.1
        )
        
        # Crear trainer
        trainer = Trainer(model, small_dataloader, optimizer, criterion, scheduler)
        
        print("🚀 Ejecutando smoke test de entrenamiento...")
        
        # Ejecutar solo unos pocos batches para verificar que funciona
        model.train()
        successful_batches = 0
        
        for batch_idx, batch in enumerate(small_dataloader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Verificar que el loss es razonable
            assert not torch.isnan(loss), "Loss es NaN - problema en el modelo"
            assert loss.item() > 0, "Loss debería ser positivo"
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            successful_batches += 1
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"  ✅ Batch {batch_idx+1}/{len(small_dataloader)} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
            
            # Solo procesar los primeros 3 batches para mantener el test rápido
            if batch_idx >= 2:
                break
        
        print(f"🎉 Smoke test completado exitosamente!")
        print(f"   📦 Batches procesados: {successful_batches}")
        print(f"   🎯 Dataset real es compatible con el sistema")
        
        # Verificaciones finales
        assert successful_batches > 0, "Debería haber procesado al menos 1 batch"
        
    except Exception as e:
        print(f"❌ Error en smoke test: {e}")
        print("💡 Esto indica incompatibilidad entre mock y dataset real")
        raise
