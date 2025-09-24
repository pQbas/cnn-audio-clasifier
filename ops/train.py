import modal
from src.train import Trainer, TrainerSettings
from src.data_operations import train_transform, val_transform

app = modal.App("cnn-audio-training")

image = (
    modal.Image.debian_slim()
    .uv_sync()
    .apt_install(["ffmpeg", "libsndfile1", "wget", "unzip"])
    .run_commands(
        [
            "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
            "cd /tmp && unzip esc50.zip",
            "mkdir -p /opt/esc50-data",
            "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
            "rm -rf /tmp/esc50.zip /tmp/ESC-50-master",
        ]
    )
    .add_local_dir("src", remote_path="/root/src")
)


volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)


def check_gpu():
    import torch

    if not torch.cuda.is_available():
        return False, None
    return True, torch.cuda.get_device_name(0)


def create_trainer_settings(n_epochs, batch_size, max_lr):
    return TrainerSettings(
        esc50_dir="/opt/esc50-data",
        transform=train_transform,
        batch_size=batch_size,
        lr=0.0005,
        max_lr=max_lr,
        weight_decay=0.01,
        num_epochs=n_epochs,
        use_onecycle=True,
    )


def save_model(trainer, model_path, n_epochs, batch_size, max_lr):
    import torch
    import os

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Get classes from the dataset
    classes = trainer.data_loader.dataset.classes
    
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": (
                trainer.scheduler.state_dict() if trainer.scheduler else None
            ),
            "classes": classes,
            "settings": {
                "max_lr": max_lr,
                "batch_size": batch_size,
                "num_epochs": n_epochs,
            },
        },
        model_path,
    )
    return model_path


def print_trainer_info(trainer):
    print("Configuración del trainer:")
    print("   Modelo: {}".format(trainer.model.__class__.__name__))
    print("   Optimizer: {}".format(trainer.optimizer.__class__.__name__))
    print("   Criterion: {}".format(trainer.criterion.__class__.__name__))
    if trainer.scheduler:
        print("   Scheduler: {}".format(trainer.scheduler.__class__.__name__))


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume, "/models": model_volume},
    timeout=60 * 60 * 3,
)
def train(n_epochs=100, batch_size=32, max_lr=0.002):
    print("Iniciando entrenamiento en Modal...")
    print(
        "Parámetros: {} épocas, batch_size={}, max_lr={}".format(
            n_epochs, batch_size, max_lr
        )
    )

    gpu_available, gpu_name = check_gpu()
    if not gpu_available:
        print("No se encontró GPU disponible. Entrenamiento finalizado.")
        return {
            "status": "failed",
            "error": "No GPU available",
            "epochs_trained": 0,
            "model_path": None,
        }

    print("GPU disponible: {}".format(gpu_name))

    settings = create_trainer_settings(n_epochs, batch_size, max_lr)
    trainer = settings.get_trainer(split="train")

    print_trainer_info(trainer)

    trainer.train(n_epochs=n_epochs)

    model_path = save_model(
        trainer, "/models/final_model.pth", n_epochs, batch_size, max_lr
    )

    print("Entrenamiento completado!")
    print("Modelo guardado en: {}".format(model_path))

    return {"status": "completed", "epochs_trained": n_epochs, "model_path": model_path}


@app.local_entrypoint()
def main():
    """Punto de entrada para ejecutar entrenamiento"""
    result = train.remote(
        n_epochs=5,
        batch_size=32,
        max_lr=0.002,
    )
    print("Resultado del entrenamiento:", result)
