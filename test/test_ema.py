import os

import torch
import torch.nn as nn
from accelerate import Accelerator

from src.ema import EMAModel


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 40)
        self.activation = nn.ReLU()
        self.output = nn.Linear(40, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        return self.output(x)


def main():
    """
    Main function demonstrating how to use Accelerator and EMA for model training.

    This function provides a complete training loop and demonstrates the usage of
    Exponential Moving Average (EMA) for stabilizing model weights. 
    """
    # Initialize accelerator for handling distributed training scenarios
    # mixed_precision can be set to "fp16" or "bf16" to enable mixed precision training
    accelerator: Accelerator = Accelerator()
    print(f"Current device: {accelerator.device}")

    # Create model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create EMA model - Exponential Moving Average
    ema_model: EMAModel = EMAModel(
        model.parameters(),
        decay=0.9999,
        use_ema_warmup=True,
    )

    model, optimizer = accelerator.prepare(model, optimizer)

    ema_model.to(accelerator.device)

    # Create sample data
    x: torch.Tensor = torch.randn(5000, 4)
    y: torch.Tensor = torch.randn(5000, 1)
    x = x.to(accelerator.device)
    y = y.to(accelerator.device)

    # Training loop
    print("Starting training...")
    num_steps: int = 1000
    for step in range(num_steps):
        optimizer.zero_grad()
        output: torch.Tensor = model(x)
        loss: torch.Tensor = ((output - y) ** 2).mean()
        accelerator.backward(loss)
        optimizer.step()

        # Only update EMA after gradient synchronization
        if accelerator.sync_gradients:
            ema_model.step(model.parameters())

        if step % 10 == 0 and accelerator.is_main_process:
            print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")

    print("Training completed!")

    # Demonstrate EMA model usage
    if accelerator.is_main_process:
        print("\n=== EMA Model Usage Example ===")

        with torch.no_grad():
            original_output: torch.Tensor = model(x)
            original_loss: torch.Tensor = ((original_output - y) ** 2).mean()
            print(f"Original model loss: {original_loss.item():.4f}")

        print("\nStoring original model weights...")
        ema_model.store(model.parameters())
        
        print("Copying EMA weights to model...")
        ema_model.copy_to(model.parameters())

        with torch.no_grad():
            ema_output: torch.Tensor = model(x)
            ema_loss: torch.Tensor = ((ema_output - y) ** 2).mean()
            print(f"Loss with EMA weights: {ema_loss.item():.4f}")

        print("\nRestoring original model weights...")
        ema_model.restore(model.parameters())

        output_dir: str = "ema_model_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        original_model_path: str = os.path.join(output_dir, "model_original.pt")
        torch.save(model.state_dict(), original_model_path)
        print(f"Saved original model weights to: {original_model_path}")

        # Store original weights again for later use
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())

        # Save EMA model weights
        ema_model_path: str = os.path.join(output_dir, "model_with_ema.pt")
        torch.save(model.state_dict(), ema_model_path)
        
        ema_model.restore(model.parameters())

        print(f"Saved model with EMA weights to: {ema_model_path}")


        # Load and test original weights
        original_checkpoint = torch.load(original_model_path)
        model.load_state_dict(original_checkpoint)

        # Test with original weights
        with torch.no_grad():
            original_output_loaded: torch.Tensor = model(x)
            original_loss_loaded: torch.Tensor = (
                (original_output_loaded - y) ** 2
            ).mean()
            print(f"   Original model loss (loaded): {original_loss_loaded.item():.4f}")

        # Load and test EMA weights
        ema_checkpoint = torch.load(ema_model_path)
        model.load_state_dict(ema_checkpoint)

        # Test with EMA weights
        with torch.no_grad():
            ema_output_loaded: torch.Tensor = model(x)
            ema_loss_loaded: torch.Tensor = ((ema_output_loaded - y) ** 2).mean()
            print(f"   EMA model loss (loaded): {ema_loss_loaded.item():.4f}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
