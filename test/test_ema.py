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
    Exponential Moving Average (EMA) for stabilizing model weights. It includes:
    - Model and optimizer initialization
    - EMA setup and configuration
    - Training loop with accelerator integration
    - Comparison between original and EMA model performance
    - Demonstration of saving and loading EMA model weights
    """
    # Initialize accelerator for handling distributed training scenarios
    # mixed_precision can be set to "fp16" or "bf16" to enable mixed precision training
    accelerator: Accelerator = Accelerator()
    print(f"Current device: {accelerator.device}")

    # Create model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create EMA model - Exponential Moving Average
    # decay: Rate that determines the ratio of new/old weights, closer to 1 means more retention of old weights
    # use_ema_warmup: Whether to use warmup, using smaller decay rates at the beginning of training
    # inv_gamma, power: Control the change of decay rate during warmup
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

    # Move data to the appropriate device
    x = x.to(accelerator.device)
    y = y.to(accelerator.device)

    # Training loop
    print("Starting training...")
    num_steps: int = 1000
    for step in range(num_steps):
        # Forward pass
        optimizer.zero_grad()
        output: torch.Tensor = model(x)
        loss: torch.Tensor = ((output - y) ** 2).mean()

        # Backward pass - use accelerator.backward instead of loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        # Update EMA model weights
        # In distributed training, only update EMA after gradient synchronization
        if accelerator.sync_gradients:
            ema_model.step(model.parameters())

        # Print loss every 10 steps
        if step % 10 == 0 and accelerator.is_main_process:
            print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")

    print("Training completed!")

    # Demonstrate EMA model usage
    if accelerator.is_main_process:
        print("\n=== EMA Model Usage Example ===")

        # 1. Test original model performance
        with torch.no_grad():
            original_output: torch.Tensor = model(x)
            original_loss: torch.Tensor = ((original_output - y) ** 2).mean()
            print(f"Original model loss: {original_loss.item():.4f}")

        # 2. Store current model weights
        print("\nStoring original model weights...")
        ema_model.store(model.parameters())

        # 3. Copy EMA weights to the model
        print("Copying EMA weights to model...")
        ema_model.copy_to(model.parameters())

        # 4. Test EMA model performance
        with torch.no_grad():
            ema_output: torch.Tensor = model(x)
            ema_loss: torch.Tensor = ((ema_output - y) ** 2).mean()
            print(f"Loss with EMA weights: {ema_loss.item():.4f}")

            # Calculate and display difference
            improvement: float = original_loss.item() - ema_loss.item()
            if improvement > 0:
                print(
                    f"EMA model improved performance: {improvement:.4f} ({improvement / original_loss.item() * 100:.2f}%)"
                )
            else:
                print(
                    f"EMA model did not improve performance in this example: {improvement:.4f}"
                )

        # 5. Restore original model weights
        print("\nRestoring original model weights...")
        ema_model.restore(model.parameters())

        # Demonstrate saving both original and EMA model weights separately
        output_dir: str = "ema_model_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save original model weights first (model already has original weights due to previous restore)
        original_model_path: str = os.path.join(output_dir, "model_original.pt")
        torch.save(model.state_dict(), original_model_path)
        print(f"Saved original model weights to: {original_model_path}")

        # Store original weights again for later use
        ema_model.store(model.parameters())

        # Copy EMA weights to model
        ema_model.copy_to(model.parameters())

        # Save EMA model weights
        ema_model_path: str = os.path.join(output_dir, "model_with_ema.pt")
        torch.save(model.state_dict(), ema_model_path)
        ema_model.restore(model.parameters())

        print(f"Saved model with EMA weights to: {ema_model_path}")

        # Demonstrate how to load and use both models
        print("\nDemonstrating how to load and use the models:")

        # Load and test original weights
        print("1. Loading original model weights...")
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
        print("2. Loading EMA model weights...")
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
