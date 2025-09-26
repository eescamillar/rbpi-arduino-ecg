import torch

device = torch.device("cuda" if torch.cuda.is_availabe() else "cpu")

print(f"Using devide: {device}")

devNumber = torch.cuda.current_device()

print(f"Current device: {devNumber}")

devName = torch.cuda.get_device_name(devNumer)

devName(f"GPU name is: {devName}")