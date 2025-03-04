import base64

def is_image_path(text):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif")
    if text.endswith(image_extensions):
        return True
    else:
        return False

def encode_image(image_path):
    """Encode image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def model_supports_vision(model_name: str):
    model_name = model_name.lower() if model_name else "dummy"
    hints = ["multimodal", "vl", "vision"]

    return any(h in model_name for h in hints)

def model_on_prem(model_name: str):
    model_name = model_name.lower() if model_name else "dummy"
    hints = ["qwen", "phi-4"]

    return any(h in model_name for h in hints)