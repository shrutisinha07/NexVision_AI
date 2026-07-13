from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tkinter import filedialog
import tkinter as tk

print("⏳ Loading AI model...")

# Load AI Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

print("✅ AI Model Loaded Successfully!")

while True:

    # Open File Picker
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )

    # If user cancels file selection
    if not file_path:
        print("❌ No image selected!")
        break

    # Load Image
    image = Image.open(file_path)
    print("✅ Image Loaded Successfully!")

    # Convert image for AI
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    output = model.generate(**inputs)

    # Decode caption
    caption = processor.decode(output[0], skip_special_tokens=True)

    # Display caption
    print("\n📝 AI Caption:")
    print(caption)

    # Ask user if they want another caption
    choice = input("\nDo you want to caption another image? (yes/no): ").lower()

    if choice != "yes":
        print("\n👋 Thank you for using NexVision AI!")
        break