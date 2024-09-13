from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Specify the input and output directories
input_dir = 'input'
output_dir = 'output'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get a list of all files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

def process_image(index, image_file):
    img_path = os.path.join(input_dir, image_file)

    try:
        # Analyze emotions using DeepFace
        analysis = DeepFace.analyze(img_path=img_path, actions=['emotion'])

        # Load the image
        img = cv2.imread(img_path)

        # Check if image is loaded
        if img is None:
            print(f"Error: Image {image_file} not found or unable to load.")
            return

        # Display the image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # Access the first element of the analysis list which is a dictionary
        plt.title(f"Emotion: {analysis[0]['dominant_emotion']}")

        # Generate a unique numbered filename
        output_image_path = os.path.join(output_dir, f"{index}.png")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the plot to free up memory

        # Save the analysis results to a text file with a numbered name
        output_txt_path = os.path.join(output_dir, f"{index}.txt")
        with open(output_txt_path, 'w') as f:
            f.write(str(analysis))

        # Print detailed emotion analysis
        print(f"Analysis for {image_file}:")
        print(analysis)

    except Exception as e:
        print(f"Error processing image {image_file}: {e}")

# Use ThreadPoolExecutor for concurrent processing
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_image, index, image_file) for index, image_file in enumerate(image_files, start=1)]
    for future in as_completed(futures):
        # Optionally, you can handle the results or exceptions here
        future.result()  # This will raise exceptions if any occurred in the thread
