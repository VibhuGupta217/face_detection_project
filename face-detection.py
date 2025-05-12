import cv2
import numpy as np
from openvino.runtime import Core

# Step 1: Load image
image = cv2.imread("sample.jpg")
if image is None:
    print("Image not found. Please check the filename.")
    exit()

original_image = image.copy()
h, w = image.shape[:2]

# Step 2: Load OpenVINO model
ie = Core()
model_path = "intel/face-detection-0200/FP16/face-detection-0200.xml"
model = ie.read_model(model=model_path)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Step 3: Get model input/output info
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Step 4: Prepare image for model
input_blob = cv2.resize(image, (input_layer.shape[3], input_layer.shape[2]))
input_blob = input_blob.transpose((2, 0, 1))  # Convert HWC to CHW
input_blob = np.expand_dims(input_blob, axis=0)

# Step 5: Run inference
results = compiled_model([input_blob])[output_layer]

# Step 6: Draw results
for result in results[0][0]:
    confidence = result[2]
    if confidence > 0.5:  # Confidence threshold
        xmin = int(result[3] * w)
        ymin = int(result[4] * h)
        xmax = int(result[5] * w)
        ymax = int(result[6] * h)
        cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

# Step 7: Show the result
cv2.imshow("Face Detection", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
