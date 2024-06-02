import numpy as np
import cv2
import os

def hopfield_model(training_data):
    num_patterns, pattern_length = training_data.shape
    weights = np.zeros((pattern_length, pattern_length))

    for pattern in training_data:
        pattern = pattern.reshape((pattern_length, 1))
        weights += np.dot(pattern, pattern.T)

    np.fill_diagonal(weights, 0)
    return weights

def identify_c_ring(image, hopfield_weights):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    flattened_image = binary_image.flatten()
    output_image = hopfield_iteration(flattened_image, hopfield_weights)
    output_image = output_image.reshape(binary_image.shape)

    contours, _ = cv2.findContours(output_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_contour = max(contours, key=cv2.contourArea)
    c_center, _ = cv2.minEnclosingCircle(c_contour)

    return c_center

def determine_coordinates_c(image, c_center):
    a_center = (c_center[0], c_center[1] - 20)  

    return a_center
  
def hopfield_iteration(input_pattern, weights, max_iterations=100):
  pattern_length = len(input_pattern)
  output_pattern = np.copy(input_pattern)

  for _ in range(max_iterations):
      for i in range(pattern_length):
          net_input = np.dot(weights[i], output_pattern)
          output_pattern[i] = np.sign(net_input)

  return output_pattern

def load_training_images(directory):
  training_images = []
  for filename in os.listdir(directory):
      if filename.endswith(".jpg") or filename.endswith(".png"):
          image_path = os.path.join(directory, filename)
          image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
          training_images.append(image.flatten())  
  return np.array(training_images)

training_directory = 'personal/inteligenciaArtificial/imagenMotor.jpg'

training_data = load_training_images(training_directory)



def main():
    training_data = np.array([...]) 
    hopfield_weights = hopfield_model(training_data)

    image = cv2.imread('personal/inteligenciaArtificial/imagenMotor.jpg')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    c_center = identify_c_ring(gray_image, hopfield_weights)

    a_center = determine_coordinates_c(gray_image, c_center)

    print("Coordenadas del centro 'A':", a_center)

if __name__ == "__main__":
    main()

