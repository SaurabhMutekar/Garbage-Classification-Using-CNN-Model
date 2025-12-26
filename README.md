♻️ Garbage-Classification-Using-CNN-Model
A Deep Learning-powered web application that classifies waste into six different categories to assist in recycling and waste management efforts. This project utilizes a Convolutional Neural Network (CNN) trained on the TrashNet dataset and serves predictions via a Flask web interface.

📌 Overview
Waste management is a critical global challenge. This application automates the process of identifying waste types using computer vision. Users can upload or drag-and-drop an image of a waste item, and the system predicts its material type.

Classifies waste into 6 categories:

Cardboard

Glass

Metal

Paper

Plastic

Trash

🛠️ Tech Stack
Frontend: HTML5, CSS3, JavaScript (Drag & Drop interface)

Backend: Python, Flask

Machine Learning: TensorFlow, Keras, NumPy

Visualization: Matplotlib

Dataset: TrashNet (Gary Thung)

📂 Project Structure



Plaintext

Garbage-Classification-Using-CNN-Model/
│
├── model/
│   └── garbage_classification_model.h5  
│
├── static/
│   ├── script.js                        # The provided frontend logic
│   └── styles.css                       # (Optional) CSS for styling
│
├── templates/
│   └── index.html                       # HTML template for the UI
│
├── app.py                               # Flask backend application
├── garbage_classification.ipynb         # Jupyter Notebook for training
├── requirements.txt                     # List of python dependencies
└── README.md
🧠 Model Architecture & Training
The model is built using TensorFlow/Keras and trained in the garbage_classification.ipynb notebook.

Workflow:
Data Loading: Images are loaded from the dataset-resized folder.

Preprocessing: Image rescaling (1./255) and augmentation (shear, zoom, horizontal flip) to prevent overfitting.

CNN Architecture:

3 Convolutional layers with ReLU activation.

Max Pooling layers to reduce dimensionality.

Flatten layer.

Dense hidden layer (128 neurons).

Output layer (6 neurons, Softmax activation).


Shutterstock
Training: The model is trained for 20 epochs using the Adam optimizer and Categorical Crossentropy loss.

🚀 Setup & Installation
1. Clone the Repository
Bash

git clone https://github.com/SaurabhMutekar/Garbage-Classification-Using-CNN-Model.git
cd Garbage-Classification-Using-CNN-Model
2. Install Dependencies
Create a virtual environment (optional) and install the required packages.

Bash

pip install flask tensorflow numpy matplotlib
3. Prepare the Model
If you haven't trained the model yet:

Open garbage_classification.ipynb in Google Colab or Jupyter.

Download the dataset: Link.

Run the cells to train the model.

Download the generated garbage_classification_20.h5 file.

Rename it to garbage_classification_model.h5 and place it in the model/ folder.

4. Run the Application
Bash

python app.py
The server will start at http://127.0.0.1:5000/.

💻 Usage
Open your web browser and navigate to the localhost URL.

Click the upload area or drag and drop an image of waste (e.g., a crushed soda can or a plastic bottle).

Click Upload.

The system will process the image and display the predicted category (e.g., "The predicted class is: Metal").

📊 Dataset Distribution
The model was trained on the TrashNet dataset, which contains 2,527 images:

Cardboard: 403

Glass: 501

Metal: 410

Paper: 594

Plastic: 482

Trash: 137

📜 License
This project is open-source and available for educational purposes. The dataset is provided by Gary Thung under the TrashNet repository.

Developed with ❤️ using Python and Deep Learning.
