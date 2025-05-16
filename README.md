
DE vs BP MNIST Comparison
This project presents a comparative study between Differential Evolution (DE) and Backpropagation (BP) for training the final classification head of a neural network on the MNIST dataset. The system is equipped with a Tkinter GUI to visualize and experiment with various DE strategies and compare their performance with standard backpropagation.

🧠 Overview
The model architecture is a simple feedforward neural network:

Input (784) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(10, Softmax)

The feature extractor layers are pretrained and frozen.

Only the final classification head is trained using:

Differential Evolution (DE) with various strategies.

Standard Backpropagation (BP) for comparison.

🚀 Features
Supports multiple DE configuration strategies:

Initialization: uniform, xavier, normal

Mutation: rand1, rand2, best1, jde, gauss

Crossover: binomial, exponential

Selection: greedy, tournament

Diversity: none, sharing, crowding (only sharing is implemented)

Real-time comparison using a Tkinter GUI with:

Accuracy, loss, fitness, and diversity plots

Confusion matrices for DE and BP

Sample predictions for digits 0–9






🗂️ Project Structure
bash
Copy
Edit
DE_vs_BP_MNIST/
├── Ea project.py             # Main script with GUI and full logic
├── README.md                 # Project documentation




Install the required Python packages:

bash
Copy
Edit
pip install numpy matplotlib scikit-learn tensorflow
Tkinter is included with most Python installations.



🧪 Running the Application
Execute the main script:

bash
Copy
Edit
python Ea project.py
A GUI will open where you can:

Choose DE strategy configurations.

Set population size, generations, F, and CR.

Click ▶ Run to start the comparison.

Browse results across three tabs:

Metrics: Accuracy, loss, fitness, and diversity over generations

Confusion: Side-by-side confusion matrices

Samples: Visualization of predictions on sample images

![Screenshot 2025-05-16 234515](https://github.com/user-attachments/assets/5cce3ed1-2312-42e3-abb5-c2e1b1c4ae4a)

![Screenshot 2025-05-16 233615](https://github.com/user-attachments/assets/e67fe44c-9290-457b-961d-390b52eac995)

![Screenshot 2025-05-16 233638](https://github.com/user-attachments/assets/3eaefdd2-002a-481c-a58a-c9fb417e0201)

![WhatsApp Image 2025-05-16 at 23 49 20_0bc99a9d](https://github.com/user-attachments/assets/646d660d-82ca-4f79-a334-c65e6331432a)


