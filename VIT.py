# -*- coding: utf-8 -*-

# Import libraries
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Define the Vision Transformer (ViT) architecture
class VisionTransformer(nn.Module):
    def __init__(self, size_of_patch=4, depth=6, attn_heads=4):
        super(VisionTransformer, self).__init__()

        # Define hyperparameters
        self.size_of_patch = size_of_patch  # The dimensions of each image patch
        self.depth = depth  # The number of Transformer blocks
        self.attn_heads = attn_heads    # The number of attention heads in the multi-head attention
        self.embed_dim = size_of_patch * size_of_patch  # Dimensionality of patch embeddings
        self.total_patches = (28 * 28) // self.embed_dim  # Total number of patches obtained from the image

        # Convolution layer to transform image patches into flat embeddings
        self.patch_embedder = nn.Conv2d(1, self.embed_dim, kernel_size=size_of_patch, stride=size_of_patch)

        # Learnable embeddings to incorporate position information into patch embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, self.total_patches + 1, self.embed_dim))

        # The Transformer's encoder layers
        transformer_block = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=attn_heads, batch_first=True)
        self.transformer_encoders = nn.TransformerEncoder(transformer_block, num_layers=depth)

        # The classification head that makes the final prediction
        self.head = nn.Linear(self.embed_dim, 10)

    def forward(self, images):
        batch_sz = images.size(0)  # Determine batch size

        # Extract patches and project them into embeddings
        patches = self.patch_embedder(images).flatten(2).transpose(1, 2)

        # Combine positional embeddings with patch embeddings
        patch_with_pos = patches + self.position_embeddings[:, :patches.size(1)]

        # Pass the sequence of patch embeddings through the Transformer encoder
        encoded_patches = self.transformer_encoders(patch_with_pos)

        # Aggregate encoded patch information and pass through a classification head
        pooled_patch_info = encoded_patches.mean(dim=1)
        class_logits = self.head(pooled_patch_info)

        return class_logits

import torchvision
import torchvision.transforms as transforms

# Load Fashion-MNIST data
fashion_mnist_train_data = torchvision.datasets.FashionMNIST(
    './data/FashionMNIST/',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

import torchvision
import torchvision.transforms as transforms

# Load Fashion-MNIST test data
fashion_mnist_test_data = torchvision.datasets.FashionMNIST(
    './data/FashionMNIST/',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

import torch.optim as optim
import torch.utils.data

# Setting up data loaders for the Fashion-MNIST dataset
# Data loaders are used to efficiently load and batch the dataset
train_data_loader = torch.utils.data.DataLoader(fashion_mnist_train_data, batch_size=64, shuffle=True)  # Training data loader
test_data_loader = torch.utils.data.DataLoader(fashion_mnist_test_data, batch_size=64, shuffle=False)  # Test data loader

# Setting the computation device based on availability
# If a GPU with CUDA support is available, it will be used; otherwise, it falls back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Prints the device being used for model training and evaluation

# Setting up the Vision Transformer model along with loss function and optimizer
# The model's parameters like patch size, number of layers, and attention heads are specified
model = VisionTransformer(size_of_patch=4, depth=6, attn_heads=4).to(device)  # Instantiating the model and moving it to the selected device

loss_function = nn.CrossEntropyLoss()  # CrossEntropyLoss is commonly used for classification tasks

# Optimizer is responsible for updating the model parameters based on the computed gradients
# Adam optimizer is used here with a learning rate of 0.0001
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Initialize lists to keep track of loss per epoch
train_loss_history = []
test_loss_history = []

# Start training loop
for epoch in range(20):  # Iterate over 20 epochs
    train_loss = 0.0
    test_loss = 0.0

    # Set model to training mode
    model.train()

    # Loop over each batch from the training set
    for i, batch in enumerate(train_data_loader):
        images, labels = batch  # Unpack the batch
        images = images.to(device)  # Move images to the device (GPU or CPU)
        labels = labels.to(device)  # Move labels to the device

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing images to the model
        predicted_output = model(images)

        # Calculate the loss
        loss = loss_function(predicted_output, labels)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

        # Update running training loss
        train_loss += loss.item()

    # Switch model to evaluation mode
    model.eval()

    # Disable gradient calculation for validation, saves memory and computations
    with torch.no_grad():
        # Loop over each batch from the test set
        for i, batch in enumerate(test_data_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: compute predicted outputs by passing images to the model
            predicted_output = model(images)

            # Calculate the loss
            loss = loss_function(predicted_output, labels)

            # Update running test loss
            test_loss += loss.item()

    # Calculate average losses
    train_loss = train_loss / len(train_data_loader)
    test_loss = test_loss / len(test_data_loader)

    # Update our loss history lists
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)

    # Print training and test loss statistics
    print(f'Epoch {epoch}, Train loss {train_loss:.4f}, Test loss {test_loss:.4f}')

# Save the trained model's state dictionary
# This saves the model after it has been trained for 20 epochs, capturing all learned parameters
torch.save(model.state_dict(), 'vision_transformer_model.pth')

# The file 'vision_transformer_model.pth' can later be loaded to recreate the model with the trained parameters

import matplotlib.pyplot as plt

# Plotting the loss curves for both training and testing phases
# This visualization helps in understanding how the model learns over epochs

# Plot training loss history
plt.plot(range(20), train_loss_history, '-', linewidth=3, label='Train Loss')  # Training loss curve

# Plot testing loss history
plt.plot(range(20), test_loss_history, '-', linewidth=3, label='Test Loss')  # Test loss curve

# Labeling the x-axis as 'Epoch'
plt.xlabel('Epoch')

# Labeling the y-axis as 'Loss'
plt.ylabel('Loss')

# Enabling grid for better readability of the plot
plt.grid(True)

# Adding a legend to differentiate between training and test loss
plt.legend()

# Displaying the plot
plt.show()

# Re-initializing the VisionTransformer model with the same configuration as during training
model = VisionTransformer(size_of_patch=4, depth=6, attn_heads=4).to(device)

# Loading the trained model parameters from the saved file
model.load_state_dict(torch.load('vision_transformer_model.pth'))

# Comments:
# 'VisionTransformer' is used in place of 'ViT' to reflect the updated class name.
# 'size_of_patch', 'depth', and 'attn_heads' parameters are specified to match the model's initial configuration.
# The model is moved to the appropriate 'device' (GPU or CPU) before loading the state dict.
# 'vision_transformer_model.pth' is the filename used to save the model, ensuring consistency.

# Switch the model to evaluation mode to disable dropout and batch normalization effects
model.eval()

# Initialize counters for correct predictions and the total number of samples
correct_predictions = 0
total_samples = 0

# Disable gradient computations for evaluation, which reduces memory usage and speeds up the process
with torch.no_grad():
    # Iterate over the test data loader to process the batches
    for batch in test_data_loader:
        images, labels = batch  # Unpack the batch to get the images and labels
        images = images.to(device)  # Transfer the images to the configured device
        labels = labels.to(device)  # Transfer the labels to the configured device

        # Forward pass: Compute the predicted outputs by passing images through the model
        predicted_output = model(images)
        # Get the predicted classes by finding the maximum value in the logits returned by the model
        _, predicted_classes = torch.max(predicted_output, 1)

        # Update the total number of samples processed
        total_samples += labels.size(0)
        # Update the total number of correct predictions
        correct_predictions += (predicted_classes == labels).sum().item()

# Calculate the test set accuracy by dividing the number of correct predictions by the total number of samples
test_accuracy = correct_predictions / total_samples

# Print the test set accuracy
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

import random
num_samples_to_visualize = 3
# Generate random indices based on the test dataset size
random_indices = random.sample(range(len(fashion_mnist_test_data)), num_samples_to_visualize)

# Lists to hold images and labels for visualization
images_to_visualize = []
labels_to_visualize = []

# Retrieve and store images and labels using the random indices
for i in random_indices:
    # Fetching the image and label from the test dataset
    image, label = fashion_mnist_test_data[i]
    # Adding a batch dimension to the image for model processing
    images_to_visualize.append(image.unsqueeze(0))
    # Storing the label
    labels_to_visualize.append(label)

# Concatenating list of images into a single tensor and transferring to the device
images_to_visualize = torch.cat(images_to_visualize, dim=0).to(device)
# Creating a tensor from the list of labels and transferring to the device
labels_to_visualize = torch.tensor(labels_to_visualize).to(device)

# Pass the images through the model and apply softmax to obtain predicted probabilities
predicted_probabilities = torch.nn.functional.softmax(model(images_to_visualize), dim=1)

# Plotting images and predicted probabilities
for i in range(num_samples_to_visualize):
    plt.figure(figsize=(8, 4))

    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(images_to_visualize[i].squeeze().cpu(), cmap='gray')
    plt.title(f'True Label: {labels_to_visualize[i].item()}')

    # Display a bar chart of predicted probabilities
    plt.subplot(1, 2, 2)
    plt.bar(range(10), predicted_probabilities[i].detach().cpu().numpy())
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Predicted Probabilities')
    plt.show()

"""concluson : Vision Transformer (ViT) model on the Fashion-MNIST dataset. This includes defining the model, loading the dataset, training the model across epochs while monitoring loss, saving the trained model, and evaluating its accuracy on the test set. The process also visualizes the model's predictions for a few sample images. The implementation showcases the ViT model's capability to handle image classification tasks, demonstrating transformers' applicability beyond natural language processing to computer vision. This work exemplifies a complete workflow from data preparation to model evaluation, highlighting the model's effectiveness through accuracy metrics and visual insights into its predictions."""