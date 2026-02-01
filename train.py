#train.py
import torch
import torch.nn as nn

from MyCnn import ConvNet
from myCnn2 import ConvNet as ConvNet2
from cnn import train, plot_metrics, run
from data import save_activations, visualize_activations

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 15
batch_size = 64
learning_rate = 0.001

#model = ConvNet().to(device)
model = ConvNet2().to(device)
loss_F = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
losses, accuracies = train(model, device, loss_F, optimizer, num_epochs, batch_size)
plot_metrics(losses, accuracies, num_epochs)

# # Test the model
# model.load_state_dict(torch.load(f'cnn_epoch{num_epochs}.pth'))
# model.eval()

# img_path = f"../TestData/hina9.png"
# pred, model = run(model, device, img_path)
# print(f"Predicted label: {pred}")

# # visualize_activations(model.conv1_x, "conv1")
# # visualize_activations(model.convStride1_x, "convStride1")
# # visualize_activations(model.conv2_x, "conv2")
# # visualize_activations(model.convStride2_x, "convStride2")

# # visualize_activations(model.view_x, "flatten")
# # visualize_activations(model.fc1_x, "fc1")
# # visualize_activations(model.fc2_x, "fc2")
# # visualize_activations(model.fc3_x, "fc3")

# save_activations(model.conv1_x, "1_conv1")
# save_activations(model.convStride1_x, "2_convStride1")
# save_activations(model.conv2_x, "3_conv2")
# save_activations(model.convStride2_x, "4_convStride2")

# save_activations(model.view_x, "5_flatten")
# save_activations(model.fc1_x, "6_fc1")
# save_activations(model.fc2_x, "7_fc2")
# save_activations(model.fc3_x, "8_fc3")

# for i in range(10):
#     if i == 4:
#         for j in range(1):
#             img_path = f"../TestData/{i}.png"
#             pred, model = run(model, device, img_path)
#             print(f"Image: {i}  Predicted label: {pred}")

#             visualize_activations(model.conv1_x, "conv1")
#             visualize_activations(model.convStride1_x, "convStride1")
#             visualize_activations(model.conv2_x, "conv2")
#             visualize_activations(model.convStride2_x, "convStride2")

#             visualize_activations(model.view_x, "flatten")
#             visualize_activations(model.fc1_x, "fc1")
#             visualize_activations(model.fc2_x, "fc2")
#             visualize_activations(model.fc3_x, "fc3")
#             #print(f"Testing image of digit {i}, run {j+1}")
#         j += 1
    
#     i += 1
