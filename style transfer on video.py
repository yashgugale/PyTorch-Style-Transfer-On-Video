# Import the necessary packages:
import numpy as np								# Numpy computations
import cv2										# OpenCV computations
from PIL import Image							# Python Image Library for image processing
import matplotlib.pyplot as plt 				# Plotting
import torch 									# Neural network computation
from torch import optim							# optimizer to minimize the loss function
from torchvision import transforms, models		# Transformations on images and pre-trained models
import os, os.path 								# To count the number of image files, make dirs, etc
import sys										# To read the command line arguments

# Names for the directories storing the input content and style frames, style transferred frames and processed video:
content_frame_dir = "input_content_frames"
style_frame_dir = "input_style_frames"
style_transferred_frame_dir = "output_style_transferred_frames"
output_video_dir = "output_processed_video"

# Store total number of frames in the input video:
total_frames = 0

# LOAD VGG19 (features only):
# vgg19.features 	: It consists of all the convolutional and pooling layers
# vgg19.classifier 	: It consists of the 3 linear classifier layers at the end

# We load in the pre-trained model and freeze the weights:
styleTransferModel = models.vgg19(pretrained=True).features

# Freeze all the VGG parameters since we are only optimizing the target image:
for params in styleTransferModel.parameters():
	params.requires_grad_(False)

# Check if GPU is available:
if torch.cuda.is_available():
	device = torch.device("cuda")
	print("Running on GPU")
else:
	device = torch.device("cpu")
	print("Running on CPU")

# Move the model to the GPU if available:
styleTransferModel.to(device)

# Load the content and style images:
def load_image(image_path, max_size=400, shape=None):
	# Load in an image and make sure that it is <= 400 pixels in the X-Y dimension:
	# Convert the image to RGB:
	image = Image.open(image_path).convert('RGB')

	# Resize the image as a large image will slow down processing:
	if max(image.size) > max_size:
		img_size = max_size
	else:
		img_size = max(image.size)

	if shape is not None:
		img_size = shape

	# Create and apply the necessary transformations:
	img_transform = transforms.Compose([
										transforms.Resize(img_size),
										transforms.ToTensor(),
										transforms.Normalize((0.485, 0.456, 0.406),
															(0.229, 0.224, 0.225))])
	# Discard the transparent alpha channel (that's the :3) and add the batch dimension:
	image = img_transform(image)[:3, :, :].unsqueeze(0)

	return image

# CONTENT AND STYLE FEATURES:
# Map the layer names to the names given in the paper:
def get_features(image, model, layers=None):
	# Run an image forward through a model and get the features for a set of layers:

	# Layers for the content and style representation of an image:
	if layers is None:
		layers = {'0': 'conv1_1',
				  '5': 'conv2_1',
				  '10': 'conv3_1',
				  '19': 'conv4_1',
				  '21': 'conv4_2',		# Content representation
				  '28': 'conv5_1',}

	features = {}
	x = image
	# model._modules is a dictionary holding each module in the model:
	for name, layer in model._modules.items():
		x = layer(x)
		if name in layers:
			features[layers[name]] = x

	return features

# GRAM MATRIX:
# Define the gram matrix of the tensor:
def gram_matrix(tensor):
	# Calculate the Gram Matrix of a given tensor:
	# Get the batch_size, depth, height and width of the image:
	batch_size, depth, height, width = tensor.size()
	
	# Vectorize the input image tensor and add all the feature maps:
	tensor = tensor.view(depth, height * width)	
	# Transpose the image tensor:
	tensor_t = tensor.t()
	# Compute the gram matrix by multiplying the matrix by its transpose:
	gram = torch.mm(tensor, tensor_t)

	# Return the gram matrix:
	return gram

def convert_video_to_frames(video_directory):
	# Create a directory to save the individual frames and perform error checking:
	try:
		os.mkdir(content_frame_dir)
	except OSError:
		print("Failed to create directory: %s" %content_frame_dir)
	else:
		print("Successfully created directory: %s" %content_frame_dir)
		
	# Load the input video:
	capture_video = cv2.VideoCapture(video_directory)
	# Read the input frame:
	success, frame = capture_video.read()
	# Set counter for number of frames read:
	count = 1

	while success:
		# Save the frame that is read:
		cv2.imwrite(content_frame_dir + '/frame_%d.jpg' % count, frame)
		# Read the next input frame:
		success, frame = capture_video.read()
		print("Read a new frame: ", count, "Success: ", success)
		count += 1
	
	# Set the total frame count:
	global total_frames
	total_frames = count

# Function to un-normalize an image and convert from a Tensor image to a NumPy image for display or writing to disk:
def img_convert(tensor):
	# Display a tensor as an image:

	image = tensor.to("cpu").clone().detach()
	image = image.numpy().squeeze()
	image = image.transpose(1, 2, 0)
	image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
	image = image.clip(0, 1)

	return image

def apply_style_transfer(content_img_dir, style_img_dir):
	# Create a directory to save the style transferred frames and perform error checking:
	try:
		os.mkdir(style_transferred_frame_dir)
	except OSError:
		print("Failed to create directory: %s" %style_transferred_frame_dir)
	else:
		print("Successfully created directory: %s" %style_transferred_frame_dir)
		
	content_image_count = 1
	style_image_count = 1
	
	# Retrieve the total number of content image frames and style frames:
	num_content_imgs = len([name for name in os.listdir(content_img_dir) if os.path.isfile(os.path.join(content_img_dir, name))])
	num_style_imgs = len([name for name in os.listdir(style_img_dir) if os.path.isfile(os.path.join(style_img_dir, name))])
	# Divide the style images equally among the input frames:
	frames_with_current_style = num_content_imgs // (num_style_imgs - 1)

	# Since there are 249 frames in the test video, you can use for loop, but prefer using while to avoid using magic numbers (hardcoding):
	while(content_image_count <= num_content_imgs): 
	#for i in range(1, total_frames):
		# Load in the content and style images and move them to the GPU if available:
		content_image = load_image(content_img_dir + '/frame_' + str(content_image_count) + '.jpg').to(device)
		if(content_image_count % frames_with_current_style == 0):
			style_image_count += 1

		style_image = load_image(style_img_dir + '/style_' + str(style_image_count) + '.jpg', shape=content_image.shape[-2:]).to(device)

		# Retrieve the features:
		content_features = get_features(content_image, styleTransferModel)
		style_features = get_features(style_image, styleTransferModel)

		# Calculate the gram matrix for each of our style representations:
		style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

		# We create a 'target image'. Note that, we are starting with the content image and cloning it instead of creating an image with white filter:
		# We want to update our image based on the total loss and so we will turn on the gradients:
		target_image = content_image.clone().requires_grad_(True).to(device)

		# LOSS AND WEIGHTS:
		# We assign weights for each style layer. Weighting earlier layers more will result in *larger* style artifacts:
		# Notice we are excluding `conv4_2`, i.e. our content representation:
		style_weights = {'conv1_1': 1.,			# More style will come from earlier layers as they are weighted more
		                 'conv2_1': 0.8,
		                 'conv3_1': 0.2,
		                 'conv4_1': 0.2,
		                 'conv5_1': 0.2}		# Less style from later layers

		content_weight = 1  # alpha
		style_weight = 1e8  # beta, 1e6 = 1000000.0

		# Iteration hyperparameters:
		# Update the target image (as we update the model.parameters() in the classifiers):
		optimizer = optim.Adam([target_image], lr=0.003)
		# Number of iterations to update your image:
		steps = 2
		show_every = 1000
		for ii in range(1, steps+1):

			# Get the features from the target image:
			target_features = get_features(target_image, styleTransferModel)
			
			# 1. Calculate the content loss:
			content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

			# 2. Calculate the style loss:
			style_loss = 0
			# iterate through each style layer and add to the style loss:
			for layer in style_weights:
				# Get the "target" style representation for the layer:
				target_feature = target_features[layer]
				batch_size, depth, height, width = target_feature.shape

				# Calculate the target gram matrix:
				target_gram = gram_matrix(target_feature)

				# Get the "style" from the style gram matrices computed earlier:
				style_gram = style_grams[layer]

				# the style loss for one layer, weighted appropriately:
				layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

				# Add to the style loss:
				style_loss += layer_style_loss / (depth * height * width)

			# Calculate the total loss:
			total_loss = content_weight * content_loss + style_weight * style_loss

			# Update the target image:
			optimizer.zero_grad()			# zero out the gradients from previous iterations
			total_loss.backward()			# Backpropagate the loss
			optimizer.step()				# Update the target image

			# Display the intermediate results if required:
			#if ii % show_every == 0:
				#print("Total loss: ", total_loss.item())
				#plt.imshow(img_convert(target_image))
				#plt.show()

		# Save the style transferred image:
		target_image = img_convert(target_image)
		plt.imsave(style_transferred_frame_dir + '/st_frame_%d.jpg' % content_image_count, target_image)
		print("completed style transfer on image: ", content_image_count)
		content_image_count += 1
		if(content_image_count <= total_frames):
			continue



# Convert the final images to video:
def img_to_video(st_output_dir):
	# Create a directory to save the final processed video and perform error checking:
	try:
		os.mkdir(output_video_dir)
	except OSError:
		print("Failed to create directory: %s" %output_video_dir)
	else:
		print("Successfully created directory: %s" %output_video_dir)
		
	img = cv2.imread(st_output_dir + '/st_frame_1.jpg')
	height, width, layers = img.shape
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	video = cv2.VideoWriter(output_video_dir + '/style_transfered_video.avi', fourcc, 25, (width, height))
	# Count the number of frames and set the for loop:
	for i in range(1, total_frames):
		video.write(cv2.imread(st_output_dir + '/st_frame_' + str(i) + '.jpg'))

	cv2.destroyAllWindows()
	video.release()

def main():
	
	# 1. Convert the input video to frames (get the path of the input video file as the command line argument):
	convert_video_to_frames(sys.argv[1])	

	# 2. Apply style transfer on the input frames:
	apply_style_transfer(content_frame_dir, style_frame_dir)
	
	# 3. Convert the Style Transferred images back to video:
	img_to_video(style_transferred_frame_dir)
	
# Call the main function as soon as the python script is run:
if __name__ == "__main__":
	main()