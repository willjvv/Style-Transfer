import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image

print("Style Transfer - Starting up...")
print(f"TensorFlow version: {tf.__version__}")

# --- 1. Configuration (Edit these for your specific images) ---

CONTENT_PATH = "content.tif"    # Your photography TIFF file
STYLE_PATH = "style.jpg"        # The artistic style image (JPEG or TIFF)
OUTPUT_PATH = "output.tif"      # Where to save the result

# Quality/Performance trade-offs
MAX_DIM = 2048                    # Maximum dimension (longest side)
CONTENT_WEIGHT = 1e2              # How much to prioritize original content
STYLE_WEIGHT = 1e-2               # How much to prioritize the artistic style
TOTAL_VARIATION_WEIGHT = 10       # How much to penalize image noise (for smoothness)
OPTIMIZER_STEPS = 1000            # More steps = better quality, but slower

# --- 2. Image Handling ---

def load_and_process_img(path):
    """
    Load and preprocess an image for the VGG network.

    Args:
        path (str): The file path to the image.

    Returns:
        tf.Tensor: A preprocessed image tensor.
    """
    print(f"Loading image: {path}")
    
    # Open with PIL and ensure it's RGB
    img = Image.open(path).convert('RGB')
    original_size = img.size
    print(f"  Original size: {original_size}")
    
    # Calculate new size while maintaining aspect ratio
    scale = MAX_DIM / max(img.size)
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    print(f"  Resized to: {new_size}")
    
    # Convert to a numpy array and normalize to the 0-1 range
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return tf.constant(img_array)

def deprocess_img(tensor):
    """
    Convert a processed tensor back to a PIL Image.

    Args:
        tensor (tf.Tensor): The image tensor in the [0, 1] range.

    Returns:
        PIL.Image.Image: The deprocessed image.
    """
    tensor = tensor.numpy()
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Remove batch dimension
    
    # Convert from 0-1 back to 0-255
    tensor = tensor * 255.0
    tensor = np.clip(tensor, 0, 255).astype('uint8')
    return Image.fromarray(tensor)

# --- 3. Model and Loss Functions ---

def get_model():
    """
    Create a VGG19 model with access to intermediate layer outputs.

    Returns:
        tuple: A Keras model, a list of style layer names, and a list of content layer names.
    """
    # Load pre-trained VGG19
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Get outputs from specific layers
    content_layers = ['block5_conv2']
    style_layers = [
        'block1_conv1',
        'block2_conv1', 
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    model = tf.keras.Model([vgg.input], outputs)
    
    return model, style_layers, content_layers

def gram_matrix(input_tensor):
    """
    Calculate the Gram matrix for style representation.

    Args:
        input_tensor (tf.Tensor): A feature map tensor from a convolutional layer.

    Returns:
        tf.Tensor: The Gram matrix.
    """
    # Flatten the spatial dimensions and compute correlations
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.cast(tf.shape(a)[0], tf.float32)
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / n

def style_loss(style_outputs, style_targets):
    """
    Calculate the style loss.

    Args:
        style_outputs (dict): Dictionary of style feature maps from the generated image.
        style_targets (dict): Dictionary of style feature maps from the style image.

    Returns:
        tf.Tensor: The total style loss.
    """
    loss = 0
    for name in style_outputs.keys():
        style_output = style_outputs[name]
        style_target = style_targets[name]
        
        # Calculate Gram matrices
        gram_style = gram_matrix(style_output)
        gram_target = gram_matrix(style_target)
        
        # Normalize by number of layers and features
        num_elements = tf.cast(tf.reduce_prod(style_output.shape[1:-1]), tf.float32)
        num_channels = tf.cast(style_output.shape[-1], tf.float32)
        layer_loss = tf.reduce_mean((gram_style - gram_target) ** 2)
        loss += layer_loss / (4.0 * (num_channels ** 2) * (num_elements ** 2))
    
    return loss / len(style_outputs)

def content_loss(content_outputs, content_targets):
    """
    Calculate the content loss.

    Args:
        content_outputs (dict): Dictionary of content feature maps from the generated image.
        content_targets (dict): Dictionary of content feature maps from the content image.

    Returns:
        tf.Tensor: The total content loss.
    """
    loss = 0
    for name in content_outputs.keys():
        content_output = content_outputs[name]
        content_target = content_targets[name]
        loss += tf.reduce_mean((content_output - content_target) ** 2)
    return loss / len(content_outputs)

def compute_loss(model, loss_weights, generated_image, content_targets, style_targets):
    """
    Compute the total weighted loss for style transfer.

    Args:
        model (tf.keras.Model): The VGG19 model.
        loss_weights (dict): Weights for style and content loss.
        generated_image (tf.Variable): The image being optimized.
        content_targets (dict): Precomputed content feature maps.
        style_targets (dict): Precomputed style feature maps.

    Returns:
        tuple: Total loss, style loss, content loss, and total variation loss.
    """
    # Get model outputs for the generated image
    model_outputs = model(generated_image * 255.0)  # Scale for VGG preprocessing
    
    # Split outputs into style and content
    style_outputs = model_outputs[:len(style_targets)]
    content_outputs = model_outputs[len(style_targets):]
    
    # Convert to dictionaries for easier handling
    style_outputs_dict = {
        f'style_{i}': output for i, output in enumerate(style_outputs)
    }
    content_outputs_dict = {
        f'content_{i}': output for i, output in enumerate(content_outputs)
    }
    
    # Calculate individual losses
    style_loss_value = style_loss(style_outputs_dict, style_targets)
    content_loss_value = content_loss(content_outputs_dict, content_targets)
    total_variation_loss_value = tf.image.total_variation(generated_image)
    
    # Weighted total loss
    total_loss = (
        loss_weights['style'] * style_loss_value + 
        loss_weights['content'] * content_loss_value +
        loss_weights['total_variation'] * total_variation_loss_value
    )
    
    return total_loss, style_loss_value, content_loss_value, total_variation_loss_value

# --- 4. Target Computation ---

def get_targets(model, content_image, style_image):
    """
    Precompute the content and style target feature maps.

    Args:
        model (tf.keras.Model): The VGG19 model.
        content_image (tf.Tensor): The preprocessed content image.
        style_image (tf.Tensor): The preprocessed style image.

    Returns:
        tuple: A dictionary of content targets and a dictionary of style targets.
    """
    print("Computing content and style targets...")
    
    # Scale images for VGG preprocessing (VGG expects 0-255)
    content_scaled = content_image * 255.0
    style_scaled = style_image * 255.0
    
    # Get model outputs
    content_outputs = model(content_scaled)
    style_outputs = model(style_scaled)
    
    # Split outputs into style and content targets based on the model definition
    style_targets = {
        f'style_{i}': output for i, output in enumerate(style_outputs[:5])
    }
    content_targets = {
        f'content_{i}': output for i, output in enumerate(content_outputs[5:])
    }
    
    return content_targets, style_targets

# --- 5. Training Step ---

@tf.function
def train_step(model, loss_weights, generated_image, content_targets, style_targets, optimizer):
    """
    Perform a single optimization step.

    Args:
        model (tf.keras.Model): The VGG19 model.
        loss_weights (dict): Weights for style and content loss.
        generated_image (tf.Variable): The image being optimized.
        content_targets (dict): Precomputed content feature maps.
        style_targets (dict): Precomputed style feature maps.
        optimizer (tf.optimizers.Optimizer): The optimizer.

    Returns:
        tuple: Total loss, style loss, content loss, and total variation loss for the step.
    """
    with tf.GradientTape() as tape:
        losses = compute_loss(
            model, loss_weights, generated_image, content_targets, style_targets
        )
    total_loss = losses[0]
    # Compute gradient and apply with clipping
    grad = tape.gradient(total_loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    
    # Clamp pixel values to the valid 0-1 range
    generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))
    
    return losses

# --- 6. Main Execution ---

def main():
    """Main function to run the style transfer process."""
    print("\n" + "="*50 + "\nNEURAL STYLE TRANSFER\n" + "="*50)
    
    # Configure GPU for memory growth to avoid pre-allocating all memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"\nFound {len(gpus)} Physical GPU(s), Configured {len(logical_gpus)} Logical GPU(s)")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"GPU Memory Growth Error: {e}")
    else:
        print("\nWARNING: No GPU found. The script will run on the CPU and may be very slow.")
    
    # Check if files exist
    if not os.path.exists(CONTENT_PATH):
        print(f"ERROR: Content image not found at {CONTENT_PATH}")
        return
    if not os.path.exists(STYLE_PATH):
        print(f"ERROR: Style image not found at {STYLE_PATH}")
        return
    
    # Load images into the 0-1 range
    print("\n1. Loading images...")
    content_image = load_and_process_img(CONTENT_PATH)
    style_image = load_and_process_img(STYLE_PATH)
    
    # Create model
    print("\n2. Loading VGG19 model...")
    model, style_layers, content_layers = get_model()
    
    # Precompute targets
    print("\n3. Computing style and content targets...")
    content_targets, style_targets = get_targets(model, content_image, style_image)
    
    # Initialize the generated image from the content image
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    
    # Configure the optimizer
    optimizer = tf.optimizers.Adam(
        learning_rate=0.02,
        beta_1=0.99, 
        epsilon=1e-1
    )
    
    loss_weights = {
        'style': STYLE_WEIGHT, 
        'content': CONTENT_WEIGHT,
        'total_variation': TOTAL_VARIATION_WEIGHT
    }
    
    print(f"\n4. Starting optimization for {OPTIMIZER_STEPS} steps...")
    print("   (This will take several minutes)")
    start_time = time.time()
    
    # Training loop
    for step in range(OPTIMIZER_STEPS):
        (total_loss, 
         style_loss_val, 
         content_loss_val, 
         tv_loss_val) = train_step(
            model, loss_weights, generated_image, content_targets, style_targets, optimizer
        )
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"   Step {step:4d}/{OPTIMIZER_STEPS}: "
                  f"Total Loss: {float(total_loss.numpy()):.2e}, "
                  f"Style: {float(style_loss_val.numpy()):.2e}, "
                  f"Content: {float(content_loss_val.numpy()):.2e}, "
                  f"TV: {float(tv_loss_val[0].numpy()):.2e}, "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n5. Optimization completed in {total_time:.1f} seconds")
    
    # Save the final image
    print("\n6. Saving result...")
    final_image = deprocess_img(generated_image)
    final_image.save(OUTPUT_PATH, format='TIFF', compression=None)
    print(f"   ✓ Saved to: {OUTPUT_PATH}")
    
    print("\n" + "="*50 + "\nDONE! Check your output TIFF file.\n" + "="*50)

if __name__ == "__main__":
    main()