# style_transfer_mvp.py
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os

print("Style Transfer MVP - Starting up...")
print(f"TensorFlow version: {tf.__version__}")

# =============================================================================
# 1. FIXED PARAMETERS (Edit these for your specific images)
# =============================================================================
CONTENT_PATH = "content.tif"    # Your photography TIFF file
STYLE_PATH = "style.jpg"         # The artistic style image (JPEG or TIFF)
OUTPUT_PATH = "output.tif"      # Where to save the result

# Quality/Performance trade-offs
MAX_DIM = 512                    # Maximum dimension (longest side)
CONTENT_WEIGHT = 1e3             # How much to preserve original content
STYLE_WEIGHT = 8e-1              # How strong to apply the style
OPTIMIZER_STEPS = 1000           # More steps = better quality but slower

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def load_and_process_img(path, target_size=None):
    """Load and preprocess an image for the VGG network.
    
    If target_size is provided, resize to that. Otherwise, scale
    based on MAX_DIM.
    """
    print(f"Loading image: {path}")
    
    # Open with PIL and ensure it's RGB
    img = Image.open(path).convert('RGB')
    original_size = img.size
    print(f"  Original size: {original_size}")

    # Calculate new size
    new_size = target_size
    if new_size is None:
        scale = MAX_DIM / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    print(f"  Resized to: {new_size}")
    
    # Convert to numpy array and add batch dimension
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess for VGG19 (same as used during training)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    return tf.constant(img_array)

def deprocess_img(tensor):
    """Convert processed tensor back to PIL Image"""
    tensor = tensor.numpy()
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Remove batch dimension
    
    # Reverse VGG19 preprocessing
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.68
    tensor = tensor[:, :, ::-1]  # BGR to RGB
    
    # Clip to valid image range and convert to uint8
    tensor = np.clip(tensor, 0, 255).astype('uint8')
    return Image.fromarray(tensor)

# =============================================================================
# 3. MODEL & LOSS FUNCTIONS
# =============================================================================
def get_model():
    """Create our model with access to intermediate layers"""
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
    """Calculate Gram matrix for style representation"""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def style_loss(style_outputs, style_targets):
    """Calculate style loss"""
    loss = tf.add_n([
        tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
        for name in style_outputs.keys()
    ])
    return loss / len(style_outputs)

def content_loss(content_outputs, content_targets):
    """Calculate content loss"""
    return tf.add_n([
        tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
        for name in content_outputs.keys()
    ])

def compute_loss(model, loss_weights, generated_image, content_targets, style_targets):
    """Compute total loss"""
    # Get model outputs for the generated image
    model_outputs = model(generated_image)
    
    # Split outputs into style and content
    style_outputs = model_outputs[:len(style_targets)]
    content_outputs = model_outputs[len(style_targets):]
    
    # Convert to dictionaries for easier handling
    style_outputs = {
        style_name: output for style_name, output in zip(style_targets.keys(), style_outputs)
    }
    content_outputs = {
        content_name: output for content_name, output in zip(content_targets.keys(), content_outputs)
    }
    
    # Calculate individual losses
    style_loss_value = style_loss(style_outputs, style_targets)
    content_loss_value = content_loss(content_outputs, content_targets)
    
    # Weighted total loss
    total_loss = (
        loss_weights['style'] * style_loss_value + 
        loss_weights['content'] * content_loss_value
    )
    
    return total_loss, style_loss_value, content_loss_value

# =============================================================================
# 4. TARGET COMPUTATION
# =============================================================================
def get_targets(model, content_image, style_image):
    """Precompute the content and style targets"""
    print("Computing content and style targets...")
    
    # Get model outputs for content image
    content_outputs = model(content_image)
    style_layers = content_outputs[:5]  # First 5 outputs are style layers
    content_layers = content_outputs[5:]  # Last 1 output is content layer
    
    style_targets = {
        f'style_{i}': output for i, output in enumerate(style_layers)
    }
    content_targets = {
        f'content_{i}': output for i, output in enumerate(content_layers)
    }
    
    # Get model outputs for style image
    style_outputs = model(style_image)
    style_outputs = style_outputs[:5]  # Only need style layers
    
    style_targets = {
        f'style_{i}': output for i, output in enumerate(style_outputs)
    }
    
    return content_targets, style_targets

# =============================================================================
# 5. TRAINING STEP
# =============================================================================
@tf.function
def train_step(model, loss_weights, generated_image, content_targets, style_targets, optimizer):
    """Single training step"""
    with tf.GradientTape() as tape:
        total_loss, style_loss, content_loss = compute_loss(
            model, loss_weights, generated_image, content_targets, style_targets
        )
    
    # Compute gradient and apply to the generated image
    grad = tape.gradient(total_loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    
    # Clamp pixel values
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=255.0))
    
    return total_loss, style_loss, content_loss

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
def main():
    print("\n" + "="*50)
    print("NEURAL STYLE TRANSFER MVP")
    print("="*50)
    
    # Check if files exist
    if not os.path.exists(CONTENT_PATH):
        print(f"ERROR: Content image not found at {CONTENT_PATH}")
        return
    if not os.path.exists(STYLE_PATH):
        print(f"ERROR: Style image not found at {STYLE_PATH}")
        return
    
    # Load images
    print("\n1. Loading images...")
    content_image = load_and_process_img(CONTENT_PATH)
    # Resize style image to match content image dimensions
    content_shape = tf.shape(content_image)[1:3]
    style_image = load_and_process_img(STYLE_PATH, target_size=(content_shape[1], content_shape[0]))
    
    # Create model
    print("\n2. Loading VGG19 model...")
    model, style_layers, content_layers = get_model()
    
    # Precompute targets
    print("\n3. Computing style and content targets...")
    content_targets, style_targets = get_targets(model, content_image, style_image)
    
    # Initialize generated image (start with content image)
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    
    # Setup optimizer and loss weights
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    loss_weights = {'style': STYLE_WEIGHT, 'content': CONTENT_WEIGHT}
    
    print(f"\n4. Starting optimization for {OPTIMIZER_STEPS} steps...")
    print("   (This will take several minutes)")
    start_time = time.time()
    
    # Training loop
    for step in range(OPTIMIZER_STEPS):
        total_loss, style_loss_val, content_loss_val = train_step(
            model, loss_weights, generated_image, content_targets, style_targets, optimizer
        )
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"   Step {step:4d}/{OPTIMIZER_STEPS}: "
                  f"Total Loss: {total_loss:.2e}, "
                  f"Style: {style_loss_val:.2e}, "
                  f"Content: {content_loss_val:.2e}, "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n5. Optimization completed in {total_time:.1f} seconds")
    
    # Save final result
    print("\n6. Saving result...")
    final_image = deprocess_img(generated_image)
    final_image.save(OUTPUT_PATH, format='TIFF', compression=None)
    print(f"   ✓ Saved to: {OUTPUT_PATH}")
    
    print("\n" + "="*50)
    print("DONE! Check your output TIFF file.")
    print("="*50)

if __name__ == "__main__":
    main()