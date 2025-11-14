import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image

print("Style Transfer with Memory Optimization - Starting up...")
print(f"TensorFlow version: {tf.__version__}")

# --- Configuration ---
CONTENT_PATH = "content.tif"
STYLE_PATH = "style.jpg"
OUTPUT_PATH = "output.tif"
MAX_DIM = 2048
CONTENT_WEIGHT = 1e2
STYLE_WEIGHT = 1e2
TOTAL_VARIATION_WEIGHT = 1
SCALES = [0.25, 0.5, 1.0]
STEPS_PER_SCALE = 300
INITIAL_STEPS = 150

# --- Gradient Checkpointing Model ---
def get_model_with_checkpoints():
    """
    Create VGG19 with gradient checkpointing to reduce memory usage.
    """
    # Load base model
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Define layers for style and content
    content_layers = ['block5_conv2']
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    
    # Create a custom model that uses gradient checkpointing
    inputs = tf.keras.Input(shape=(None, None, 3))
    
    # VGG19 layers with explicit naming for checkpointing
    x = inputs
    layer_outputs = {}
    
    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    layer_outputs['block1_conv1'] = x
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    layer_outputs['block2_conv1'] = x
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    layer_outputs['block3_conv1'] = x
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # Block 4
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    layer_outputs['block4_conv1'] = x
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    # Block 5
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    layer_outputs['block5_conv1'] = x
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    layer_outputs['block5_conv2'] = x
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    # Create model with checkpointing
    model = tf.keras.Model(inputs, [layer_outputs[layer] for layer in style_layers + content_layers])
    
    # Load pre-trained weights
    for layer in model.layers:
        if layer.name in [l.name for l in vgg.layers]:
            try:
                layer.set_weights(vgg.get_layer(layer.name).get_weights())
            except:
                pass
    
    return model, style_layers, content_layers

# --- Image Processing (same as before) ---
def load_and_process_img(path, scale=1.0):
    print(f"Loading image: {path}")
    img = Image.open(path).convert('RGB')
    original_size = img.size
    print(f"  Original size: {original_size}")
    
    current_max_dim = int(MAX_DIM * scale)
    scale_factor = current_max_dim / max(img.size)
    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    print(f"  Resized to: {new_size} (scale: {scale})")
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return tf.constant(img_array)

def deprocess_img(tensor):
    tensor = tensor.numpy()
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    tensor = tensor * 255.0
    tensor = np.clip(tensor, 0, 255).astype('uint8')
    return Image.fromarray(tensor)

def resize_tensor(tensor, new_size):
    img_array = tensor.numpy()[0]
    img = Image.fromarray((img_array * 255.0).astype('uint8'))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    resized_array = np.array(img, dtype=np.float32) / 255.0
    resized_array = np.expand_dims(resized_array, axis=0)
    return tf.constant(resized_array)

# --- Loss Functions (same as before) ---
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.cast(tf.shape(a)[0], tf.float32)
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / n

def style_loss(style_outputs, style_targets):
    loss = 0
    for name in style_outputs.keys():
        style_output = style_outputs[name]
        style_target = style_targets[name]
        gram_style = gram_matrix(style_output)
        gram_target = gram_matrix(style_target)
        num_elements = tf.cast(tf.reduce_prod(style_output.shape[1:-1]), tf.float32)
        num_channels = tf.cast(style_output.shape[-1], tf.float32)
        layer_loss = tf.reduce_mean((gram_style - gram_target) ** 2)
        loss += layer_loss / (4.0 * (num_channels ** 2) * (num_elements ** 2))
    return loss / len(style_outputs)

def content_loss(content_outputs, content_targets):
    loss = 0
    for name in content_outputs.keys():
        content_output = content_outputs[name]
        content_target = content_targets[name]
        loss += tf.reduce_mean((content_output - content_target) ** 2)
    return loss / len(content_outputs)

def compute_loss(model, loss_weights, generated_image, content_targets, style_targets):
    # Use gradient checkpointing for the forward pass
    model_outputs = tf.recompute_grad(model)(generated_image * 255.0)
    
    style_outputs = model_outputs[:len(style_targets)]
    content_outputs = model_outputs[len(style_targets):]
    
    style_outputs_dict = {f'style_{i}': output for i, output in enumerate(style_outputs)}
    content_outputs_dict = {f'content_{i}': output for i, output in enumerate(content_outputs)}
    
    style_loss_value = style_loss(style_outputs_dict, style_targets)
    content_loss_value = content_loss(content_outputs_dict, content_targets)
    total_variation_loss_value = tf.image.total_variation(generated_image)
    
    total_loss = (
        loss_weights['style'] * style_loss_value + 
        loss_weights['content'] * content_loss_value +
        loss_weights['total_variation'] * total_variation_loss_value
    )
    
    return total_loss, style_loss_value, content_loss_value, total_variation_loss_value

# --- Training with Memory Optimization ---
def create_train_step():
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    
    @tf.function
    def train_step(model, loss_weights, generated_image, content_targets, style_targets):
        with tf.GradientTape() as tape:
            losses = compute_loss(model, loss_weights, generated_image, content_targets, style_targets)
        total_loss = losses[0]
        grad = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))
        return losses
    
    return train_step

# --- Target Computation ---
def get_targets(model, content_image, style_image):
    print("Computing content and style targets...")
    content_scaled = content_image * 255.0
    style_scaled = style_image * 255.0
    
    # Use checkpointing for target computation too
    content_outputs = tf.recompute_grad(model)(content_scaled)
    style_outputs = tf.recompute_grad(model)(style_scaled)
    
    style_targets = {f'style_{i}': output for i, output in enumerate(style_outputs[:5])}
    content_targets = {f'content_{i}': output for i, output in enumerate(content_outputs[5:])}
    
    return content_targets, style_targets

# --- Process at Scale with Memory Monitoring ---
def process_at_scale(scale, model, train_step_func, initial_image=None):
    print(f"\n--- Processing at scale {scale} ---")
    
    current_content = load_and_process_img(CONTENT_PATH, scale)
    current_style = load_and_process_img(STYLE_PATH, scale)
    
    content_targets, style_targets = get_targets(model, current_content, current_style)
    
    if initial_image is not None:
        current_size = (current_content.shape[2], current_content.shape[1])
        resized_initial = resize_tensor(initial_image, current_size)
        generated_image = tf.Variable(resized_initial, dtype=tf.float32)
    else:
        generated_image = tf.Variable(current_content, dtype=tf.float32)
    
    loss_weights = {
        'style': STYLE_WEIGHT, 
        'content': CONTENT_WEIGHT,
        'total_variation': TOTAL_VARIATION_WEIGHT
    }
    
    if scale == SCALES[0]:
        steps = STEPS_PER_SCALE + INITIAL_STEPS
    else:
        steps = STEPS_PER_SCALE
    
    print(f"Running {steps} optimization steps...")
    start_time = time.time()
    
    for step in range(steps):
        losses = train_step_func(model, loss_weights, generated_image, content_targets, style_targets)
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            total_loss, style_loss_val, content_loss_val, tv_loss_val = losses
            print(f"   Scale {scale}: Step {step:4d}/{steps}: "
                  f"Total Loss: {float(total_loss.numpy()):.2e}, "
                  f"Style: {float(style_loss_val.numpy()):.2e}, "
                  f"Content: {float(content_loss_val.numpy()):.2e}, "
                  f"Time: {elapsed:.1f}s")
    
    scale_time = time.time() - start_time
    print(f"Scale {scale} completed in {scale_time:.1f} seconds")
    
    return generated_image

def clear_tensorflow_session():
    tf.keras.backend.clear_session()
    import gc
    gc.collect()

# --- Main Execution ---
def main():
    print("\n" + "="*60 + "\nNEURAL STYLE TRANSFER WITH MEMORY OPTIMIZATION\n" + "="*60)
    
    # Enhanced GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Set larger memory limit if needed
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
                )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"\nFound {len(gpus)} Physical GPU(s), Configured {len(logical_gpus)} Logical GPU(s)")
        except RuntimeError as e:
            print(f"GPU Configuration Error: {e}")
    else:
        print("\nWARNING: No GPU found. This will be very slow at 2048px.")
    
    if not os.path.exists(CONTENT_PATH):
        print(f"ERROR: Content image not found at {CONTENT_PATH}")
        return
    if not os.path.exists(STYLE_PATH):
        print(f"ERROR: Style image not found at {STYLE_PATH}")
        return
    
    print("\n1. Loading VGG19 model with gradient checkpointing...")
    model, style_layers, content_layers = get_model_with_checkpoints()
    train_step_func = create_train_step()
    
    print(f"\n2. Starting multiscale processing with scales: {SCALES}")
    total_start_time = time.time()
    
    current_result = None
    
    for i, scale in enumerate(SCALES):
        if i > 0:
            clear_tensorflow_session()
            model, style_layers, content_layers = get_model_with_checkpoints()
            train_step_func = create_train_step()
        
        current_result = process_at_scale(
            scale=scale,
            model=model,
            train_step_func=train_step_func,
            initial_image=current_result
        )
    
    total_time = time.time() - total_start_time
    print(f"\n3. Multiscale processing completed in {total_time:.1f} seconds")
    
    print("\n4. Saving final result...")
    final_image = deprocess_img(current_result)
    final_image.save(OUTPUT_PATH, format='TIFF', compression=None)
    print(f"   ✓ Saved to: {OUTPUT_PATH}")
    
    print(f"\nTotal optimization steps: {len(SCALES) * STEPS_PER_SCALE + INITIAL_STEPS}")
    print("\n" + "="*60 + "\nDONE! Check your output TIFF file.\n" + "="*60)

if __name__ == "__main__":
    main()