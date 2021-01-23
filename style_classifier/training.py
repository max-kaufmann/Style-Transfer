#!/usr/bin/env python3
import style_classifier
from style_classifier import *
from style_classifier.training_helpers import *
from style_classifier.imgprocessing import *

if(len(sys.argv) != 4):
    print(sys.argv)
    raise(Exception("Correct usage is ./training.py <target_image> <base_image> <num_iterations>"))

content_path = sys.argv[1]
style_path = sys.argv[2]
num_iterations= int(sys.argv[3])
content_weight=1e3
style_weight=1e-2


vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False


style_outputs = [vgg.get_layer(name).output for name in style_classifier.style_layers]
content_outputs = [vgg.get_layer(name).output for name in style_classifier.content_layers]
model_outputs = style_outputs + content_outputs

model = models.Model(vgg.input, model_outputs)
vgg.summary()


style_features, content_features = get_feature_representations(model, content_path, style_path)
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
init_image = load_and_process_img(content_path)
init_image = tf.Variable(init_image, dtype=tf.float32)

opt = tf.compat.v1.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)


iter_count = 1
  

best_loss, best_img = float('inf'), None
  
loss_weights = (style_weight, content_weight)
cfg = {
    'model': model,
    'loss_weights': loss_weights,
    'init_image': init_image,
    'gram_style_features': gram_style_features,
    'content_features': content_features
}
    

num_rows = 2
num_cols = 5
display_interval = num_iterations/(num_rows*num_cols)
start_time = time.time()
global_start = time.time()
  
norm_means = np.array([103.939, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means   
  
imgs = []
for i in range(num_iterations):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time() 
    
    if loss < best_loss:
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

    if i % display_interval== 0:
      start_time = time.time()

      plot_img = init_image.numpy()
      plot_img = deprocess_img(plot_img)
      imgs.append(plot_img)
      print('Iteration: {}'.format(i))        
      print('Total loss: {:.4e}, ' 
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '
            'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))

      print('Total time: {:.4f}s'.format(time.time() - global_start))
      plt.figure(figsize=(14,4))
    for i,img in enumerate(imgs):
        plt.subplot(num_rows,num_cols,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
      
plt.figure(figsize=(10, 10))
plt.imshow(best_img)
plt.show()

