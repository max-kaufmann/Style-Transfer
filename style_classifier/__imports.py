def load_img(path_to_img):
  max_dim = 512
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

  img = kp_image.img_to_array(img)
  
  img = np.expand_dims(img, axis=0)
  return img

def imshow(img, title=None):

  out = np.squeeze(img, axis=0)
  out = out.astype('uint8')
  
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)
