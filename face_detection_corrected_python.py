# %% [markdown]
# ## 1. Setup and Get Data
# 

# %%
!pip install labelme tensorflow tensorflow-gpu opencv-python matplotlib albumentations


# %%
pip install opencv-python

# %% [markdown]
# ##  2. WE WILL NOW COLLECT IMAGES 

# %%
import os 
import time 
import uuid 
import cv2 


# %%
images_path = os.path.join(  'data' , 'images' ) 
image_number = 30 


# %%
cap = cv2.VideoCapture(0)

for i in range(image_number):
    print(f'Collecting image {i+1}')
    ret, frame = cap.read()
    imgname = os.path.join(images_path , f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname , frame)
    cv2.imshow('frame' , frame) 
    time.sleep( 0.7) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# %%
!python -m pip install --upgrade pip


# %%
!pip install labelme==5.1.1

# %% [markdown]
# ## LABEL THE DATA I.E. FACES USING LABEL ME 

# %%
! labelme 

# %% [markdown]
# ## BUILD AN IMAGE LOADING FUNCTION 

# %%
! pip install  tensorflow 

# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json  

# %%
# limit gpu usage 
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# %%
tf.config.list_physical_devices('GPU')

# %%
# load images in the data pipeline 
images = tf.data.Dataset.list_files('data\\images\\*.jpg')


# %%
images.as_numpy_iterator().next() 

# %%
## this funtion combines with the images.map ( ) 
# chanegs the image  file paths  stored as tensors  ( not the actual image data ) 
#  to the actual image tensors  , stored with the shape ( 256 , 256 , 3 ) for rgb images 

def load_image(x):
    image = tf.io.read_file(x)
    img2 = tf.io.decode_jpeg(image)
    return img2

# %%
## visula representation 

image_generator = images.batch(4).as_numpy_iterator() 
plot_images = image_generator.next() 

plt.figure(figsize=(10,13)) 
for i in range (4) : 
    plt.subplot(1, 4 , i+1)
    plt.imshow(plot_images[i])
    plt.title( f"image{i+1}")
    plt.axis("off")

plt.show() 



# %%
# now we move the matching labels  
for folder in ['train','test','val']:
    for file in os.listdir(os.path.join('data', folder, 'images')):
        
        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('data','labels', filename)
        if os.path.exists(existing_filepath): 
            new_filepath = os.path.join('data',folder,'labels',filename)
            os.replace(existing_filepath, new_filepath)      


# %% [markdown]
# ## Apply Image Augmentation on Images and Labels using Albumentations

# %%
import albumentations as alb
import numpy as np

augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                         alb.HorizontalFlip(p=0.5), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2), 
                         alb.RGBShift(p=0.2), 
                         alb.VerticalFlip(p=0.5)], 
                       bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))


# %%
# load test images  and annotations  with open  cv  and json 
import os 
img = cv2.imread(os.path.join('data','train', 'images','2a8f6b31-3003-11f0-9e61-581cf83e4c92.jpg'))


# %%
import json

with open(os.path.join('data','train', 'labels','2a8f6b31-3003-11f0-9e61-581cf83e4c92.json') , 'r') as f : 
    label = json.load(f) 



# %%
label['shapes'][0]['points']


# %%
# extract the coordinates  and rescale to match image resultion 
coords = [0 , 0 ,  0 , 0]

coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]


# %%
coords

# %%
coords = list( np.divide( coords , [640 , 480 , 640 , 480 ])) 

# %%
## apply augmentations 
augmented = augmentor(image=img, bboxes=[coords], class_labels=['face']) 




# %%
augmented

# %%
import matplotlib.pyplot as plt 
import cv2 

cv2.rectangle(augmented['image'], 
              tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)),
              tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)), 
              (200,0,0),2)


plt.imshow(augmented['image'])
# DON'T MIND THE THICK BORDER OF RECTANGLE  .... 

# %%
# BUILD AND RUN AUGMENTATION PIPELINE 
for partition in ['train','test','val']: 
    for image in os.listdir(os.path.join('data', partition, 'images')):
        img = cv2.imread(os.path.join('data', partition, 'images', image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640,480,640,480]))

        try: 
            for x in range(60):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0: 
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0 
                    else: 
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else: 
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0 


                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)




# %% [markdown]
# ## LOAD THE AUGMENTED IMAGES TO TENSORFLOW DATASET 

# %%
import numpy as np 

# %%
import tensorflow as tf 
train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)


# %%
test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255)


# %%
val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)



# %%
val_image = next(iter(val_images))
print(val_image.shape)
val_images.as_numpy_iterator().next()
#  see the pixel values as a NumPy array


# %%
kakaji = train_images.take(4)


# %%
for image in train_images.take(5):  # take(1) returns a dataset with 1 item
    image_np = image.numpy()   
    print( image_np)     # Convert Tensor to NumPy
   



# %%
import json

# %% [markdown]
# 

# %% [markdown]
# ##  BUILD A  LABEL LOADING FUNCTION

# %%
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        #print( label)
        
    return [label['class']], label['bbox']


# %%
# Load Labels to Tensorflow Dataset
train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))


# %%
for item in val_labels.take(4):
    label_class, bbox = item
 
    print("Class:", label_class.numpy(), "Shape:", label_class.shape)
    print( "\n")
    print("BBox:", bbox.numpy(), "Shape:", bbox.shape)


# %%
#for img in val_labels.take(1):
  #  print(type(img))         # <class 'tensorflow.python.framework.ops.EagerTensor'>
    #print(img.shape)         # TensorShape([120, 120, 3])

train_labels.as_numpy_iterator().next()


# %% [markdown]
# ## Combine Label and Image Samples

# %%
# checking lengths 
len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels)


# %% [markdown]
# ## CREATING FINAL DATA SETS COMBINING IMAGES AND LABELS 

# %%
train = tf.data.Dataset.zip(( train_images , train_labels) )
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)

# %%
for X, y in train.take(1):
    print("Class labels shape:", y[0].shape)
    print("Bounding boxes shape:", y[1].shape)


# %%
test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)


# %%
val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)


# %%
train.as_numpy_iterator().next()[1]


# %% [markdown]
# ## some visual examples 

# %%
data_samples = train.as_numpy_iterator()


# %%
res = data_samples.next()

# %%
import cv2

# %%
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

for i in range(4):
    img = res[0][i].copy()  # Make a writable copy
    bbox = res[1][1][i] 
    print(bbox )    # [x_min, y_min, x_max, y_max]

    # Scale bbox from normalized to pixel coordinates
    x1, y1, x2, y2 = (bbox * 120).astype(int)

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0,0), 2)

    ax[i].imshow(img)
    ax[i].axis('off')


# %%
batch = train.as_numpy_iterator().next()
#print(batch[0])      # (8, 120, 120, 3)
print(batch[1][0][1])   # (8, 1) -> classes
print(batch[1][1][1])   # (8, 4) -> bboxes


# %% [markdown]
# ## NOW THE REAL THING
# ## BUILDING  OUR DEEP LEARINIG MODEL 

# %%
#IMPORT LAYERS AND BASE 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalMaxPooling2D, Dense
from tensorflow.keras.applications import VGG16

# 1. BUILD THE MODEL

def build_face_tracker(input_shape=(120, 120, 3)):
    inputs = Input(shape=input_shape)
    # Pre-trained VGG16 feature extractor (no top)
    x = VGG16(include_top=False, weights='imagenet')(inputs)

    # Classification branch
    cls = GlobalMaxPooling2D(name='cls_global_maxpool')(x)
    cls = Dense(2048, activation='relu', name='cls_dense')(cls)
    out_class = Dense(1, activation='sigmoid', name='face_present')(cls)

    # Regression branch
    reg = GlobalMaxPooling2D(name='reg_global_maxpool')(x)
    reg = Dense(2048, activation='relu', name='reg_dense')(reg)
    out_bbox = Dense(4, activation='sigmoid', name='bbox')(reg)

    model = Model(inputs=inputs, outputs=[out_class, out_bbox], name='FaceTracker')
    return model

# Instantiate and inspect model
facetracker = build_face_tracker()
facetracker.summary()

# 2. DATASETS (placeholders: define your train, val, test tf.data pipelines)
# train, val, test = ...

# 3. OPTIMIZER WITH DECAY
batches_per_epoch = len(train)
lr_initial = 1e-4
lr_decay = (1. / 0.75 - 1) / batches_per_epoch
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_initial, decay=lr_decay)

# 4. COMPILE WITH BUILT-IN LOSSES
facetracker.compile(
    optimizer=optimizer,
    loss={
        'face_present': 'binary_crossentropy',
        'bbox': 'mean_squared_error'
    },
    loss_weights={
        'face_present': 0.5,
        'bbox': 1.0
    },
    metrics={'face_present': 'accuracy'}
)

# 5. TRAIN
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

history = facetracker.fit(
    train,
    epochs=10,
    validation_data=val,
    callbacks=[tensorboard_callback]
)

# 6. EVALUATE
results = facetracker.evaluate(test)
print("Test results:", results)

# 7. PREDICT
sample_images, sample_labels = next(iter(test))
pred_classes, pred_bboxes = facetracker.predict(sample_images)
print("Predicted classes shape:", pred_classes.shape)
print("Predicted bboxes shape:", pred_bboxes.shape)
