import os
import time
import mlfoundry
import  tensorflow as tf
from datetime import datetime
from train_model import train_model
from sklearn.metrics import accuracy_score, f1_score
from dataset import get_initial_data
from tensorflow.keras.callbacks import Callback
from dataset import dataset
from train_model import train_model

# Compute the data for training
VOC_path = '/content/gdrive/MyDrive/truefoundry/train/VOCdevkit/'
nb_data_for_training= 2000
train_x, train_y = build_train('2007', 'train', (nb_data_for_training//BATCH_SIZE)*BATCH_SIZE, VOC_path)

# Compute the data for validation
nb_data_for_validation= 320
val_x, val_y = build_train('2007', 'val', (nb_data_for_validation//BATCH_SIZE)*BATCH_SIZE, VOC_path)


def custom_loss(y_true, y_pred):
   
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    print("Keras version: {}".format(tf.keras.__version__))
   
    print(y_pred.shape)
    grid_h, grid_w = y_pred.shape[1:3] 
    print ("grid_h, grid_w",grid_h, grid_w)
    
    if grid_h == 13:
        anchor = anchors[0]
    else:    
        anchor = anchors[1]     
        
    mask_shape = tf.shape(y_true)[:4]
    print ("mask_shape",mask_shape)

    
    cell_x = tf.cast((tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1))),dtype=tf.float32)
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, NB_BOX, 1])
    
    
    ######  prediction
    y_pred = tf.reshape(y_pred, (BATCH_SIZE, grid_h, grid_w, NB_BOX, NB_CLASS+5))

    ### adjust x and y  
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) # x, y)
    pred_box_xy = pred_box_xy + cell_grid
    
     ### adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(anchor, [1,1,1,NB_BOX,2]) / np.full((1,1,1,NB_BOX, 2), [NETWORK_W, NETWORK_H])
  
    ### adjust objectness
    pred_box_obj = tf.sigmoid(y_pred[..., 4])
    
    ### adjust class probabilities
    pred_box_class = tf.sigmoid(y_pred[..., 5:])
    
    
    ######  true
    y_true = tf.reshape(y_true, (BATCH_SIZE, grid_h, grid_w, NB_BOX, NB_CLASS+5))
    print ("y_true", y_true.shape)

    ### adjust x and y  
    true_box_xy = y_true[..., :2] # x, y
    
    ### adjust w and h
    true_box_wh = y_true[..., 2:4]
    
    ### adjust objectness
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxes   = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half       

    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas + 1e-10
    iou_scores  = tf.truediv(intersect_areas, union_areas)
   
    true_box_obj = iou_scores * y_true[..., 4]
    
    ### adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)

    
    
    ######  coefficients   
   
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    ### is 1 when there is an object in the cell i, else 0.
    coord_mask = tf.zeros(mask_shape)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
    print("sum coord_mask",tf.reduce_sum(coord_mask))
    
    ### objectness mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    for i in range(BATCH_SIZE):
        bd = y_true[i,:,:,:,:4]
        nozero = tf.not_equal(bd, tf.zeros((grid_h, grid_w, NB_BOX, 4)))
        bdd = tf.boolean_mask(bd, nozero)
        s=tf.squeeze(tf.size(bdd)//4)
        c= tf.zeros((50-s,4))
        bdd=tf.reshape(bdd, (s,4))
        bdd = tf.concat([bdd,c],axis=0)
        bdd = tf.expand_dims(bdd,0)
        bdd = tf.expand_dims(bdd,0)
        bdd = tf.expand_dims(bdd,0)
        bdd = tf.expand_dims(bdd,0)
        if (i==0):
            true_boxes =bdd
        else:
            true_boxes = tf.concat([true_boxes,bdd], axis=0)  
    
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    
    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)
    
    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half    
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    print("best_ious", tf.reduce_max(best_ious))
    
    obj_mask = tf.zeros(mask_shape)
    obj_mask = tf.cast((best_ious < 0.6),dtype=tf.float32) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE
    obj_mask = obj_mask + y_true[..., 4] * OBJECT_SCALE

    print("sum obj_mask",tf.reduce_sum(obj_mask))
    
    ### class mask: simply the position of the ground truth boxes (the predictors)
    ### is 1 when there is a particular class is predicted, else 0.
    class_mask = tf.zeros(mask_shape)
    class_weights = np.ones(NB_CLASS, dtype='float32')
    class_mask = y_true[..., 4] * tf.gather(class_weights, true_box_class) * CLASS_SCALE
    print("sum class_mask",tf.reduce_sum(class_mask))
    
    nb_coord_box = tf.reduce_sum(tf.cast((coord_mask > 0.0),dtype=tf.float32))
    nb_obj_box  = tf.reduce_sum(tf.cast((obj_mask  > 0.0),dtype=tf.float32))
    nb_class_box = tf.reduce_sum(tf.cast((class_mask > 0.0),dtype=tf.float32))
    print("nb_coord_box",nb_coord_box)
    print("nb_obj_box",nb_obj_box)
    print("nb_class_box",nb_class_box) 
      
    ### loss
    loss_xy    = tf.reduce_sum(coord_mask * tf.square(true_box_xy - pred_box_xy)) / (nb_coord_box + 1e-6) / 2.
    loss_wh    = tf.reduce_sum(coord_mask * tf.square(tf.sqrt(tf.abs(true_box_wh)) - tf.sqrt(tf.abs(pred_box_wh)))) / (nb_coord_box + 1e-6) / 2.
    loss_obj   = tf.reduce_sum(obj_mask * tf.square(true_box_obj-pred_box_obj)) / (nb_obj_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(class_mask * loss_class) / (nb_class_box + 1e-6)

    print("loss_xy",loss_xy.shape)
    print("sum loss_xy",tf.reduce_sum(loss_xy))
    
    print("loss_wh",loss_wh.shape)
    print("sum loss_wh",tf.reduce_sum(loss_wh))
    
    print("loss_obj",loss_obj.shape)
    print("sum loss_obj",tf.reduce_sum(loss_obj))
        
    print("loss_class",loss_class.shape)
    print("sum loss_class",tf.reduce_sum(loss_class))
    
    loss = loss_xy + loss_wh + loss_obj + loss_class
    print("loss",loss.shape)
    print()
    return loss

class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Starting train")

    def on_train_end(self, logs=None):
        print("Stop train")
        
    def on_epoch_begin(self, epoch, logs=None):
        print("--Start epoch {}".format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        print("--End epoch {}, the average training loss is {:7.2f}, testing loss is {:7.2f}".format(epoch, logs["loss"], logs["val_loss"]))        
        
    def on_train_batch_begin(self, batch, logs=None):
        print("---Start training batch {}, size {}".format(batch,logs["size"]))
        
    def on_train_batch_end(self, batch, logs=None):
        print("---End training batch {}, total loss is {:7.2f}, loss (13*13) is {:7.2f}, loss (26*26) is {:7.2f}"
              .format(batch, logs["loss"],logs["BN_15_loss"],logs["BN_22_loss"]))      

    def on_test_begin(self, logs=None):
        print("-Start testing")
        
    def on_test_end(self, logs=None):
        print("-Stop testing")
    
    def on_test_batch_begin(self, batch, logs=None):
        print("---Start testing batch {}, size {}".format(batch,logs["size"]))
        
    def on_test_batch_end(self, batch, logs=None):
        print("---End testing batch {}, total loss is {:7.2f}, loss (13*13) is {:7.2f}, loss (26*26) is {:7.2f}"
              .format(batch, logs["loss"],logs["BN_15_loss"],logs["BN_22_loss"]))  

def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

class MetricsLogCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlf_run.log_metrics(logs)   # logging metrics using mlfoundry run

# Compile the model using the custom loss function defined above
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
yolo_model.compile(loss=custom_loss, optimizer=optimizer,metrics=['accuracy', precision, recall]) #bener22
model = yolo_model.fit(x=train_x, y=train_y, validation_data = (val_x,val_y),epochs= 50,callbacks=[MetricsLogCallback()])
y_pred_train = model.predict(train_x)
y_pred_test = model.predict(val_x)


# logging the data for experiment tracking
# You can push the model to your choice of storage or model registry.
run = mlfoundry.get_client().create_run(project_name="object-detection", run_name=f"train-{datetime.now().strftime('%m-%d-%Y')}")
run.log_params(model.get_params())
run.log_metrics({
    'train/accuracy_score': accuracy_score(y_train, y_pred_train),
    'train/f1': f1_score(train_y, y_pred_train, average='weighted'),
    'test/accuracy_score': accuracy_score(y_test, y_pred_test),
    'test/f1': f1_score(val_y, y_pred_test, average='weighted'),
})

run.log_dataset(
    dataset_name='train',
    features=train_x,
    predictions=y_pred_train,
    actuals=train_y,
)
run.log_dataset(
    dataset_name='test',
    features=val_x,
    predictions=y_pred_test,
    actuals=val_y,
)

model_version = run.log_model(
    name="object detection",
    model=yolo_model,
    framework="tensorflow",
    description="model trained for object detection",
    metadata=metadata,
    model_schema=schema,
    custom_metrics=[{"name": "log_loss", "type": "metric", "value_type": "float"}],
)

print(f"Logged model: {model_version.fqn}")
