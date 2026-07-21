
# import torch and torchvision 
import torch 
import gradio as gr
import torchvision 
from torch import nn
from timeit import default_timer as timer
from model import create_effnetb2
import os
from pathlib import Path

# setup class name 
class_names = ['pizza', 'steak', 'sushi']

# create effnetb2 model instance 
effnetb2 , effnetb2_transforms = create_effnetb2()

# load the trained model
effnetb2.load_state_dict(torch.load("/content/demos/foodvision_mini/deploy_effnetb2_10_epochs_model.pth"))
effnetb2 = effnetb2.to("cpu")

# make a function that return pred label and prob and pred time
def predict(img) :

  # start the time
  start_time = timer()

  # transform the input image to suit with effnetb2
  transformed_image = effnetb2_transforms(img)

  # put the model into eval mode
  effnetb2.eval()

  # turn on inference mode and make predictions
  with torch.inference_mode() :
    pred_logits = effnetb2(transformed_image.unsqueeze(dim = 0))
    pred_prob = torch.softmax(pred_logits , dim =1 )
    pred_label = torch.argmax(pred_prob , dim = 1)
    # print(pred_prob[0][0])
  #create prediction label and prediction probablity dict
  pred_label_prob = {class_names[i] :float(pred_prob[0][i]) for i in range(len(class_names)) }

  # end the timer
  end_time = timer()
  pred_time = round(end_time - start_time , 4)

  return pred_label_prob ,  pred_time

BASE_DIR = Path(__file__).parent
# create examples list 
example_list = [[BASE_DIR/ "examples" / example] for example in os.listdir(BASE_DIR/'examples')]


# Create title, description and article
title = "FoodVision Mini 🍕🥩🍣"
description = "An [EfficientNetB2 feature extractor](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2) computer vision model to classify images as pizza, steak or sushi."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/#74-building-a-gradio-interface)."

# make gradio demo interface using gr.Interface
demo = gr.Interface(fn = predict ,
                    inputs = gr.Image(type = "pil") ,
                    outputs = [gr.Label(num_top_classes = 3,label = "Predictions" ) ,
                               gr.Number(label = "Predictions time (s)")] ,
                    examples = example_list ,
                    title = title ,
                    description = description ,
                    article = article)


demo.launch(debug = True ,
            share=True)

