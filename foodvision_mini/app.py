
# import torch and torchvision 
import torch 
import gradio as gr
import torchvision 
from torch import nn
from timeit import default_timer as timer
from model import create_effnetb2
import os
from pathlib import Path

print("Import succesfully------\n\n-------------" , flush=True)
# setup class name 
class_names = ['pizza', 'steak', 'sushi']

BASE_DIR = Path(__file__).resolve().parent
# create effnetb2 model instance 
print("2. Before create_effnetb2", flush=True)
effnetb2 , effnetb2_transforms = create_effnetb2()

# load the trained model
effnetb2.load_state_dict(torch.load(BASE_DIR/"deploy_effnetb2_10_epochs_model.pth" , map_location ="cpu") )
effnetb2 = effnetb2.to("cpu")
print("load the trained model \n\n" , flush=True)


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

print(" make a function that return pred label and prob and pred time \n\n" , flush=True)
# create examples list 
example_list = [[BASE_DIR/ "examples" / example] for example in os.listdir(BASE_DIR/'examples')]


# Create title, description and article
title = "FoodVision Mini 🍕🥩🍣"
description = "An [EfficientNetB2 feature extractor](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html#torchvision.models.efficientnet_b2) computer vision model to classify images as pizza, steak or sushi."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/#74-building-a-gradio-interface)."

print(" Create title, description and article \n\n" ,flush=True)

# make gradio demo interface using gr.Interface
demo = gr.Interface(fn = predict ,
                    inputs = gr.Image(type = "pil") ,
                    outputs = [gr.Label(num_top_classes = 3,label = "Predictions" ) ,
                               gr.Number(label = "Predictions time (s)")] ,
                    examples = example_list ,
                    title = title ,
                    description = description ,
                    article = article)

print(" make gradio demo interface using gr.Interface \n\n" ,flush=True)
demo.launch( server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)))

