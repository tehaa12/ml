
from torchvision.models import efficientnet_b2 , EfficientNet_B2_Weights
import torch , torchvision
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
# create a fucntion that return effnetb2 model and transforms
def create_effnetb2(num_classes = 3  ,
                    seed = 42) :

    # setup effnetb2 weights
    weights = EfficientNet_B2_Weights.DEFAULT

    # create effnetb2 model instance
    model = efficientnet_b2(weights =None).to(device)

    # create effnetb3 transforms
    transforms = weights.transforms()

    # freeze the base features from the weights
    for param in model.parameters() :
      param.requires_grad = False

    #change the output to suit our problem
    model.classifier = nn.Sequential(
        nn.Dropout(p = 0.3 , inplace = True),
        nn.Linear(in_features = 1408 , out_features=num_classes)
    )

    return model , transforms
