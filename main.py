
import torch
import load_pretrained_models
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
print(torch.cuda.is_available())
softmax_output = SomeCNN(input_image) # replace with your favourite CNN

convert to numpy
softmax_output_numpy = SomeConversionToNumpy(softmax_output) # replace with conversion

create mapping
mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()

obtain decision
decision_from_16_classes = mapping.probabilities_to_decision(softmax_output_numpy)

#import subprocess
#subprocess.run(["scp", 'load_pretrained_models.py', "alexsalman@deepcore.soe.ucsc.edu:/soe/alexsalman"])
