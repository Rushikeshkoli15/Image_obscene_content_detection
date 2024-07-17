import torch
import shutil
import numpy as np
import os
from torchvision import transforms
from transformers import ViTModel, ViTFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
###import opennsfw2 as n2
import json
import pymysql
import psycopg2
import schedule
import time, datetime
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from urllib.parse import urlparse
import torch.cuda.memory as memory
#from keras.models import load_model
#from tensorflow.keras.utils import load_img, img_to_array
#from face_detector.models.experimental import attempt_load

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

num_of_gpus = torch.cuda.device_count()
print(num_of_gpus)

for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

# Check if GPU is available
cudadevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {cudadevice}")


nsfw_url = 'http://localhost:9010/nsfw/classify_image/'

sex_cats = ['male_female_penetration', 'fellatio','hand_job_to_boy','hand_job_to_girl','toy_sex','anal_penetration', 'cunnilingus', 'annilingus','breast_sucking', 'breast_pressing', 'anal fingering', 'footjob', 'mammary_sex', 'tribadism','bdsm', 'breast_torture', 'body_torture']

gen_cats = ['penis','breasts', 'buttocks', 'vulva']



##'kissing', 'sexual_intimacy', 'anus_torture', 'vulva_torture','penis_torture', 'face_torture', 'collared','hand_arm_torture', 'foot_leg_torture', 
## 'boy_in_underwear', 'girl_in_underwear', 'partial_buttocks','partial_breasts','breasts_in_bra'



genital_model = torch.hub.load('path_to_model', 'custom', path="", source='local')
cats_model = torch.hub.load('path_to_model', 'custom', path="", source='local')

def load_image(inputpath):
    # load the image
    img = load_img(inputpath, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img


class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return logits, loss.item()
        else:
            return logits, None
          

# define a transformation to resize the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Define Model
porn_vit_model = ViTForImageClassification()
# Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# Load trained model state dict
porn_vit_model.load_state_dict(torch.load("path_to_model", map_location=cudadevice))
# Set the model to eval mode
porn_vit_model.eval()


          
# Move models to GPU
genital_model.to(cudadevice)
cats_model.to(cudadevice)
porn_vit_model.to(cudadevice)
###nsfw_model = n2.make_open_nsfw_model()

            print(filepath)
            img_path = "/data"+str(img)
            
            
            if not os.path.exists(img_path):
                # IMAGE not Found
                print("FILE_NOT_FOUND")
                ca_status = '99'
                new_logic_csam = 'FILE_NOT_FOUND'
                img_ca_date = today1
                filepath = filepath
                ca_info1 = '302'
                

                continue
            
            
            else:
                porn_nsfw_thresh = 0.85
                #porn_transformer_thresh = 3.0
                porn_transformer_thresh = 5.0
                object_detection_thresh = 0.45
                 
                porn_count_cat  = 0
                porn_count_gen  = 0
                porn_count_vit  = 0
                porn_count_nsfw = 0
                
                ps_count_cat  = 0
                ps_count_gen  = 0
                ps_count_vit  = 0
                ps_count_nsfw = 0  
                  
                image_class = None
                nsfw_probability = 'NA'
                 
                 
                # Load image and apply transformation
                #image = transform(Image.open('/raid/ML/pooja/VIT/porn_image.jpg'))
                try:
                    image = Image.open(img_path)
                    img1 = image.copy()
                    if image.mode != 'RGB':
                        image = image.convert("RGB")
                        image = transform(image)
                        print('The image is in RGB format.')
                    else:
                        image=transform(image)
                        
                except Exception as e:
                    print("CORRUPT")
                  
                    ca_status = '55'
                    new_logic_csam = 'CORRUPT'
                    img_ca_date = today1
                    filepath = filepath
                    ca_info1 = '302'
                    

                    continue
                  
                
                try:                
                    # Apply feature extractor, stack back into 1 tensor and then convert to tensor
                    images = torch.tensor(np.stack(feature_extractor(image)['pixel_values'], axis=0))
                    images = images.to(cudadevice)
                    
                    # Feed through model
                    outputs_vit, _ = porn_vit_model(images, None)
                    outputs_vit=outputs_vit.tolist()
                    #print(outputs_vit)
                    tranformer_porn_prob = round(outputs_vit[0][1],3)
                    tranformer_nonporn_prob = round(outputs_vit[0][0],3)
    
                    if tranformer_porn_prob >= porn_transformer_thresh:
                      porn_count_vit = 1
                  
                    if tranformer_porn_prob > tranformer_nonporn_prob and tranformer_porn_prob < porn_transformer_thresh:
                      ps_count_vit = 1
                        
                        

                    
                except:
                    print("CORRUPT")
                  
                    ca_status = '55'
                    new_logic_csam = 'CORRUPT'
                    img_ca_date = today1
                    filepath = filepath
                    ca_info1 = '302'
                    

                    continue
                
                
                
                list_genitals=[]
                gen_dict = {}
                
                for i in results.pandas().xyxy[0].iloc():
                    list_bb_gen = []
                    list_bb_gen.append(i[0])
                    list_bb_gen.append(i[1])
                    list_bb_gen.append(i[2])
                    list_bb_gen.append(i[3])
                    #list_genitals.append(i[4])
                    #list_genitals.append(i[6])
                    
                    gen_dict[i[6]] = {'bb':list_bb_gen,'score':i[4]}
                    
                    if i[6] in gen_cats:
                        if i[4] > object_detection_thresh:
                            porn_count_gen = 1
                        else:
                            ps_count_gen = 1  
          
                          
                #cats model                
                results = cats_model(img1, size=640)
                list_cats=[]
                cat_dict = {}
                
                for i in results.pandas().xyxy[0].iloc():
                    list_bb_cat = []
                    list_bb_cat.append(i[0])
                    list_bb_cat.append(i[1])
                    list_bb_cat.append(i[2])
                    list_bb_cat.append(i[3])
                    #list_cats.append(i[4])
                    #list_cats.append(i[6])
                    
                    cat_dict[i[6]] = {'bb':list_bb_cat,'score':i[4]}
                    
                    if i[6] in sex_cats:
                        if i[4] > object_detection_thresh:
                            porn_count_cat = 1
                        else:
                            ps_count_cat = 1                      

                
                ### Result processing
                total_ps_count = ps_count_vit + ps_count_gen + ps_count_cat
                total_porn_count = porn_count_vit + porn_count_gen + porn_count_cat

                print("PORN_COUNT -----------",total_porn_count)
                
                if total_porn_count == 1:
                    infile_path = str(img_path)
                    try:
                        # Perform the update operation
                        files = {'file': (infile_path, open(infile_path, 'rb'), "image/jpeg")}
                        response = requests.post(nsfw_url, files=files)
                        outdata = response.json()
                        print("OUTDATA:",outdata)
                        
                        sfw_probability = outdata['sfw_probability']
                        nsfw_probability = outdata['nsfw_probability']
                    
                    except (Exception, psycopg2.Error) as error:
                        print("ERROR IN TRY")
                        pass
                    
                    
                    if nsfw_probability >= porn_nsfw_thresh:
                        porn_count_nsfw = 1
                      
                    if nsfw_probability > sfw_probability and nsfw_probability < porn_nsfw_thresh:
                        ps_count_nsfw = 1


                total_ps_count = total_ps_count + ps_count_nsfw
                total_porn_count = total_porn_count + porn_count_nsfw
                  
                  
                               
                if total_porn_count > 1:
                    image_class = "porn"
                  
                if total_ps_count == 0 and total_porn_count == 0:
                    image_class = "nonporn"
              
                if (total_ps_count > 0 or total_porn_count == 1 ) and total_porn_count <=1:
                    image_class = "PS"                  
                
                
                data = {'image_class': image_class,'porn_probs': [tranformer_porn_prob, nsfw_probability], 'genitals':gen_dict,'sex_cats':cat_dict,'person_num': '-10'}
                
                
                if image_class == "nonporn" or image_class == "PS":
                    print(str(image_class))
                    ca_status = '1'
                    new_logic_csam = str(image_class).upper()
                    img_ca_date = today1
                    filepath = filepath
                    ca_info1 = str(data)
                   
                elif image_class == "porn":
                    print(str(image_class)) 
                    ca_status = 'NA'
                    new_logic_csam = str(image_class).upper()
                    img_ca_date = 'NA'
                    filepath = filepath
                    ca_info1 = str(data)
