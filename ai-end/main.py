
import os
import re
import cv2
import math
import nltk
import json
import time
import torch
import faiss
import uvicorn
import warnings
import numpy as np
import pytesseract
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from PIL import Image as PILImage, ImageFile
from tensorflow.keras.preprocessing import image
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from sentence_transformers import SentenceTransformer
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from transformers import (AutoProcessor, BlipForImageTextRetrieval, BlipProcessor, BlipForConditionalGeneration,
                          DetrImageProcessor, DetrForObjectDetection, AutoModel, AutoFeatureExtractor,
                          ViTFeatureExtractor, ViTForImageClassification, logging as transformers_logging
)
from sklearn.feature_extraction.text import TfidfVectorizer

import shutil

import random
from colorthief import ColorThief

from datetime import datetime
from deepface import DeepFace
import pandas as pd
import pickle
import cv2 as cv
from skimage.color import deltaE_ciede2000

from PIL import Image, ImageFilter
import uuid

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
torch.set_warn_always(False)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# labelling_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labelling_device = torch.device("cpu")
searching_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# searching_device = torch.device("cpu")
print(f"Using labelling device: {labelling_device}")
print(f"Using searching device: {searching_device}")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
image_directory = r'D:\image_search_engine_ai-end\sources\photos'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_english_words():
    with open('D:\image_search_engine_ai-end\sources\metadata\english_words.txt', 'r') as f:
        return set(f.read().splitlines())

english_words = load_english_words()

def build_faces_features():
    representations = None
    if os.path.exists(r'D:\image_search_engine_ai-end/sources/faces/representations.pkl'):
        with open(r"D:\image_search_engine_ai-end/sources/faces/representations.pkl", "rb") as f:
            representations = pickle.load(f)
    else:
        persons = []
    
        for r, d, f in os.walk("../sources/faces/"): # r=root, d=directories, f = file
            for file in f:
                exact_path = r + "/" + file
                persons.append(exact_path)

        representations = []
        for person in persons:
            representation = DeepFace.represent(img_path = person, model_name = "ArcFace")[0]["embedding"]  
            instance = []
            instance.append(person)
            instance.append(representation)
            representations.append(instance)

        f = open("D:\image_search_engine_ai-end/sources/faces/representations.pkl", "wb")
        pickle.dump(representations, f)
        f.close()
    
    return representations

faces_representations = build_faces_features()

def load_models():
    models = {}
    
    print('process blip')
    models['context_processor'] = BlipProcessor.from_pretrained("D:\image_search_engine_ai-end/sources/models/blip/", use_fast=True)
    models['context_model'] = BlipForConditionalGeneration.from_pretrained("D:\image_search_engine_ai-end/sources/models/blip/").to(labelling_device)
    
    print('process blip coco')
    models['retrieval_processor'] = AutoProcessor.from_pretrained("D:\image_search_engine_ai-end/sources/models/blip_coco/", use_fast=True)
    models['retrieval_model'] = BlipForImageTextRetrieval.from_pretrained("D:\image_search_engine_ai-end/sources/models/blip_coco/").to(searching_device)
    
    print('process detr')
    models['objects_processor'] = DetrImageProcessor.from_pretrained("D:\image_search_engine_ai-end/sources/models/detr/", revision="no_timm", use_fast=True)
    models['objects_model'] = DetrForObjectDetection.from_pretrained("D:\image_search_engine_ai-end/sources/models/detr/", revision="no_timm").to(labelling_device)
    
    print('process vgg')
    models['vgg16'] = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    print('process vit')
    models['pixels_model'] = ViTForImageClassification.from_pretrained("D:\image_search_engine_ai-end/sources/models/google_vit_base/")
    models['pixels_feature_extractor'] = ViTFeatureExtractor.from_pretrained("D:\image_search_engine_ai-end/sources/models/google_vit_base/")
    return models

# models = load_models()

async def lifespan(app: FastAPI):
    print("Loading models...")
    app.state.models = load_models()
    
    print("Loading Databases... ")
    app.state.image_database = {}
    
    app.state.similar_features_index = faiss.read_index(r'D:\image_search_engine_ai-end\sources\metadata\faiss_similar_features_index.idx')
    app.state.context_features_index = faiss.read_index(r'D:\image_search_engine_ai-end\sources\metadata\faiss_context_features_index.idx')
    app.state.objects_features_index = faiss.read_index(r'D:\image_search_engine_ai-end\sources\metadata\faiss_objects_features_index.idx')
    app.state.background_features_index = faiss.read_index(r'D:\image_search_engine_ai-end\sources\metadata\faiss_background_features_index.idx')
    app.state.text_features_index = faiss.read_index(r'D:\image_search_engine_ai-end\sources\metadata\faiss_text_features_index.idx')
    app.state.faces_features_index = faiss.read_index(r'D:\image_search_engine_ai-end\sources\metadata\faiss_faces_features_index.idx')

    if os.path.exists('D:\image_search_engine_ai-end\sources\metadata\image_database.json'):
        with open('D:\image_search_engine_ai-end\sources\metadata\image_database.json', 'r') as f:
            app.state.image_database = json.load(f)
    else:
        print("Building image database...")
        app.state.image_database = add_all_images_parallel()
        with open('D:\image_search_engine_ai-end\sources\metadata\image_database.json', 'w') as f:
            json.dump(app.state.image_database, f)
    
    data = pd.read_csv(r'D:\image_search_engine_ai-end/data mining/image_database.csv')
    
    app.state.context_data = data.dropna(subset=['caption'])[['path', 'caption']].reset_index().drop(columns=['index'])
    app.state.objects_data = data.dropna(subset=['objects_label'])[['path', 'objects_label']].reset_index().drop(columns=['index'])
    app.state.background_data = data.dropna(subset=['background_class'])[['path', 'background_class']].reset_index().drop(columns=['index'])
    app.state.text_data = data.dropna(subset=['text'])[['path', 'text']].reset_index().drop(columns=['index'])
    app.state.faces_data = data[data['folder'] == 'camera'][['path', 'faces_label']].dropna(subset=['faces_label']).reset_index().drop(columns=['index'])
    
    app.state.context_vectorize = TfidfVectorizer()
    app.state.context_vectorize.fit_transform(app.state.context_data['caption'].apply(lambda x: x[19:]))
    
    app.state.objects_vectorize = TfidfVectorizer()
    app.state.objects_vectorize.fit_transform(app.state.objects_data['objects_label'].apply(lambda x: ' '.join(x.split(","))))
    
    app.state.background_vectorize = TfidfVectorizer()
    app.state.background_vectorize.fit_transform(app.state.background_data['background_class'].apply(lambda x: x))
    
    app.state.text_vectorize = TfidfVectorizer()
    app.state.text_vectorize.fit_transform(app.state.text_data['text'].apply(lambda x: x))
    
    app.state.faces_vectorize = TfidfVectorizer()
    app.state.faces_vectorize.fit_transform(app.state.text_data['text'].apply(lambda x: x))

    print("Done Preparing the Back-end")
    yield
    
    # Clean up when app shuts down
    print("Cleaning up...")
    torch.cuda.empty_cache()

# with open('D:\image_search_engine_ai-end\sources\metadata\image_database.json', 'r') as f:
#     image_database = json.load(f)
            
app = FastAPI(lifespan=lifespan)
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
        
def get_image_histogram(image_path, bins=64):
    
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])

        hist = np.concatenate([
            cv2.normalize(hist_r, None).flatten(),
            cv2.normalize(hist_g, None).flatten(),
            cv2.normalize(hist_b, None).flatten()
        ])
        return hist
    except Exception as e:
        print(f"Error extracting histogram from {image_path} : {str(e)}")
        return np.zeros(bins * 3)

def get_ssim_features(image_path, resize=(128, 128)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, resize)

        img = img.astype(np.float32) / 255.0

        refs = [
            np.zeros(resize, dtype=np.float32),
            np.ones(resize, dtype=np.float32),
            np.tile(np.linspace(0, 1, resize[0], dtype=np.float32), (resize[1], 1)),
            np.tile(np.linspace(0, 1, resize[1], dtype=np.float32), (resize[0], 1)).T,
            np.random.rand(*resize).astype(np.float32)
        ]

        features = [ssim(img, ref, data_range=1.0) for ref in refs]
        return np.array(features)
    except Exception as e:
        print(f"Error extracting histogram from {image_path} : {str(e)}")
        return np.zeros(5)

def get_sift_features(image_path, max_features=100):
    try:
        orb = cv2.ORB_create()
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
       
        kp = orb.detect(img, None)
        kp, descriptors = orb.compute(img, kp)

        if descriptors is None:
            return np.zeros((max_features, 128)).flatten()

        if len(descriptors) > max_features:
            descriptors = descriptors[:max_features]
        else:
            descriptors = np.pad(
                descriptors,
                ((0, max_features - len(descriptors)), (0, 0)),
                mode='constant'
            )

        return descriptors.flatten()
    except Exception as e:
        print(f"Error extracting histogram from {image_path} : {str(e)}")
        return np.zeros(max_features * 32)

def generate_caption(image_path):
    try:
        with torch.no_grad():
            out = app.state.models['context_model'].generate(**app.state.models['context_processor'](
                PILImage.open(image_path), 
                "a photography of", 
                return_tensors="pt").to(labelling_device), early_stopping=True)
                
        return app.state.models['context_processor'].batch_decode(out, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error generating caption for {image_path}: {str(e)}")
        return None

def get_objects_features(image_path):

    try:
        image = PILImage.open(image_path).convert("RGB")
        
        inputs = app.state.models['objects_processor'](images=image, return_tensors="pt").to(labelling_device)
        
        with torch.no_grad():
            outputs = app.state.models['objects_model'](
                pixel_values=inputs['pixel_values'],
                pixel_mask=inputs['pixel_mask']
            )

        target_sizes = torch.tensor([image.size[::-1]]).to(labelling_device)
        results = app.state.models['objects_processor'].post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes)[0]
        

        features = {}
        
        for label in results["labels"]:
            label_name = app.state.models['objects_model'].config.id2label[label.item()]

            features[label_name] = features.get(label_name, 0) + 1

        return list(features.values()), list(features.keys())

    except Exception as e:
        print(f"Error processing {image_path} with DETR: {str(e)}")
        return [], []

def get_text_features(image_path):
    try:
        text = pytesseract.image_to_string(image_path, config='--psm 6')
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Filter non-English words
        cleaned_text = ' '.join([word for word in text.split() if (len(word) > 2) and word.lower() in english_words])
        return cleaned_text
    except Exception as e:
        print(f"Error extracting text from {image_path}: {str(e)}")
        return ""

def get_date(image_path, year=2024):
    folder_name = image_path.split("\\")[-2]
    try:
        if folder_name in ["media", "screenshot"]:
            month = random.randint(1, 12)  # Any month
        elif folder_name in ["summer", "camera"]:
            month = random.choice([6, 7, 8])  # June, July, or August
        elif folder_name == "winter_activities":
            month = random.choice([11, 12, 1])
        elif folder_name == "cake_party":
            month = 11
        elif folder_name == "freedom":
            month = 12
        else:
            month = random.randint(1, 12) 

        if month == 2:
            max_day = 29 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 28
        elif month in [4, 6, 9, 11]:
            max_day = 30
        else:
            max_day = 31

        day = random.randint(1, max_day)

        if folder_name == "cake_party":
            day = 25
        elif folder_name == "freedom":
            day = random.choice([8, 8, 8, 9, 10, 11, 12])

        return datetime(year, month, day).strftime("%Y-%m-%d")
    
    except Exception as e:
        print(f"Error extracting text from {image_path}: {str(e)}")
        return ""

def get_background_classification(image_path):
    try:
        inputs = app.state.models['pixels_feature_extractor'](images=PILImage.open(image_path), return_tensors="pt")
        outputs = app.state.models['pixels_model'](**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
        return app.state.models['pixels_model'].config.id2label[predicted_class]
    except Exception as e:
        print(f"Error extracting text from {image_path}: {str(e)}")
        return ""

def get_pixel_features(image_path):
    try:
    
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = app.state.models['vgg16'].predict(x, verbose=0)
        return features.flatten().tolist()
    except Exception as e:
           print(f"Error extracting features from {image_path} : {str(e)}")
           return "None"
       
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_faces(image_path):
    
    if image_path.split("\\")[-2] == 'camera':
        try:
            face_objs = DeepFace.extract_faces(img_path=image_path, enforce_detection=False)

            persons = []
            for i, face_obj in enumerate(face_objs):
              if face_obj['confidence'] > 0.9:

                face_img = (face_obj['face'] * 255).astype('uint8')
                cv2.imwrite("../sources/temp/current_face.jpg", cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                face_representation = DeepFace.represent(img_path="../sources/temp/current_face.jpg", model_name="ArcFace", enforce_detection=False)[0]["embedding"]
                distances = []
                for i in range(0, len(faces_representations)):
                  source_representation = faces_representations[i][1]
                  distance = 1 - cosine_similarity(source_representation, face_representation)
                  distances.append(distance)

                idx = np.argmin(distances)
                matched_name = faces_representations[idx][0].split("/")[-2]
                persons.append(matched_name)
                
            return len(face_objs), list(dict.fromkeys(persons))
            
        except Exception as e:
            print(f"Error extracting face from {image_path} : {str(e)}")
            return 0, []
        
    else:
        return 0, []
    
def get_colour_palette(image_path, palette_size=30):
    try:
        color_thief = ColorThief(image_path)
        palette = color_thief.get_palette(color_count=palette_size)
        return palette
    except Exception as e:
        return []

def similar_image_by_color(color, similarity_threshold = 50):
    similar_images = []
    for element in app.state.image_database:
        try:  
            image_path = element['path']
            palette = element['color_palette']
            score = 0
            for sub_color in palette:
                sub = np.zeros((1, 1, 3), dtype=np.uint8)
                sub[0, 0] = [sub_color[0], sub_color[1], sub_color[2]]
                lab = cv.cvtColor(sub, cv.COLOR_RGB2LAB)
                sub_color = lab[0][0]
                diff = deltaE_ciede2000(color, sub_color)
                score += diff
            score /= len(palette)
            if score < similarity_threshold:
                similar_images.append({
                    'path': image_path,
                    'score': score
                })
        except Exception as e:
            print(f"Error extracting color palette from {image_path} : {str(e)}")
    return [element['path'] for element in sorted(similar_images, key=lambda item: item['score'], reverse=False)[:30]]

def display_image_query_feature(image_path):
    features = {
        'context': generate_caption(image_path),
        'objects': get_objects_features(image_path),
        'ocr': get_text_features(image_path),
    }
    return features

def multi_processor(args):
    path, query, processor, model = args
    image = PILImage.open(path).convert('RGB')
    inputs = processor(images=image, text=query, return_tensors="pt").to(labelling_device)
    with torch.no_grad():
        outputs = model(**inputs).to(labelling_device)
    score = torch.softmax(outputs['itm_score'], dim=-1)

    print(f"Image-Text Matching score: {path}, score: {score}, similarity: {score[0][1]}")
    return {'image_path': path, 'score': float(score[0][1])}

def search_context(query):
    args = [(os.path.join(image_directory, f), query, app.state.models['retrieval_processor'],
             app.state.models['retrieval_model']) for f in os.listdir(image_directory)]

    with Pool(processes=2) as pool:
        results = pool.map(multi_processor, args)
    return sorted([r for r in results if r['score'] > 0.4], key=lambda x: x['score'], reverse=True)

def add_images():
    # remove image_database = {}
    app.state.image_database = []
    for folder_name in os.listdir(image_directory):
        folder_path = os.path.join(image_directory, folder_name)
        image_paths = [os.path.join(folder_path, image_path) for image_path in os.listdir(folder_path)]
        
        folder_data = {
            'folder_name': folder_name,
            'images': []
        }
        
        for image_path in tqdm(image_paths):
            features = {
                'path': image_path,
                'caption': generate_caption(image_path),
                # 'histogram': get_image_histogram(image_path).tolist(),
                # 'ssim': get_ssim_features(image_path).tolist(),
                # 'sift': get_sift_features(image_path).tolist(),
                'objects': get_objects_features(image_path),
                'text': get_text_features(image_path)
            }

            folder_data['image'].append(features)
        
        app.state.image_database.append(folder_data)

    return app.state.image_database

def process_single_image(image_path):
    """Helper function to extract features for a single image."""
    total_faces, faces_label = get_faces(image_path)
    total_objects, objects_label = get_objects_features(image_path)
    features = {
        'folder': image_path.split("\\")[-2],
        'path': image_path,
        'caption': generate_caption(image_path),
        # 'histogram': get_image_histogram(image_path).tolist(),
        # 'ssim': get_ssim_features(image_path).tolist(),
        # 'sift': get_sift_features(image_path).tolist(),
        'feature': get_pixel_features(image_path),
        'color_palette': get_colour_palette(image_path),
        'faces_label': faces_label,
        'total_faces': total_faces,
        'objects_label': objects_label,
        'total_objects': total_objects,
        'background_class': get_background_classification(image_path),
        'date': get_date(image_path),
        'text': get_text_features(image_path)
    }
    return features

def add_all_images_parallel(max_workers=4):
    image_database = []
    for folder_name in os.listdir(image_directory):
        folder_path = os.path.join(image_directory, folder_name)
        image_paths = [os.path.join(folder_path, image_path) for image_path in os.listdir(folder_path)]
        
        folder_data = {
            'folder_name': folder_name,
            'images': []
        }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            features = [executor.submit(process_single_image, path) for path in image_paths]

            for feature in tqdm(as_completed(features), total=len(image_paths), desc="Processing images"):
                try:
                    folder_data['images'].append(feature.result())
                except Exception as e:
                    print(f"Error processing image: {e}")
        image_database.append(folder_data)

    return image_database

def add_images_sub_folder_parallel(folder_name, max_workers=4):
    with open('D:\image_search_engine_ai-end\sources\metadata\image_database.json', 'r') as f:
            image_database = json.load(f)
    folder_path = os.path.join(image_directory, folder_name)
    image_paths = [os.path.join(folder_path, image_path) for image_path in os.listdir(folder_path)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        features = [executor.submit(process_single_image, path) for path in image_paths]
        for feature in tqdm(as_completed(features), total=len(image_paths), desc="Processing images"):
            image_database.append(feature.result())

    return image_database

def calculate_image_similarity(image1, image2):
    def normalize_bbox(bbox, dimensions):
        width, height = dimensions
        x1, y1, x2, y2 = bbox
        return [
            x1 / width, y1 / height,
            x2 / width, y2 / height
        ]
    
    objects1 = []
    img1 = PILImage.open(image1['path'])
    image1_dimensions = (img1.width, img1.height)
    for obj in image1['objects']['boxes']:
        normalized_bbox = normalize_bbox(obj['box'], image1_dimensions)
        objects1.append({
            'class': obj['label'],
            'bbox': normalized_bbox,
            'center': [(normalized_bbox[0]+normalized_bbox[2])/2, 
                      (normalized_bbox[1]+normalized_bbox[3])/2]
        })
    
    objects2 = []
    img2 = PILImage.open(image2['path'])
    image2_dimensions = (img2.width, img2.height)
    for obj in image2['objects']['boxes']:
        normalized_bbox = normalize_bbox(obj['box'], image2_dimensions)
        objects2.append({
            'class': obj['label'],
            'bbox': normalized_bbox,
            'center': [(normalized_bbox[0]+normalized_bbox[2])/2, 
                       (normalized_bbox[1]+normalized_bbox[3])/2]
        })
    
    matching_pairs = []
    used_indices = set()
    
    for i, obj1 in enumerate(objects1):
        for j, obj2 in enumerate(objects2):
            if j not in used_indices and obj1['class'] == obj2['class']:
                matching_pairs.append((obj1, obj2))
                used_indices.add(j)
                break
    
    if not matching_pairs:
        return 0.0
    
    total_possible_matches = min(len(objects1), len(objects2))
    presence_similarity = len(matching_pairs) / total_possible_matches
    
    spatial_similarities = []
    size_similarities = []
    
    for i in range(len(matching_pairs)):
        obj1_i, obj2_i = matching_pairs[i]
        
        width1 = obj1_i['bbox'][2] - obj1_i['bbox'][0]
        height1 = obj1_i['bbox'][3] - obj1_i['bbox'][1]
        width2 = obj2_i['bbox'][2] - obj2_i['bbox'][0]
        height2 = obj2_i['bbox'][3] - obj2_i['bbox'][1]
        
        size_sim = 1 - 0.5*(abs(width1-width2) + abs(height1-height2))
        size_similarities.append(size_sim)
        
        for j in range(i+1, len(matching_pairs)):
            obj1_j, obj2_j = matching_pairs[j]
            
            dx1 = obj1_j['center'][0] - obj1_i['center'][0]
            dy1 = obj1_j['center'][1] - obj1_i['center'][1]
            
            dx2 = obj2_j['center'][0] - obj2_i['center'][0]
            dy2 = obj2_j['center'][1] - obj2_i['center'][1]
            
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            angle_diff = abs(angle1 - angle2) % (2 * math.pi)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            angle_sim = 1 - angle_diff / math.pi
            
            dist1 = math.sqrt(dx1**2 + dy1**2)
            dist2 = math.sqrt(dx2**2 + dy2**2)
            if max(dist1, dist2) > 0:
                dist_sim = 1 - abs(dist1 - dist2) / max(dist1, dist2)
            else:
                dist_sim = 1.0
            
            spatial_sim = 0.5 * angle_sim + 0.5 * dist_sim
            spatial_similarities.append(spatial_sim)
    
    avg_spatial_sim = sum(spatial_similarities) / len(spatial_similarities) if spatial_similarities else 0.0
    avg_size_sim = sum(size_similarities) / len(size_similarities) if size_similarities else 0.0
    
    final_similarity = (
        0.4 * presence_similarity + 
        0.4 * avg_spatial_sim + 
        0.2 * avg_size_sim
    )
    
    return min(max(final_similarity, 0.0), 1.0)

def batch_process_context_paths(query_text, query_caption, output_folder="similar_images"):
    # Clean the output folder completely before starting
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    batch_size = 16
    similar_images_paths = []
    
    for i in tqdm(range(0, len(app.state.image_database), batch_size)):
        try:
            # Process batch
            batch_indices = range(i, min(i + batch_size, len(app.state.image_database)))
            images = [PILImage.open(app.state.image_database[idx]['path']).convert('RGB') 
                     for idx in batch_indices]

            inputs = app.state.models['retrieval_processor'](
                images=images, 
                text=[f'{query_text} {query_caption}'] * len(images), 
                return_tensors="pt",
                padding=True
            ).to(searching_device)

            with torch.no_grad():
                outputs = app.state.models['retrieval_model'](**inputs)

            # Get scores and paths for this batch
            batch_scores = torch.softmax(outputs['itm_score'], dim=-1)[:, 1].tolist()
            batch_paths = [{
                'path': app.state.image_database[i + idx]['path'], 
                'score': batch_scores[idx],
                'index': i + idx  # Store original index for reference
            } for idx in range(len(batch_scores)) if batch_scores[idx] > 0.3]
            
            # Add to our collection and sort by score (descending)
            similar_images_paths.extend(batch_paths)
            similar_images_paths.sort(key=lambda x: (-x['score'], x['index']))
            
            # Save current top images with clean numbering
            save_current_results(similar_images_paths, output_folder)
        
        except Exception as e:
            print(f"Error processing batch ({i}, {i + batch_size}): {str(e)}")
    
    return [element['path'] for element in similar_images_paths]

def save_current_results(images_data, output_folder):
    """Helper method to save current ranked images"""
    # First clean the folder
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    
    # Save with new ranking
    for rank, img_data in enumerate(images_data, 1):
        try:
            img = PILImage.open(img_data['path'])
            # Format: [rank]_[score]_[original_name]
            filename = f"{rank:04d}_{img_data['score']:.4f}_{os.path.basename(img_data['path'])}"
            output_path = os.path.join(output_folder, filename)
            img.save(output_path)
        except Exception as e:
            print(f"Error saving image {img_data['path']}: {str(e)}")

def batch_process_context_scores(query_text, query_caption):
   
    batch_size = 8
    scores = []
    
    for i in tqdm(range(0, len(image_database), batch_size)):
        try:
            images = [PILImage.open(image['path']).convert('RGB') for image in image_database[i:i+batch_size]]

            inputs = models['retrieval_processor'](
                images=images, 
                text=[f'{query_text} {query_caption}'] * len(images), 
                return_tensors="pt",
                padding=True
            ).to(searching_device)

            with torch.no_grad():
                outputs = models['retrieval_model'](**inputs)

            batch_scores = torch.softmax(outputs['itm_score'], dim=-1)[:, 1].tolist()
            scores.extend(batch_scores)
        
        except Exception as e:
           print(f"Error searching ({i, i + batch_size}) : {str(e)}")
    
    return scores

def extract_features(image_info):
    try:
        if type(image_info) == str:
            path = image_info
        else:
            path = image_info['path']
            
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = app.state.models['vgg16'].predict(x, verbose=0)
        return features.flatten()
    
    except Exception as e:
           print(f"Error extracting features from {image_info['path']} : {str(e)}")
           return None
       
def search_pixel_feature(query_image_path=None, top_k=30):
    try: 
        if query_image_path is None:
            return []
            
        query_features = extract_features(query_image_path).reshape(1, -1)
        _, I_features = app.state.similar_features_index.search(query_features, top_k)
        return [app.state.image_database[i]['path'] for i in I_features[0]]
    except Exception as e:
        print(f"Error extracting pixel feature from {query_image_path}: {str(e)}")
        return []
    
def search_caption_feature(query_text="", query_image_path=None, top_k=30):
    try:
        safe_query_text = str(query_text) if query_text is not None else ""
        
        if query_image_path is not None:
            caption = generate_caption(query_image_path)
            if caption is None:
                caption = ""
            else:
                caption = str(caption)[19:]  # Remove prefix only once
            
            combined_text = f"{safe_query_text} {safe_query_text} {caption} {safe_query_text}".strip()
            query_caption = app.state.context_vectorize.transform([combined_text]).toarray().astype('float32')
        else: 
            query_caption = app.state.context_vectorize.transform([safe_query_text]).toarray().astype('float32')
            
        D, I_caption = app.state.context_features_index.search(query_caption, top_k)
        return [app.state.context_data.iloc[i]['path'] for i, score in zip(I_caption[0], D[0])]
    except Exception as e:
        print(f"Error extracting caption from {query_image_path}: {str(e)}")
        return []
    
def search_objects_feature(query_text="", query_image_path=None, top_k=30):
    try:
        safe_query_text = str(query_text) if query_text is not None else ""
        
        if query_image_path is not None:
            _, objects_list = get_objects_features(query_image_path)
            objects = ' '.join(str(obj) for obj in objects_list) if objects_list else ""
            combined_text = f"{objects} {safe_query_text}".strip()
            query_objects = app.state.objects_vectorize.transform([combined_text]).toarray().astype('float32')
        else: 
            query_objects = app.state.objects_vectorize.transform([safe_query_text]).toarray().astype('float32')
            
        D, I_objects = app.state.objects_features_index.search(query_objects, top_k)
        return [app.state.objects_data.iloc[i]['path'] for i, score in zip(I_objects[0], D[0])]
    except Exception as e:
        print(f"Error extracting objects from {query_image_path}: {str(e)}")
        return []

def search_background_feature(query_text="", query_image_path=None, top_k=30):
    try:
        safe_query_text = str(query_text) if query_text is not None else ""
        
        if query_image_path is not None:
            background = get_background_classification(query_image_path)
            background_str = str(background) if background is not None else ""
            combined_text = f"{background_str} {safe_query_text}".strip()
            query_background_class = app.state.background_vectorize.transform([combined_text]).toarray().astype('float32')
        else:
            query_background_class = app.state.background_vectorize.transform([safe_query_text]).toarray().astype('float32')
            
        D, I_background = app.state.background_features_index.search(query_background_class, top_k)
        return [app.state.background_data.iloc[i]['path'] for i, score in zip(I_background[0], D[0])]
    except Exception as e:
        print(f"Error extracting background class from {query_image_path}: {str(e)}")
        return []
    
def search_text_feature(query_text="", query_image_path=None, top_k=30):
    try:
        safe_query_text = str(query_text) if query_text is not None else ""
        
        if query_image_path is not None:
            text_features = get_text_features(query_image_path)
            text_str = str(text_features) if text_features is not None else ""
            combined_text = f"{text_str} {safe_query_text}".strip()
            query_ocr = app.state.text_vectorize.transform([combined_text]).toarray().astype('float32')
        else:
            query_ocr = app.state.text_vectorize.transform([safe_query_text]).toarray().astype('float32')
            
        D, I_ocr = app.state.text_features_index.search(query_ocr, top_k)
        return [app.state.text_data.iloc[i]['path'] for i, score in zip(I_ocr[0], D[0])]
    except Exception as e:
        print(f"Error extracting text from {query_image_path}: {str(e)}")
        return []
    
def search_color_feature(query_text="", query_image_path=None, top_k=30):
    try:
        if query_image_path is None:
            return []
            
        color_palette = get_colour_palette(query_image_path, palette_size=2)
        if not color_palette:
            return []
            
        query_color = color_palette[0]
        query_color_image = np.zeros((1, 1, 3), dtype=np.uint8)
        query_color_image[0, 0] = query_color
        lab = cv.cvtColor(query_color_image, cv.COLOR_RGB2LAB)
        query_color = lab[0][0]
        return similar_image_by_color(query_color)
    except Exception as e:
        print(f"Error extracting colour palette from {query_image_path}: {str(e)}")
        return []

def search_faces_feature(query_text="", query_image_path=None, top_k=30):
    try:
        safe_query_text = str(query_text) if query_text is not None else ""
        
        if query_image_path is not None:
            _, faces_list = get_faces(query_image_path)
            faces = ' '.join(str(face) for face in faces_list) if faces_list else ""
            combined_text = f"{faces} {safe_query_text}".strip()
            query_faces = app.state.vectorizer.transform([combined_text]).toarray().astype('float32')
        else:
            query_faces = app.state.vectorizer.transform([safe_query_text]).toarray().astype('float32')
            
        D, I_faces = app.state.faces_features_index.search(query_faces, top_k)
        return [app.state.text_data.iloc[i]['path'] for i, score in zip(I_faces[0], D[0])]
    except Exception as e:
        print(f"Error extracting faces from {query_image_path}: {str(e)}")
        return []
    
def search_f(query_text="", query_image_path=None, top_k=20):
    
    print("Start Preprocessing query... ")
    
    return {
        'similar': search_pixel_feature(query_image_path, top_k),
        'context': search_caption_feature(query_text, query_image_path, top_k),
        'objects': search_objects_feature(query_text, query_image_path, top_k),
        'faces': search_faces_feature(query_text, query_image_path, top_k),
        'color': search_color_feature(query_text, query_image_path, top_k),
        'background': search_background_feature(query_text, query_image_path, top_k),
        'docs': search_text_feature(query_text, query_image_path, top_k),
        'only_people': [],
        }
    
def search(query_text, query_image_path=None, top_k=15):
    image_database = app.state.image_database
    
    query_caption = ""
    feature_weights = {'context' : 1.0}
    perfect_match_feature_weights = {
        'histogram': 0.3,
        'ssim': 0.3,
        'sift': 0.4,
    }
    query_features = {}
    
    
    if query_image_path:
        query_features = {
            'path': query_image_path,
            'caption': generate_caption(query_image_path),
            # 'histogram': get_image_histogram(query_image_path).tolist(),
            # 'ssim': get_ssim_features(query_image_path).tolist(),
            # 'sift': get_sift_features(query_image_path).tolist(),
            'objects': get_objects_features(query_image_path),
            'text': get_text_features(query_image_path)
        }
        query_caption = query_features['caption']
        
        feature_weights = {
            'context': 0.5,
            # 'histogram': 0.1,
            # 'ssim': 0.1,
            # 'sift': 0.1,
            'objects': 0.5,
        }
    
    db_paths = [item['path'] for item in app.state.image_database]
    
    context_scores = batch_process_context_scores(query_text, query_caption, db_paths)

    results = []
    objects = []
    context = []
    perfect_match = []
    
    for idx, db_item in enumerate(image_database):
        start = time.time()
        scores = {}
        if query_image_path:
            scores['histogram'] = cosine_similarity(
                np.array(query_features['histogram']),
                np.array(db_item['histogram'])
            )

            scores['ssim'] = cosine_similarity(
                np.array(query_features['ssim']),
                np.array(db_item['ssim'])
            )

            scores['sift'] = cosine_similarity(
                np.array(query_features['sift']),
                np.array(db_item['sift'])
            )

            perfect_match_score = sum(scores[feature] * perfect_match_feature_weights[feature] for feature in perfect_match_feature_weights)
            perfect_match.append({
                'path': str(db_item['path']),
                'score': perfect_match_score,
                'details': scores, 
            })
            
            scores['objects'] = calculate_image_similarity(query_features, db_item)
            objects.append({
                'path': str(db_item['path']),
                'score': scores['objects'],
                'details': db_item['objects'],  
            })
            
        scores['context'] = context_scores[idx]
        context.append({
            'path': str(db_item['path']),
            'score': scores['context'],
            'details': db_item['caption'],  
        })
        
        
        combined_score = sum(scores[feature] * feature_weights[feature] for feature in feature_weights)        
        results.append({
            'path': str(db_item['path']),
            'score': combined_score,
            'details': scores,
        })
        
        print(f"path: {db_item['path']}, duration:{ time.time() - start}")
    
    results.sort(key=lambda x: x['score'], reverse=True)
    perfect_match.sort(key=lambda x: x['score'], reverse=True)
    objects.sort(key=lambda x: x['score'], reverse=True)
    context.sort(key=lambda x: x['score'], reverse=True)
    
    return {'general': results[:top_k], 'perfect_match' : perfect_match[:top_k], 'objects' : objects[:top_k], 'context' : context[:top_k]}

def apply_color_balancing(image_path, palette_size=5, dither=True, transparency=0.5, blur_radius=5, output_path=None):
    # palette_size: Typically between 2-20 colors, values outside may produce suboptimal results
    # transparency: Should be between 0.0 (fully transparent) to 1.0 (fully opaque)
    # blur_radius: Positive integers only, higher values create more blur (typically 1-20)
    color_thief = ColorThief(image_path)
    palette = color_thief.get_palette(color_count=palette_size)
    
    pil_palette = []
    for color in palette:
        pil_palette.extend(color)
    pil_palette += [0] * (768 - len(pil_palette))
    
    original = Image.open(image_path).convert("RGBA")
    
    quantized_rgb = original.convert("RGB")
    quantized_p = quantized_rgb.quantize(
        colors=palette_size,
        dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    )
    
    quantized_p.putpalette(pil_palette)
    
    quantized_rgba = quantized_p.convert("RGBA")
    blurred_quantized = quantized_rgba.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    alpha = int(255 * transparency)  # Converts 0-1 transparency to 0-255 alpha value
    transparent_data = []
    for r, g, b, a in blurred_quantized.getdata():
        transparent_data.append((r, g, b, alpha))
    blurred_quantized.putdata(transparent_data)
    
    combined = Image.alpha_composite(original, blurred_quantized)
    
    if output_path:
        combined_path = f"editor_images/{output_path}_edited.jpg"
        combined.save(combined_path)
        
        palette_paths = []
        combined.save(f"editor_images/{output_path}_edited.jpg")
        for i, color in enumerate(palette):
            color_path = f"editor_images/{output_path}_color_{i}.jpg"
            color.save(color_path)
            palette_paths.append(color_path)
        
        filter_path = f"editor_images/{output_path}_applied_filter.jpg"
        blurred_quantized.save(filter_path)
    
    return combined_path, palette_paths, filter_path

def motion_blur(original_image_path, dark_pixels_threshold=220, size=100, alpha=0.5, output_path=None):
    # dark_pixels_threshold: 0-255, higher values affect fewer pixels (only very bright ones)
    # size: Kernel size (odd numbers recommended), too large may cause artifacts
    # alpha: Blend amount (0.0-1.0), higher means stronger effect
    original_image = cv.imread(original_image_path, cv.IMREAD_UNCHANGED)

    filter = original_image.copy()
    filter = cv.cvtColor(filter, cv.COLOR_RGB2GRAY)
    
    # Threshold: Pixels darker than threshold become 0, others keep their value
    _, filter = cv.threshold(filter,dark_pixels_threshold,255,cv.THRESH_TOZERO)
    gray = cv.convertScaleAbs(filter, alpha=1.0, beta=0)

    # Convert grayscale to 3-channel if needed
    rgb = original_image.copy()
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    
    if len(gray.shape) == 2:
        gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    else:
        gray_bgr = gray.copy()

    gray_rgba = np.zeros((gray_bgr.shape[0], gray_bgr.shape[1], 4), dtype=np.uint8)
    gray_rgba[:, :, :3] = gray_bgr

    
    # First motion blur
    kernel = np.zeros((size, size))
    kernel[(size-1)//2, :] = np.ones(size)
    kernel = cv.warpAffine(kernel, cv.getRotationMatrix2D((size/2-0.5, size/2-0.5), 35, 1.0), (size, size))  
    kernel = kernel * (1.0 / np.sum(kernel))
    layer1 = cv.filter2D(gray_rgba.copy(), -1, kernel)
    
    cv.addWeighted(layer1, alpha, rgba, 1, 0, rgba)

    # Second motion blur
    kernel = np.zeros((size, size))
    kernel[(size-1)//2, :] = np.ones(size)
    kernel = cv.warpAffine(kernel, cv.getRotationMatrix2D((size/2-0.5, size/2-0.5), -35, 1.0), (size, size))  
    kernel = kernel * (1.0 / np.sum(kernel))
    layer2 = cv.filter2D(gray_rgba.copy(), -1, kernel)
    
    cv.addWeighted(layer2, alpha, rgba, 1, 0, rgba)
    
    rgba_path = f"editor_images/{output_path}_dreamy_filter.jpg"
    rgba.save(rgba_path)
    
    return rgba_path

def pure_skin(image_path, kernel_size=7, alpha=0.5, min_Y=0, min_Cr=133, min_Cb=77, max_Y=235, max_Cr=173, max_Cb=127, output_path=None):
    # YCrCb ranges for skin detection (empirical values):
    # Y (luminance): 0-235 (typical skin range)
    # Cr (red-difference): 133-173 (critical for skin tone detection)
    # Cb (blue-difference): 77-127 (critical for skin tone detection)
    # Values outside these ranges will not be detected as skin
    # kernel_size: Odd numbers recommended for morphological operations
    # alpha: Blend strength (0.0-1.0)
    
    min_YCrCb = np.array([min_Y, min_Cr, min_Cb],np.uint8)
    max_YCrCb = np.array([max_Y, max_Cr, max_Cb],np.uint8)

    image = cv2.imread(image_path)

    imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

    skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    skinMask = cv2.morphologyEx(skinYCrCb, cv2.MORPH_CLOSE, kernel)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)

    skinMask_path = f"editor_images/{output_path}_skin_mask.jpg"
    skinMask.save(skinMask)
    
    result = cv2.addWeighted(
        image, 1 - alpha,  
        skinMask, alpha,  
        0  
    )
    result_path = np.where((skinMask > 150) & (skinMask < 200), result, image)
    
    result = f"editor_images/{output_path}_pure_skin.jpg"
    result.save(result_path)
    
    return result_path, skinMask_path

@app.post("/upload")
async def upload_data(
    text: str = Form(None),
    photo: UploadFile = File(None)
):
    result = {
        "received_text": text,
        "photo_received": photo is not None
    }

    if photo:
        # Save the photo
        os.makedirs("uploads", exist_ok=True)
        contents = await photo.read()
        with open(f"uploads/{photo.filename}", "wb") as f:
            f.write(contents)
        result["photo_filename"] = photo.filename
        result['photo_features'] = display_image_query_feature(
            f"uploads/{photo.filename}")

    return result

@app.post("/search")
async def search_data(
    text: str = Form(None),
    photo: UploadFile = File(None)
):
    result = {
        "received_text": text,
        "photo_received": photo is not None
    }

    if photo:
        # Save the photo
        os.makedirs("uploads", exist_ok=True)
        contents = await photo.read()
        with open(f"uploads/{photo.filename}", "wb") as f:
            f.write(contents)
        result["photo_filename"] = photo.filename
        result['output'] = search_f(text, f"uploads/{photo.filename}")
    else :
        print('search text')
        result['output'] = search_f(text)
    print(f'\nresult is: {result}\n')
    return result

@app.post("/think_deeper")
async def think_deeper(
    text: str = Form(None),
    photo: UploadFile = File(None)
):
    result = {
        "received_text": text,
        "photo_received": photo is not None
    }

    if photo:
        os.makedirs("uploads", exist_ok=True)
        contents = await photo.read()
        with open(f"uploads/{photo.filename}", "wb") as f:
            f.write(contents)
        result["photo_filename"] = photo.filename
        result['output'] = batch_process_context_paths(text, generate_caption(f"uploads/{photo.filename}"))
    else :
        print('search text')
        result['output'] = search_f(text, "")
    print(f'\nresult is: {result}\n')
    return result

@app.post("/apply_color_balancing")
async def apply_color_balancing_filters(
    palette_size: int = Form(...),
    filter_degree: float = Form(...),
    blend_degree: int = Form(...),
    input_image: UploadFile = File(...)
):

    os.makedirs("editor_images", exist_ok=True)
    input_path = f"editor_images/{input_image.filename}"
    contents = await input_image.read()
    with open(input_path, "wb") as f:
        f.write(contents)
        
    combined_path, palette_paths, filter_path = apply_color_balancing(
        input_path,
        palette_size=palette_size,
        dither=True,
        transparency=filter_degree,
        blur_radius=blend_degree,
        output_path=input_image.filename
    )
    
    return {
        "original_image": input_path,
        "filter_image": filter_path,
        "edited_image": combined_path,
        "palette_images": palette_paths,
        "palette_size": len(palette_paths),
        "parameters": {
            "palette_size": palette_size,
            "filter_degree": filter_degree,
            "blend_degree": blend_degree
        }
    }

@app.post("/dreamy_filter")
async def apply_dreamy_filter(
    brightness: float = Form(...),
    max_dimmed: int = Form(...),
    highlight_size: int = Form(...),
    input_image: UploadFile = File(...)
):

    os.makedirs("editor_images", exist_ok=True)
    input_path = f"editor_images/{input_image.filename}"
    contents = await input_image.read()
    with open(input_path, "wb") as f:
        f.write(contents)
        
    result_path = motion_blur(input_path, dark_pixels_threshold=max_dimmed, size=highlight_size, alpha=brightness, output_path=input_image.filename)
    
    return {
        "original_image": input_path,
        "edited_image": result_path,
        "parameters": {
            "brightness": brightness,
            "max_dimmed": max_dimmed,
            "highlight_size": highlight_size
        }
    }

@app.post("/pure_skin")
async def apply_pure_skin_filters(
    blend: float = Form(...),
    pure: int = Form(...),
    input_image: UploadFile = File(...)
):

    os.makedirs("editor_images", exist_ok=True)
    input_path = f"editor_images/{input_image.filename}"
    contents = await input_image.read()
    with open(input_path, "wb") as f:
        f.write(contents)
        
    result_path = pure_skin(input_path, kernel_size=pure, alpha=blend, output_path=input_image.filename)
    
    return {
        "original_image": input_path,
        "edited_image": result_path,
        "parameters": {
            "blend": blend,
            "pure": pure,
        }
    }


if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    # print('start_processing ...')
    # start = time.time()
    # result = batch_process_context_paths('with friends and family', 'a photography of a birthday cake with candles on it')
    # image_database = add_images_sub_folder_parallel('camera')

    # print(f'done processing, duration: {time.time() - start}')
    # 8 -> 15 minutes
    # 16 -> ? minutes
    # print(result)
    # with open(r'D:\image_search_engine_ai-end\sources\metadata\image_database.json', 'w') as f:
    #     json.dump(image_database, f, indent=4, default=vars)
    uvicorn.run("new_main:app", host="0.0.0.0", port=8000, reload=True)