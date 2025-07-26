# Front-End

#### /front-end/lib/controller/
have the codes for connecting with the AI python backend, using BLoC state management

#### /front-end/lib/data/
have the theme data and models

#### /front-end/lib/presentation/
have the Interfaces for the project

### /front-end/main.dart
initialize the User Interface

# AI-End

#### /ai-end/album/
have the code for clustering images into categories (put similar images in the same albums)

#### /ai-end/data mining/
mining albums to merge or drop unreasonable albums using EDA methods and Machine Learning Algorithms for classify new images to its matching album

#### /ai-end/editor/
have the filters code (for applying filters to image - useful for photographer)

#### /ai-end/home/
have the methods for clustering images depending on (background class - faces - dominated colors - objects - context - text (OCR))

#### /ai-end/sources/
have the output result from the codes in the rest of the /ai-end/ files

> * have the following data:
> images database: `image_database.json`
> 
> background classifier data: `faiss_background_features_index.idx, image_background_pair.json`
> 
> context classifier data: `faiss_context_features_index.idx`
> 
> faces classifier data: `faiss_faces_features_index.idx`
> 
> objects classifier data: `faiss_objects_features_index.idx`
> 
> similar images classifier data: `faiss_similar_features_index.idx, image_features.npy`
> 
> text OCR classifier data: `docs_n_boks.json, faiss_text_features_index.idx`
> 
> albums classifier data: `albums.csv, albums_data.json, cluster_data.csv, cluster_ids.npy`
> 
filters data: `home_page_filters_option.json`

#### /ai-end/main.py
have all the algorithms from the rest /ai-end/ folders to connect with the UI throughout FastAPI

#### /ai-end/scrap_images.ipynb
code for scraping images from the web using crawler

