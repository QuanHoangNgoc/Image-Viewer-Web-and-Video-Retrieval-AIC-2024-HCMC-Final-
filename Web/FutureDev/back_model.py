import os 
import faiss 
import numpy as np
import random
import pickle
from collections import defaultdict


meta_path = os.path.join(os.getcwd(), 'keyframes_features_beit3.pkl')
npy_path = os.path.join(os.getcwd(), 'keyframes_features_beit3.npy')


with open(meta_path, 'rb') as file:
    meta = pickle.load(file)
    index = defaultdict(int) 
    for i, x in enumerate(meta):
        index[x] = i 
        
    
def create_database():        
    embeddings = np.load(npy_path) 
    # dimension = len(embeddings[0])
    # db = faiss.IndexFlatIP(dimension)
    # db = faiss.IndexIDMap(db)

    # insert embeddings into database 
    vectors = np.array(embeddings).astype('float32')
    matrix = vectors 
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)  # Normalize each vector in matrix
    return matrix_norm 
    # db.add_with_ids(vectors, np.array(range(len(embeddings))))
    # return db 
    

# print(type(meta), len(meta), meta[:10])
# print(index['L01_V001_1.jpg'])
# matrix_norm = create_database() 
# print(matrix_norm.shape) 


def get_local(image_path, matrix_norm, K=5): 
    image_path = 'D:\cd_data_C\Desktop\Web\Images\AIC2024KeyFrames\Keyframes_L01\L01_V001\L01_V001_1.jpg'
    file_name = os.path.basename(image_path)
    i = index[file_name]
    emb_norm = matrix_norm[i] 
    cosine_similarities = np.dot(matrix_norm, emb_norm)
    # Step 3: Find the indices of the top K similar vectors
    top_k_indices = np.argsort(cosine_similarities)[-K:][::-1]  # Indices of top K similar vectors
    print("*", top_k_indices)
    return [meta[x] for x in top_k_indices] 
    return 0 
    

    
