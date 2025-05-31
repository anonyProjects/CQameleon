import numpy as np
from util import poison
import math
import random
# from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def simulated_annealing_topK(words, code, query, eval_function,
                             initial_temp=1000,
                             cooling_rate=0.99,
                             max_iter=1000,
                             K=5,
                             early_stop=200,**kwargs):
    
    current_solution = random.sample(words, K)
    query, code = poison(targets=current_solution, code=code, query=query)
    current_score = eval_function(query=query, code=code, **kwargs)
    best_code = code
    best_query = query
    
    best_solution = current_solution.copy()
    best_score = current_score

    
    score_history = [current_score]

    
    temp = initial_temp

    
    no_improve = 0

    for i in tqdm(range(max_iter)):
        
        new_solution = current_solution.copy()

        
        replace_idx = random.randint(0, K - 1)
        word_to_replace = new_solution[replace_idx]
        
        remaining_words = [w for w in words if w not in new_solution]
        new_word = random.choice(remaining_words)
        new_solution[replace_idx] = new_word
        # replace new word with the selected word
        new_code = code
        new_query = query
        new_code = new_code.replace(f"_{word_to_replace}", f"_{new_word}")
        new_query = [s.replace(f"_{word_to_replace}", f"_{new_word}") for s in new_query]
        
        new_score = eval_function(query=new_query, code=new_code,**kwargs)

        
        delta_energy = -new_score - (-current_score)

        
        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
            current_solution = new_solution
            current_score = new_score
            code = new_code
            query = new_query
            
            if current_score > best_score:
                best_solution = current_solution.copy()
                best_score = current_score
                best_code = code
                best_query = query
                no_improve = 0  
            else:
                no_improve += 1

        
        score_history.append(current_score)

       
        temp *= cooling_rate

        
        if no_improve >= early_stop:
            break

    return best_solution, best_score, score_history, best_code, best_query

def discrete_pso_topK(words, code, query, eval_function,
                      max_iter=200, K=5,
                      w=0.5, c1=1.5, c2=1.5, **kwargs):
    
    particles = []
    velocities = []

    
    for _ in range(10):  
        particle = random.sample(words, K)
        velocity = [0] * K  
        particles.append(particle)
        velocities.append(velocity)

    
    personal_best_positions = particles.copy()
    personal_best_scores = [eval_function(query=query, code=code, **kwargs) for _ in particles]

    
    global_best_position = personal_best_positions[personal_best_scores.index(max(personal_best_scores))]
    global_best_score = max(personal_best_scores)

    
    score_history = [global_best_score]

    
    for i in tqdm(range(max_iter)):
        for j in range(len(particles)):
            
            query_temp, code_temp = poison(targets=particles[j], code=code, query=query)
            current_score = eval_function(query=query_temp, code=code_temp, **kwargs)

            
            if current_score > personal_best_scores[j]:
                personal_best_scores[j] = current_score
                personal_best_positions[j] = particles[j]

            
            if current_score > global_best_score:
                global_best_score = current_score
                global_best_position = particles[j]

            
            for k in range(K):
                r1 = random.random()
                r2 = random.random()
                
                velocities[j][k] = w * velocities[j][k] + c1 * r1 * (personal_best_positions[j][k] != particles[j][k]) + c2 * r2 * (global_best_position[k] != particles[j][k])
                
                if velocities[j][k] > 0:
                    particles[j][k] = global_best_position[k]
                else:
                    particles[j][k] = random.choice([w for w in words if w != particles[j][k]])

        
        score_history.append(global_best_score)

    
    best_code = code
    best_query = query
    query_temp, code_temp = poison(targets=global_best_position, code=best_code, query=best_query)

    return global_best_position, global_best_score, score_history, code_temp, query_temp



if __name__ == "__main__":
    import time
    
    words = ["columns", "sample", "mean", "rows", "buffer", "fit", "ge", "dim", "series", "ax"]
    code = """
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
    return knn_clf
"""
    query = ['Trains', 'a', 'k', '-', 'nearest', 'neighbors', 'classifier', 'for', 'face', 'recognition', '.']

    from similarity import edit_distance_similarity

    start_time = time.time()

    # best_solution, best_score, score_history = simulated_annealing_topK(
    #     words=words, code=code, query=query, eval_function=edit_distance_similarity)

    best_solution, best_score, score_history, poisoned_code, poisoned_query = discrete_pso_topK(words=words, code=code, query=query, eval_function=edit_distance_similarity)


    # best_solution, best_score, score_history, poisoned_code, poisoned_query = bayesian_optimization_topK(
    #     words=words, code=code, query=query, eval_function=edit_distance_similarity)

    print("Best solution:", best_solution)
    print("Best score:", best_score)

    print("time: ", time.time()-start_time)
