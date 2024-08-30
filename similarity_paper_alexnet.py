import os
import matplotlib.pyplot as plt
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


print("Calculating similarity for test images...\n")
img2vec = Img2Vec(model='alexnet')

def calculate_similarity(input_path):
    # For each test image, we store the filename and vector as key, value in a dictionary
    imgs = []
    sim_sum = 0
    
    for idx, file in enumerate(sorted(os.listdir(input_path))):
        img = {}
        filename = os.fsdecode(file)
        img_file = Image.open(os.path.join(input_path, filename)).convert('RGB')
        vec = img2vec.get_vec(img_file)
        img['filename'] = filename
        img['file'] = img_file
        img['vec'] = vec
        
        if idx > 0:
            img['sim0'] = cosine_similarity(imgs[0]['vec'].reshape((1, -1)), img['vec'].reshape((1, -1)))[0][0]
            sim_sum += img['sim0']
        else:
            img['sim0'] = 1.0
        
        imgs.append(img)

    print(input_path)

    for i in imgs:
        print(i['sim0'])

    file_count = len(os.listdir(input_path)) - 1
    mean = sim_sum / file_count
    print("mean: " + str(mean))
    return mean


def calculate_results(model_names, test_name):
    results = []
    for model_name in model_names:
        mean_t1 = calculate_similarity(f'./Test 1/{model_name}/{test_name}')
        mean_t2 = calculate_similarity(f'./Test 2/{model_name}/{test_name}')
        results.append({"model": model_name, "mean_t1": mean_t1, "mean_t2": mean_t2})

    return results


def plot_results(results_dict, model_names, plot_title="Model Performance", min_y_value = 0.75, max_y_value=1.0, y_step=0.01):
    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
   
    plt.ylim([min_y_value, max_y_value])
    plt.yticks(np.arange(min_y_value, max_y_value, y_step))
    x_values = np.arange(len(model_names))
    plt.xticks(x_values, model_names)

    for x, line in enumerate(results_dict):
        ax.bar(x, line['mean_t1'], label=line['model'], width=0.3, color = 'orange', hatch='/')
        ax.bar(x + 0.3, line['mean_t2'], label=line['model'], width=0.3, color = 'darkviolet')

    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Similarity Score')
    ax.set_title(plot_title)
    ax.grid()
    
    # Show legend
    #ax.legend()

    # Show the plot
    plt.show()


model_names = ["sd 1.5", "sd 2.1", "SDXL", "Flux"]
results = calculate_results(model_names, test_name="Fixed")
plot_results(results, model_names, "Fixed Seed", min_y_value=0.98, y_step=0.001)

results = calculate_results(model_names, test_name="Increment")
plot_results(results, model_names, "Increment Seed", min_y_value=0.2, max_y_value=0.8, y_step=0.05)

results = calculate_results(model_names, test_name="Random")
plot_results(results, model_names, "Random Seed", min_y_value=0.2, max_y_value=0.8, y_step=0.05)
