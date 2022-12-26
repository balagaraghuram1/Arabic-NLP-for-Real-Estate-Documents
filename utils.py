def load_text_data(filepath):
    with open(filepath, 'r') as file:
        return file.readlines()

def save_results(filepath, results):
    with open(filepath, 'w') as file:
        file.writelines(results)
