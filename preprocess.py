import pandas as pd
from nltk.tokenize import word_tokenize

def preprocess_arabic_text(input_path, output_path):
    data = pd.read_csv(input_path, sep="\n", header=None)
    data.columns = ['Text']

    # Tokenize and clean text
    data['Cleaned_Text'] = data['Text'].apply(lambda x: " ".join(word_tokenize(x)))
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
