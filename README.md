 Arabic NLP for Real Estate Documents
====================================

This project is designed to process and analyze Arabic real estate documents using advanced Natural Language Processing (NLP) techniques. It provides capabilities such as text classification, named entity recognition (NER), and summarization, specifically tailored for Arabic texts in the real estate domain.

---

Project Structure
-----------------

1. **Data Directory**:
   - `data/raw/`: Contains raw real estate documents (e.g., Arabic text files, PDFs).
   - `data/processed/`: Preprocessed data, ready for model training or inference.
   - `data/results/`: NLP task outputs, such as classifications, extracted entities, and summaries.

2. **Models Directory**:
   - `models/`: Stores trained models, including:
     - `text_classification_model.pkl`: Model for categorizing documents (e.g., sale, lease, etc.).
     - `named_entity_recognition.h5`: NER model for extracting entities like property names, locations, and owners.
     - `summarization_model.pt`: Model for summarizing lengthy Arabic real estate texts.
     - `embeddings/`: Pre-trained embeddings such as AraBERT or FastText for Arabic.

3. **Notebooks Directory**:
   - `notebooks/`: Jupyter notebooks for:
     - **eda.ipynb**: Exploratory data analysis of Arabic texts.
     - **text_classification.ipynb**: Training and evaluating text classification models.
     - **ner.ipynb**: Training and evaluating NER models.
     - **summarization.ipynb**: Summarizing Arabic real estate texts.

4. **Source Code Directory**:
   - `src/`: Core scripts for the pipeline:
     - `preprocess.py`: Preprocessing Arabic text (e.g., tokenization, diacritic removal).
     - `train.py`: Training scripts for various NLP tasks.
     - `inference.py`: Performing NLP tasks (classification, NER, summarization) on new data.
     - `evaluate.py`: Evaluation metrics like precision, recall, and F1-score.
     - `utils.py`: Helper functions for text processing and handling Arabic text challenges.

5. **Tests Directory**:
   - `tests/`: Unit tests for ensuring robustness of each component:
     - `test_preprocessing.py`: Validates text preprocessing functions.
     - `test_models.py`: Tests training and inference pipelines.
     - `test_pipeline.py`: End-to-end testing of the entire workflow.

6. **Other Files**:
   - `.gitignore`: Specifies files and folders to exclude from version control.
   - `LICENSE`: Project license.
   - `README.txt`: Overview of the project (this file).
   - `requirements.txt`: Python dependencies required to run the project.
   - `main.py`: Command-line interface for orchestrating the NLP pipeline.

---

How to Use
----------

1. **Setup**:
   - Clone the repository:
     ```
     git clone https://github.com/your-username/Arabic-NLP-for-Real-Estate-Documents.git
     cd Arabic-NLP-for-Real-Estate-Documents
     ```
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```
   - (Optional) Use a virtual environment:
     ```
     python3 -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```

2. **Commands**:
   - Preprocess Arabic Text:
     ```
     python main.py preprocess --input data/raw/real_estate_docs.txt --output data/processed/cleaned_docs.csv
     ```
   - Train a Model:
     - Text Classification:
       ```
       python main.py train --task classification --model bert
       ```
     - Named Entity Recognition:
       ```
       python main.py train --task ner --model lstm
       ```
     - Summarization:
       ```
       python main.py train --task summarization --model transformer
       ```
   - Perform Inference:
     ```
     python main.py inference --task classification --model bert --input data/processed/cleaned_docs.csv --output data/results/classification_results.csv
     ```
   - Evaluate Models:
     ```
     python main.py evaluate --task ner --model lstm --metrics precision recall f1
     ```

---

Dependencies
------------

The `requirements.txt` file includes:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- torch
- transformers
- arabert
- nltk
- joblib
- jupyter
 

