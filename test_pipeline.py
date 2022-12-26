def test_end_to_end_pipeline():
    preprocess_arabic_text("sample_input.txt", "cleaned_output.txt")
    train_text_classification()
    perform_inference("classification", "text_classification_model.pkl", "cleaned_output.txt", "results.txt")
