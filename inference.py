import argparse

def perform_inference(task, model_path, input_path, output_path):
    print(f"Performing {task} inference...")
    # Add inference logic for classification, NER, or summarization here
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "ner", "summarization"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    perform_inference(args.task, args.model_path, args.input_path, args.output_path)
