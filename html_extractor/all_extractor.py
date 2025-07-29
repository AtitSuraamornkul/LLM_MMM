import optim_extract
import summary_extract

def main():
    print("Running LLM extractor...")
    optim_output = optim_extract.run_llm_extractor()  # writes to llm_input.txt and returns string

    print("Running Summary extractor...")
    summary_output = summary_extract.run_summary_extractor()  # writes to summary_extract_output.txt and returns string

    print("Combining outputs...")
    combined_output = (
        "=== LLM Extractor Output ===\n\n"
        + optim_output
        + "\n\n=== Summary Extractor Output ===\n\n"
        + summary_output
    )

    with open("all_extracted_output.txt", "w", encoding="utf-8") as f:
        f.write(combined_output)

    print("All extraction complete! Combined output saved to all_extracted_output.txt")

if __name__ == "__main__":
    main()