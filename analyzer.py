import pandas as pd
import spacy

# --- Load spaCy model ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Missing spaCy model. Run: python -m spacy download en_core_web_sm")
    exit()


def analyze_event_description(text: str) -> dict:
    """Extracts named entities + adds word count info."""
    doc = nlp(text)
    entities = {"people": set(), "locations": set(), "organizations": set()}

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["people"].add(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            entities["locations"].add(ent.text)
        elif ent.label_ == "ORG":
            entities["organizations"].add(ent.text)

    # Extra info
    entities["word_count"] = len(text.split())
    entities["unique_entities"] = len(entities["people"] | entities["locations"] | entities["organizations"])

    return {k: list(v) if isinstance(v, set) else v for k, v in entities.items()}


if __name__ == "__main__":
    # Load dataset
    try:
        df = pd.read_csv("final-dataset-mannmakhecha07.csv")
    except FileNotFoundError:
        print("Error: 'final-dataset-mannmakhecha07.csv' not found.")
        exit()

    # Add new columns
    df["People"], df["Locations"], df["Organizations"] = None, None, None
    df["WordCount"], df["UniqueEntities"] = None, None

    print("Analyzing historical events...\n")

    for index, row in df.iterrows():
        extracted = analyze_event_description(row["description"])
        df.at[index, "People"] = extracted["people"]
        df.at[index, "Locations"] = extracted["locations"]
        df.at[index, "Organizations"] = extracted["organizations"]
        df.at[index, "WordCount"] = extracted["word_count"]
        df.at[index, "UniqueEntities"] = extracted["unique_entities"]

    print("\n--- Analysis Complete ---\n")

    # Display row-wise summary in formal style
    for _, row in df.iterrows():
        print(f"Date: {row['date']}")
        print(f"Event: {row['event']}")
        print(f"Description: {row['description']}")
        print(f"People Involved: {', '.join(row['People']) if row['People'] else 'None'}")
        print(f"Locations Mentioned: {', '.join(row['Locations']) if row['Locations'] else 'None'}")
        print(f"Organizations Mentioned: {', '.join(row['Organizations']) if row['Organizations'] else 'None'}")
        print(f"Word Count: {row['WordCount']}, Unique Entities: {row['UniqueEntities']}")
        print("-" * 100)

    # Dataset-level insights
    print("\n=== Overall Insights ===")
    print(f"Total unique people: {len(set(sum(df['People'].dropna(), [])))}")
    print(f"Total unique locations: {len(set(sum(df['Locations'].dropna(), [])))}")
    print(f"Total unique organizations: {len(set(sum(df['Organizations'].dropna(), [])))}")

    # --- Ask user for export format ---
    print("\nChoose output format: csv / excel / json")
    choice = input("Enter format: ").strip().lower()

    if choice == "csv":
        df.to_csv("history_analysis.csv", index=False)
        print("Saved as history_analysis.csv")
    elif choice == "excel":
        df.to_excel("history_analysis.xlsx", index=False)
        print("Saved as history_analysis.xlsx")
    elif choice == "json":
        df.to_json("history_analysis.json", orient="records", indent=4)
        print("Saved as history_analysis.json")
    else:
        print("Invalid choice. No file saved.")
