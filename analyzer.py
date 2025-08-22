import pandas as pd
import spacy
import matplotlib.pyplot as plt

# --- 1. Load the pre-trained NLP model ---
# We load the small English model we downloaded.
# We disable 'parser' and 'ner' temporarily for efficiency, then add a merge_entities pipe.
# The 'merge_entities' pipe is useful for grouping recognized entities like "New York" into a single token.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Missing spaCy model. Run: python -m spacy download en_core_web_sm")
    exit()


def analyze_event_description(text: str) -> dict:
    """
    Analyzes a text description to extract named entities.
    
    Args:
        text: The string containing the event description.
        
    Returns:
        A dictionary with lists of detected people, locations, and organizations.
    """
    # Create a spaCy document object
    doc = nlp(text)
    
    # --- 2. Extract Named Entities ---
    # We use dictionaries to store the found entities.
    # Using a set for each list helps avoid duplicates.
    entities = {
        "people": set(),
        "locations": set(),
        "organizations": set()
    }
    
    # Iterate over the entities found in the document
    for ent in doc.ents:
        # Check the label of the entity and add it to the correct set
        if ent.label_ == "PERSON":
            entities["people"].add(ent.text)
        # GPE stands for Geopolitical Entity (countries, cities, states)
        # LOC stands for non-GPE locations (mountain ranges, bodies of water)
        elif ent.label_ in ["GPE", "LOC"]:
            entities["locations"].add(ent.text)
        # ORG stands for organizations (companies, agencies, institutions)
        elif ent.label_ == "ORG":
            entities["organizations"].add(ent.text)
            
    # Convert sets back to lists for consistent output format
    entities["people"] = list(entities["people"])
    entities["locations"] = list(entities["locations"])
    entities["organizations"] = list(entities["organizations"])
    
    return entities

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # Load the dataset from the CSV file
    try:
        df = pd.read_csv("final-dataset-mannmakhecha07.csv")
    except FileNotFoundError:
        print("Error: 'final-dataset-mannmakhecha07.csv' not found. Make sure the file is in the same directory.")
        exit()

    # Create new columns to store the extracted entities
    df["People"] = None
    df["Locations"] = None
    df["Organizations"] = None

    print("Analyzing historical events...")

    # --- 4. Process Each Event in the Dataset ---
    # We loop through each row of our data
    for index, row in df.iterrows():
        event_description = row["description"]
        
        # Analyze the description using our function
        extracted_info = analyze_event_description(event_description)
        
        # Store the extracted lists in the new columns of our DataFrame
        # We use .at[] for safely setting values in a specific cell
        df.at[index, "People"] = extracted_info["people"]
        df.at[index, "Locations"] = extracted_info["locations"]
        df.at[index, "Organizations"] = extracted_info["organizations"]

    # --- 5. Display the Final, Enriched Data ---
    print("\n--- Analysis Complete ---")
    # Set pandas display options to show all columns and wider text
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(df)


# --- 6. (Optional) Create a Visual Timeline ---
# This block should be added at the end of the `if __name__ == "__main__":` block
# in your analyzer.py script.

print("\nGenerating visual timeline...")

# Convert date strings to datetime objects with a custom function
def safe_date_conversion(date_str):
    try:
        # First try pandas datetime conversion
        return pd.to_datetime(date_str)
    except:
        # For very old dates, use just the year
        year = int(date_str.split('-')[0])
        # Use a reference date for plotting
        return pd.Timestamp(f'{1677 if year < 1677 else year}-01-01')

# Convert and sort dates
df['plot_date'] = df['date'].apply(safe_date_conversion)
df = df.sort_values(by='plot_date')

# Create the plot
fig, ax = plt.subplots(figsize=(15, 10))

# Calculate levels for event spacing
base_levels = [-5, 5, -3, 3, -1, 1]
levels = (base_levels * (len(df) // len(base_levels) + 1))[:len(df)]

# Create timeline spine
ax.vlines(df['plot_date'], 0, levels, color="tab:red")

# Plot events
ax.plot(df['plot_date'], [0]*len(df), "-o",
        color="k", markerfacecolor="w")

# Add event labels
for i, (date, event_text) in enumerate(zip(df['plot_date'], df['event'])):
    # Create a more compact label if text is too long
    label = event_text if len(event_text) < 30 else event_text[:27] + "..."
    ax.text(date, levels[i], label,
            horizontalalignment='right' if i % 2 == 0 else 'left',
            verticalalignment='bottom' if levels[i] > 0 else 'top',
            rotation=0,
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

# Format the plot
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.yaxis.set_visible(False)
ax.spines[['left', 'top', 'right']].set_visible(False)
ax.set_title('Historical Events Timeline', fontweight='bold', fontsize=16)

# Adjust layout and display
plt.tight_layout()
plt.show()