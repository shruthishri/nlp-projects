import spacy
import matplotlib.pyplot as plt
from collections import Counter

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "Apple is looking at buying U.K. startup for $1 billion"

# Process text
doc = nlp(text)

# Extract entities
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(entities)

# Visualization
labels = [ent.label_ for ent in doc.ents]
label_counts = Counter(labels)

plt.bar(label_counts.keys(), label_counts.values())
plt.title('Entity Counts')
plt.xlabel('Entity Type')
plt.ylabel('Count')
plt.show()
