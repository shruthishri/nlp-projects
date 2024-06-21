import openai

# Set up OpenAI API key
openai.api_key = 'your-api-key'

# Define prompt
prompt = "Once upon a time"

# Generate text
response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  max_tokens=100
)

# Print generated text
print(response.choices[0].text.strip())
