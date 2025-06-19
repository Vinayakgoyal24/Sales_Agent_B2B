
import openai

# Replace with your actual API key
client = openai.OpenAI(api_key="sk-proj-UZIz7Fn3dIy1BHywWVrXT3BlbkFJImUsx697xQbmU9cc0hyM")

def check_openai_key():
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"}
            ]
        )
        print("✅ API key is working. Response:")
        print(response.choices[0].message.content)
    except openai.AuthenticationError:
        print("❌ Invalid API key.")
    except Exception as e:
        print(f"⚠️ Something went wrong: {e}")

check_openai_key()
