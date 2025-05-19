from openai import OpenAI

api_key = ""
client = OpenAI(api_key=api_key)

def query_api(user_input):

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a helpful assistant. Your name is Inmoov. You are a robot. Do not provide long responses. Answer with 1-3 sentences"""},
            {
                "role": "user",
                "content": user_input
            }
        ]
    )

    return completion.choices[0].message.content





