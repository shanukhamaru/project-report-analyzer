from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from app.core.config import settings

def main():
    print("Base URL:", settings.LMSTUDIO_API_BASE)
    print("Model:", settings.LMSTUDIO_MODEL)

    llm = ChatOpenAI(
        model=settings.LMSTUDIO_MODEL,
        temperature=0,
        base_url=settings.LMSTUDIO_API_BASE.strip(),
        api_key=settings.LMSTUDIO_API_KEY,
    )

    response = llm.invoke("Say hello in one sentence.")
    print("\nLLM Response:")
    print(response.content)

if __name__ == "__main__":
    main()
