import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    """
    Quick sanity check script for the custom gpt-oss-120b model.

    Usage:
        OPENAI_API_KEY=... OPENAI_BASE_URL=... python test_gpt_oss_120b.py
    or set these values in a .env file in the same directory.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key or not base_url:
        raise RuntimeError(
            "Both OPENAI_API_KEY and OPENAI_BASE_URL must be set to test gpt-oss-120b."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Testing model 'gpt-oss-120b' against base URL: {base_url}")

    try:
        # Try the new Responses API first (what the app uses)
        resp = client.responses.create(
            model="gpt-oss-120b",
            input=[
                {
                    "role": "user",
                    "content": "Reply with a short sentence confirming this model is working.",
                }
            ],
        )
        print("Responses API call succeeded.")
        print("Raw response:")
        print(resp)
    except Exception as e:
        print("Responses API call failed, error:")
        print(e)
        print("\nTrying legacy chat.completions API instead...\n")

        try:
            chat_resp = client.chat.completions.create(
                model="gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with a short sentence confirming this model is working.",
                    }
                ],
                max_tokens=50,
            )
            print("chat.completions API call succeeded.")
            print("First choice content:")
            print(chat_resp.choices[0].message.content)
        except Exception as e2:
            print("chat.completions API call also failed.")
            print(e2)


if __name__ == "__main__":
    main()

