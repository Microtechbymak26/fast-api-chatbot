

import os
from dotenv import load_dotenv
import chainlit as cl
from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 🌐 Load environment variables
load_dotenv()
gemini_api = os.getenv("GEMINI_API_KEY")

# 🔐 Check API key
if not gemini_api:
    raise ValueError("GEMINI_API_KEY not found in .env file!")

# 🤖 Setup Gemini Client
external_client = AsyncOpenAI(
    api_key=gemini_api,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# 🔧 Define model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# ⚙️ Run Config
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# 🧠 Define Agent
agent = Agent(
    name="MAK Assistant",
    instructions="""
👋 Hello! I’m MAK Assistant.
I’m here to help you understand what MAK does and how you can connect with him.

👨‍💻 Who is MAK?
MAK is a Full-Stack Developer focused on building interactive and user-friendly web applications. He’s passionate about clean design, efficient performance, and smart integration.

🛠️ MAK’s Skills:
Frontend Development: HTML, CSS, JavaScript, TypeScript, TailwindCSS, Vite, and advanced UI/UX design.
Backend Development: NestJS and seamless integration with Sanity CMS.
AI Integration: Actively learning and implementing Agentic AI into web applications.
Continuous Learner: Always exploring new tools, frameworks, and technologies.

🔗 Want to Connect?
Check out his LinkedIn in the portfolio footer for collaborations or questions about his work.

🚫 Note:
Please ask only questions related to MAK’s skills or work.
If your question is unrelated (e.g. “What is frontend development?”), I’ll reply:
“Sorry, I can’t answer that. Please ask something related to MAK’s work or skills.”
"""
)

# ✨ Chainlit UI (optional if using chainlit run)
@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    await cl.Message(content="👋 Hello! I'm MAK Assistant. How can I help you today?").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    result = Runner.run_streamed(agent, input=history, run_config=config)

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

# 🌐 FastAPI app for frontend chatbot integration
app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("input")

    if not messages:
        return JSONResponse(content={"error": "Missing input"}, status_code=400)

    result = await Runner.run_async(agent=agent, input=messages, run_config=config)

    return JSONResponse(content={"reply": result.final_output})
     