"""Main entrypoint for the app."""
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.memory.buffer import ConversationBufferMemory
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from courses import CourseManager
from prompt import TUTOR_PROMPT
from query_data import get_chain
from schemas import ChatResponse
from tutor_conversation import get_tutor_chain, COURSE_INTRODUCTION_MESSAGE

# from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


# @app.on_event("startup")
# async def startup_event():
#     logging.info("loading vectorstore")
#     if not Path("vectorstore.pkl").exists():
#         raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
#     with open("vectorstore.pkl", "rb") as f:
#         global vectorstore
#         vectorstore = pickle.load(f)


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()

    if client_id not in os.environ.get("AUTHORIZED_CLIENT_IDS", []):
        return

    course_manager = CourseManager(client_id)
    logging.info(f"[{client_id}] - Connection authorized.")
    question_handler = QuestionGenCallbackHandler(websocket, client_id)
    stream_handler = StreamingLLMCallbackHandler(websocket, client_id)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)
    while True:
        try:
            # Receive and send back the client message
            response = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=response, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            response, content_question = course_manager.update_inputs(response, chat_history)
            result = await qa_chain.acall(
                {
                    "response": response,
                    "question": content_question,
                    "chat_history": chat_history,
                }
            )
            chat_history.append((response, result["answer"]))
            logging.info(f"Etudiant: {response}")
            logging.info(f"Parolla: {result['answer']}")
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Oups, une erreur est survenue, veuillez réessayer.",
                type="error",
            )
            await websocket.send_json(resp.dict())


@app.websocket("/tutor/{client_id}")
async def tutor_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()

    if client_id not in os.environ.get("AUTHORIZED_CLIENT_IDS", []):
        return

    logging.info(f"[tutor][{client_id}] - Connection authorized.")
    stream_handler = StreamingLLMCallbackHandler(websocket, client_id)
    conversation_memory = ConversationBufferMemory(ai_prefix="Professeur", human_prefix="Étudiant")
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    tutor_chain = get_tutor_chain(stream_handler, conversation_memory, TUTOR_PROMPT, tracing=False)
    for paragraph in COURSE_INTRODUCTION_MESSAGE.split("\n\n"):
        start_resp = ChatResponse(sender="bot", message="", type="start")
        await websocket.send_json(start_resp.dict())
        introduction = ChatResponse(
            sender="bot",
            message=paragraph,
            type="stream",
        )
        new_line = ChatResponse(
            sender="bot",
            message="\n",
            type="stream",
        )
        await websocket.send_json(introduction.dict())
        await websocket.send_json(new_line.dict())
        end_resp = ChatResponse(sender="bot", message="", type="end")
        await websocket.send_json(end_resp.dict())
    while True:
        try:
            # Receive and send back the client message
            response = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=response, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await tutor_chain.acall({"input": response})
            logging.info(f"[tutor][{client_id}] - Etudiant: {response}")
            logging.info(f"[tutor][{client_id}] - Parolla: {result}")
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info(f"[tutor][{client_id}] - websocket disconnect")
            break
        except Exception as e:
            logging.exception(e)
            resp = ChatResponse(
                sender="bot",
                message="Oups, une erreur est survenue, veuillez réessayer.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_config="log.ini")
