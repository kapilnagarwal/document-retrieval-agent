import json
from queue import Empty, Queue
from threading import Thread
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.models.schemas import ChatRequest, ChatResponse, ChatTurn
from backend.rag.generator import GenerationError

router = APIRouter()


class NewConversation(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=120)
    scope: str = Field(default="all", pattern="^(all|notes|pdf)$")
    document_ids: list[str] = Field(default_factory=list, max_length=200)


@router.get("/conversations")
def conversations(request: Request):
    return request.app.state.services.chats.list()


@router.post("/conversations", status_code=201)
def create_conversation(body: NewConversation, request: Request):
    return request.app.state.services.chats.create(body.title, body.model_dump(exclude={"title"}))


@router.get("/conversations/{identifier}")
def get_conversation(identifier: str, request: Request):
    try:
        return request.app.state.services.chats.get(identifier)
    except KeyError:
        raise HTTPException(404, "Conversation not found") from None


@router.delete("/conversations/{identifier}")
def delete_conversation(identifier: str, request: Request):
    try:
        request.app.state.services.chats.delete(identifier)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"deleted": identifier}


def prepare(body, services):
    if not body.conversation_id:
        return body, None
    try:
        conversation = services.chats.get(body.conversation_id)
        settings = conversation["settings"]
        if settings != {"scope": body.scope, "document_ids": body.document_ids}:
            raise ValueError("Document filters changed. Start a new conversation.")
        request_id = body.request_id or str(uuid4())
        cached = services.chats.begin(body.conversation_id, request_id, body.question)
        conversation = services.chats.get(body.conversation_id)
        turns = [t["response"] for t in conversation["turns"] if t["status"] == "complete"][-4:]
        body = body.model_copy(
            update={
                "request_id": request_id,
                "history": [
                    ChatTurn(question=t["question"][:8000], answer=t["answer"][:32000])
                    for t in turns
                ]
                or body.history,
            }
        )
        return body, cached
    except KeyError:
        raise HTTPException(404, "Conversation not found") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


def execute(body, services, emit=None):
    try:
        result = services.pipeline.answer(body, emit=emit)
        result.update(conversation_id=body.conversation_id, request_id=body.request_id)
        result = ChatResponse(**result).model_dump()
        if body.conversation_id:
            services.chats.finish(body.request_id, result)
        return result
    except Exception:
        if body.conversation_id:
            services.chats.finish(body.request_id)
        raise


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request):
    services = request.app.state.services
    body, cached = prepare(body, services)
    return cached or execute(body, services)


@router.post("/chat/stream")
def stream_chat(body: ChatRequest, request: Request):
    services = request.app.state.services
    body, cached = prepare(body, services)
    # The producer finishes and saves even if the browser disconnects. Reopening
    # the conversation recovers the final response without repeating the provider call.
    events = Queue()

    def emit(kind, payload):
        events.put({"event": kind, "data": payload})

    def produce():
        try:
            emit("start", {"conversation_id": body.conversation_id, "request_id": body.request_id})
            emit("final", cached or execute(body, services, emit))
        except Exception as exc:
            emit(
                "error",
                {
                    "message": str(exc)
                    if isinstance(exc, GenerationError)
                    else "Generation failed. Reopen the saved conversation and retry."
                },
            )
        finally:
            events.put(None)

    worker = Thread(target=produce, daemon=True, name="chat-stream")
    services.stream_threads[:] = [thread for thread in services.stream_threads if thread.is_alive()]
    services.stream_threads.append(worker)
    worker.start()

    def lines():
        while True:
            try:
                event = events.get(timeout=10)
            except Empty:
                yield json.dumps({"event": "heartbeat", "data": {}}) + "\n"
                continue
            if event is None:
                return
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
