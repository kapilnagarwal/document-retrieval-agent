from fastapi import APIRouter, Request

from backend.models.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request):
    services = request.app.state.services
    return services.pipeline.answer(body)
