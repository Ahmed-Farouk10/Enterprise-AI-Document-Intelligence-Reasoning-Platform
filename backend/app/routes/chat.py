from fastapi import APIRouter, HTTPException
from typing import List

from app.schemas import (
    ChatSessionCreate, 
    ChatSessionResponse, 
    ChatMessageCreate, 
    ChatMessageResponse
)
from app.database import Database

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_sessions():
    """Get all chat sessions"""
    sessions = Database.get_all_sessions()
    
    # Add messages to each session
    for session in sessions:
        session["messages"] = []  # Don't load all messages in list view
    
    return sessions

@router.post("/sessions", response_model=ChatSessionResponse)
async def create_session(session_data: ChatSessionCreate):
    """Create a new chat session"""
    session = Database.create_chat_session(
        title=session_data.title,
        document_ids=session_data.document_ids
    )
    session["messages"] = []
    return session

@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_session(session_id: str):
    """Get a specific chat session with all messages"""
    session = Database.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    success = Database.delete_chat_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"success": True, "message": "Session deleted successfully"}

@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_message(session_id: str, message_data: ChatMessageCreate):
    """Send a message in a chat session"""
    
    # Check if session exists
    session = Database.get_chat_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Create user message
    user_message = Database.create_message(
        session_id=session_id,
        role="user",
        content=message_data.content
    )
    
    # Generate AI response (simplified - just echo for now)
    ai_response_content = f"I received your message: '{message_data.content}'. This is a placeholder response. In production, this would call your AI model."
    
    ai_message = Database.create_message(
        session_id=session_id,
        role="assistant",
        content=ai_response_content
    )
    
    # Return the AI message (frontend will optimistically add user message)
    return ai_message

@router.post("/sessions/{session_id}/stream", response_model=ChatMessageResponse)
async def stream_message(session_id: str, message_data: ChatMessageCreate):
    """Stream a message response (placeholder for now)"""
    # For now, just use the regular send_message
    return await send_message(session_id, message_data)