"""
Job/task status tracking routes
"""

from fastapi import APIRouter, HTTPException
from app.workers.celery_app import celery_app
from app.core.logging_config import get_logger

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
logger = get_logger(__name__)


@router.get("/{task_id}/status")
async def get_job_status(task_id: str):
    """
    Get the status of a background job/task
    
    Returns:
        - state: PENDING, STARTED, SUCCESS, FAILURE, RETRY
        - result: Task result if completed
        - info: Additional task information
    """
    task = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "state": task.state,
        "ready": task.ready(),
    }
    
    if task.state == "PENDING":
        response["info"] = "Task is queued or does not exist"
        
    elif task.state == "STARTED":
        response["info"] = "Task is currently processing"
        
    elif task.state == "SUCCESS":
        response["result"] = task.result
        response["info"] = "Task completed successfully"
        
    elif task.state == "FAILURE":
        response["error"] = str(task.info)
        response["info"] = "Task failed"
        
    elif task.state == "RETRY":
        response["info"] = "Task is being retried"
        
    else:
        response["info"] = task.info
    
    logger.info("job_status_checked", task_id=task_id, state=task.state)
    
    return response


@router.delete("/{task_id}")
async def cancel_job(task_id: str):
    """
    Cancel a running job/task
    """
    task = celery_app.AsyncResult(task_id)
    
    if task.state in ["PENDING", "STARTED"]:
        task.revoke(terminate=True)
        logger.info("job_cancelled", task_id=task_id)
        return {"message": "Task cancelled successfully", "task_id": task_id}
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel task in state: {task.state}"
        )
