"""Task management for submitting and tracking Celery tasks."""

from celery.result import AsyncResult
from shared.celery_app import create_celery_app
from shared.logging_config import setup_logging
from shared.exceptions import ProcessingError
from datetime import datetime

logger = setup_logging("task-manager")

# Create Celery app for task management
celery_app = create_celery_app("api-gateway")


class TaskManager:
    """Manages Celery task submission and status tracking."""

    @staticmethod
    def submit_task(
        task_name: str,
        args: list | None = None,
        kwargs: dict | None = None,
        queue: str | None = None,
    ) -> str:
        """Submit a task to Celery.

        Args:
            task_name: Full task name (e.g., "transcription.transcribe")
            args: Task arguments
            kwargs: Task keyword arguments
            queue: Target queue name

        Returns:
            Task ID
        """
        try:
            task = celery_app.send_task(
                task_name,
                args=args or [],
                kwargs=kwargs or {},
                queue=queue,
            )
            logger.info(f"Task submitted: {task.id} ({task_name})")
            return task.id
        except Exception as e:
            logger.error(f"Failed to submit task {task_name}: {e}")
            raise ProcessingError(f"Failed to submit task: {e}") from e

    @staticmethod
    def get_task_status(task_id: str) -> dict:
        """Get status of a Celery task.

        Args:
            task_id: Task ID

        Returns:
            Dict with task status and result
        """
        try:
            result = AsyncResult(task_id, app=celery_app)

            response = {
                "task_id": task_id,
                "status": result.status.lower(),
                "ready": result.ready(),
            }

            if result.status == "SUCCESS":
                response["result"] = result.result
                response["completed_at"] = datetime.utcnow().isoformat()
            elif result.status == "FAILURE":
                response["error"] = str(result.result)
                response["traceback"] = result.traceback
            elif result.status == "STARTED":
                response["info"] = result.info

            return response

        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            raise ProcessingError(f"Failed to get task status: {e}") from e

    @staticmethod
    def revoke_task(task_id: str, terminate: bool = False) -> bool:
        """Revoke/cancel a task.

        Args:
            task_id: Task ID
            terminate: Whether to terminate running task

        Returns:
            True if revoked successfully
        """
        try:
            celery_app.control.revoke(task_id, terminate=terminate)
            logger.info(f"Task revoked: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to revoke task {task_id}: {e}")
            return False


# Global task manager instance
task_manager = TaskManager()
