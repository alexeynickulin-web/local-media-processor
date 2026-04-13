#!/usr/bin/env python
"""Test script to verify Redis + Celery infrastructure."""

import sys

def test_imports():
    """Test all shared imports."""
    print("Testing imports...")
    try:
        from shared import (
            get_settings,
            create_celery_app,
            setup_logging,
            TaskStatus,
            TaskResponse,
            HealthResponse,
            validate_language_code,
            validate_file_exists,
        )
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_settings():
    """Test settings loading."""
    print("\nTesting settings...")
    try:
        from shared import get_settings
        settings = get_settings()
        assert settings.redis_host == "localhost"
        assert settings.redis_port == 6379
        print(f"  ✓ Settings loaded: redis://{settings.redis_host}:{settings.redis_port}")
        return True
    except Exception as e:
        print(f"  ✗ Settings failed: {e}")
        return False


def test_celery():
    """Test Celery app creation."""
    print("\nTesting Celery app...")
    try:
        from shared import create_celery_app
        app = create_celery_app("test-service")
        assert app.main == "test-service"
        assert app.conf.task_serializer == "json"
        assert app.conf.timezone == "UTC"
        print(f"  ✓ Celery app created: {app.main}")
        print(f"  ✓ Serializer: {app.conf.task_serializer}")
        return True
    except Exception as e:
        print(f"  ✗ Celery failed: {e}")
        return False


def test_logging():
    """Test logging setup."""
    print("\nTesting logging...")
    try:
        from shared import setup_logging
        logger = setup_logging("test-service")
        logger.info("Test message")
        print("  ✓ Logging configured")
        return True
    except Exception as e:
        print(f"  ✗ Logging failed: {e}")
        return False


def test_models():
    """Test Pydantic models."""
    print("\nTesting models...")
    try:
        from shared import TaskResponse, TaskStatus, HealthResponse
        from datetime import datetime
        
        task = TaskResponse(
            task_id="test-123",
            status=TaskStatus.PENDING,
            message="Test task"
        )
        assert task.task_id == "test-123"
        assert task.status == TaskStatus.PENDING
        
        health = HealthResponse(
            service="test",
            status="healthy"
        )
        assert health.service == "test"
        
        print("  ✓ Models work correctly")
        return True
    except Exception as e:
        print(f"  ✗ Models failed: {e}")
        return False


def test_language_validation():
    """Test language code validation."""
    print("\nTesting language validation...")
    try:
        from shared import validate_language_code, get_nllb_code
        
        assert validate_language_code("en") == "en"
        assert validate_language_code("ru") == "ru"
        assert get_nllb_code("en") == "eng_Latn"
        assert get_nllb_code("ru") == "rus_Cyrl"
        
        print("  ✓ Language validation works")
        return True
    except Exception as e:
        print(f"  ✗ Language validation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Redis + Celery Infrastructure")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_settings,
        test_celery,
        test_logging,
        test_models,
        test_language_validation,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if all(results):
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
