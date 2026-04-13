#!/usr/bin/env python
"""Test API Gateway functionality."""

import sys

def test_imports():
    """Test API Gateway imports."""
    print("Testing API Gateway imports...")
    try:
        from api_gateway.app import app
        from api_gateway.routes import router
        from api_gateway.service_client import ServiceClient, ServiceRouter
        from api_gateway.task_manager import task_manager
        from api_gateway.auth import verify_api_key
        from api_gateway.rate_limiter import RateLimiter
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fastapi_app():
    """Test FastAPI app creation."""
    print("\nTesting FastAPI app...")
    try:
        from api_gateway.app import app
        assert app is not None
        assert app.title == "Local Media Processor - API Gateway"
        print(f"  ✓ FastAPI app created: {app.title}")
        return True
    except Exception as e:
        print(f"  ✗ FastAPI app failed: {e}")
        return False


def test_routes():
    """Test route definitions."""
    print("\nTesting routes...")
    try:
        from api_gateway.app import app
        routes = [r.path for r in app.routes]
        
        expected_routes = ["/", "/health", "/docs", "/redoc", "/openapi.json"]
        for route in expected_routes:
            assert route in routes, f"Missing route: {route}"
        
        # Check API routes
        api_routes = [r.path for r in app.routes if r.path.startswith("/api")]
        assert len(api_routes) > 0, "No API routes defined"
        
        print(f"  ✓ Routes defined: {len(routes)} total, {len(api_routes)} API routes")
        return True
    except Exception as e:
        print(f"  ✗ Routes failed: {e}")
        return False


def test_service_client():
    """Test service client."""
    print("\nTesting service client...")
    try:
        from api_gateway.service_client import ServiceClient
        client = ServiceClient("http://localhost:8001")
        assert client.base_url == "http://localhost:8001"
        assert client.timeout == 30.0
        print(f"  ✓ Service client created: {client.base_url}")
        return True
    except Exception as e:
        print(f"  ✗ Service client failed: {e}")
        return False


def test_task_manager():
    """Test task manager."""
    print("\nTesting task manager...")
    try:
        from api_gateway.task_manager import task_manager
        assert task_manager is not None
        print("  ✓ Task manager created")
        return True
    except Exception as e:
        print(f"  ✗ Task manager failed: {e}")
        return False


def test_rate_limiter():
    """Test rate limiter."""
    print("\nTesting rate limiter...")
    try:
        from api_gateway.rate_limiter import RateLimiter
        limiter = RateLimiter(requests_per_minute=5, requests_per_hour=100)
        assert limiter.is_allowed("test") is True
        assert limiter.is_allowed("test") is True
        print("  ✓ Rate limiter works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Rate limiter failed: {e}")
        return False


def test_http_client():
    """Test HTTP client with TestClient."""
    print("\nTesting HTTP client...")
    try:
        from fastapi.testclient import TestClient
        from api_gateway.app import app
        
        with TestClient(app) as client:
            # Test root
            response = client.get("/")
            assert response.status_code == 200
            print("  ✓ Root endpoint works")
            
            # Test health
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["service"] == "api-gateway"
            print("  ✓ Health endpoint works")
            
            # Test docs
            response = client.get("/docs")
            assert response.status_code == 200
            print("  ✓ OpenAPI docs accessible")
        
        return True
    except Exception as e:
        print(f"  ✗ HTTP client failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing API Gateway Service")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_fastapi_app,
        test_routes,
        test_service_client,
        test_task_manager,
        test_rate_limiter,
        test_http_client,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if all(results):
        print("✓ All API Gateway tests passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
