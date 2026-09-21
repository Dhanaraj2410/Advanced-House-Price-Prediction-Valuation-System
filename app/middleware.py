import time
import logging

logger = logging.getLogger(__name__)

class RequestPerformanceMiddleware:
    """
    Custom Django middleware that measures request processing latency
    and attaches an 'X-Response-Time-ms' header to responses.
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()
        response = self.get_response(request)
        duration_ms = (time.time() - start_time) * 1000.0

        response['X-Response-Time-ms'] = f"{duration_ms:.2f}"

        if duration_ms > 500:
            logger.warning(f"Slow request: {request.method} {request.path} took {duration_ms:.2f}ms")

        return response
