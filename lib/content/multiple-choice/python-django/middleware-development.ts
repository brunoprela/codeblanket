import { MultipleChoiceQuestion } from '@/lib/types';

export const MiddlewareDevelopmentMultipleChoice = [
  {
    id: 1,
    question:
      'In Django middleware, what order do response processing methods execute?',
    options: [
      'A) Same order as defined in MIDDLEWARE setting',
      'B) Reverse order of MIDDLEWARE setting',
      'C) Random order based on load',
      'D) Alphabetically by middleware name',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Reverse order of MIDDLEWARE setting**

Request processing goes top-to-bottom, response processing goes bottom-to-top through the MIDDLEWARE list.

\`\`\`python
MIDDLEWARE = [
    'SecurityMiddleware',    # 1st for requests, 4th for responses
    'SessionMiddleware',     # 2nd for requests, 3rd for responses
    'AuthMiddleware',        # 3rd for requests, 2nd for responses
    'CustomMiddleware',      # 4th for requests, 1st for responses
]
\`\`\`

This allows outer middleware to wrap inner middleware responses.
      `,
  },
  {
    question:
      'What should middleware return to short-circuit the request and prevent the view from executing?',
    options: [
      'A) None',
      'B) An HttpResponse object',
      'C) False',
      'D) raise HttpResponseForbidden',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) An HttpResponse object**

Returning an HttpResponse from middleware skips the view and remaining middleware, immediately starting the response phase.

\`\`\`python
class RateLimitMiddleware:
    def __call__(self, request):
        if is_rate_limited (request):
            # Short-circuit: skip view, return immediately
            return JsonResponse({'error': 'Rate limited'}, status=429)
        
        response = self.get_response (request)
        return response
\`\`\`

This is useful for authentication, rate limiting, and maintenance mode.
      `,
  },
  {
    question:
      'Which middleware hook is best for handling exceptions raised by views?',
    options: [
      'A) __call__',
      'B) process_request',
      'C) process_exception',
      'D) process_response',
    ],
    correctAnswer: 2,
    explanation: `
**Correct Answer: C) process_exception**

\`process_exception()\` is specifically designed to handle exceptions from views and other middleware.

\`\`\`python
class ErrorHandlingMiddleware:
    def process_exception (self, request, exception):
        if isinstance (exception, ValidationError):
            return JsonResponse({
                'error': str (exception)
            }, status=400)
        
        # Return None to let other middleware handle it
        return None
\`\`\`

Return an HttpResponse to handle the exception, or None to propagate it.
      `,
  },
  {
    question:
      'When implementing request logging middleware, where should you store request-specific data?',
    options: [
      'A) In the middleware class instance',
      'B) As request attributes',
      'C) In global variables',
      'D) In the response object',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) As request attributes**

Store request-specific data as attributes on the request object to pass data between middleware and views.

\`\`\`python
class TimingMiddleware:
    def __call__(self, request):
        # Store on request object
        request.start_time = time.time()
        
        response = self.get_response (request)
        
        # Access stored data
        duration = time.time() - request.start_time
        response['X-Request-Time'] = f'{duration:.3f}s'
        
        return response
\`\`\`

Never use class instance variables (not thread-safe) or globals (shared across requests).
      `,
  },
  {
    question:
      'What is the purpose of the get_response parameter in middleware __init__?',
    options: [
      'A) To get the final HTTP response',
      'B) To call the next middleware or view in the chain',
      'C) To access the response headers',
      'D) To bypass remaining middleware',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) To call the next middleware or view in the chain**

\`get_response\` is a callable that invokes the next middleware in the chain or the view if this is the last middleware.

\`\`\`python
class CustomMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Before view
        print("Request processing")
        
        # Call next middleware/view
        response = self.get_response (request)
        
        # After view
        print("Response processing")
        
        return response
\`\`\`

This creates a chain where each middleware wraps the next.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));
