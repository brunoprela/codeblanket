import { djangoArchitectureDeepDive } from '../sections/python-django/django-architecture-deep-dive';
import { djangoOrmAdvancedTechniques } from '../sections/python-django/django-orm-advanced-techniques';
import { customManagersQuerysets } from '../sections/python-django/custom-managers-querysets';
import { signalsHooks } from '../sections/python-django/signals-hooks';
import { middlewareDevelopment } from '../sections/python-django/middleware-development';
import { customDjangoAdmin } from '../sections/python-django/custom-django-admin';
import { djangoRestFrameworkFundamentals } from '../sections/python-django/django-rest-framework-fundamentals';
import { drfSerializersDeepDive } from '../sections/python-django/drf-serializers-deep-dive';
import { drfViewsetsRouters } from '../sections/python-django/drf-viewsets-routers';
import { authenticationPermissionsDrf } from '../sections/python-django/authentication-permissions-drf';
import { filteringSearchingPagination } from '../sections/python-django/filtering-searching-pagination';
import { cachingDjango } from '../sections/python-django/caching-django';
import { celeryDjangoIntegration } from '../sections/python-django/celery-django-integration';
import { testingDjangoApplications } from '../sections/python-django/testing-django-applications';
import { djangoSecurityBestPractices } from '../sections/python-django/django-security-best-practices';
import { djangoProductionDeployment } from '../sections/python-django/django-production-deployment';

import { djangoArchitectureDeepDiveQuiz } from '../quizzes/python-django/django-architecture-deep-dive';
import { djangoOrmAdvancedTechniquesQuiz } from '../quizzes/python-django/django-orm-advanced-techniques';
import { customManagersQuerysetsQuiz } from '../quizzes/python-django/custom-managers-querysets';
import { signalsHooksQuiz } from '../quizzes/python-django/signals-hooks';
import { middlewareDevelopmentQuiz } from '../quizzes/python-django/middleware-development';
import { customDjangoAdminQuiz } from '../quizzes/python-django/custom-django-admin';
import { djangoRestFrameworkFundamentalsQuiz } from '../quizzes/python-django/django-rest-framework-fundamentals';
import { drfSerializersDeepDiveQuiz } from '../quizzes/python-django/drf-serializers-deep-dive';
import { drfViewsetsRoutersQuiz } from '../quizzes/python-django/drf-viewsets-routers';
import { authenticationPermissionsDrfQuiz } from '../quizzes/python-django/authentication-permissions-drf';
import { filteringSearchingPaginationQuiz } from '../quizzes/python-django/filtering-searching-pagination';
import { cachingDjangoQuiz } from '../quizzes/python-django/caching-django';
import { celeryDjangoIntegrationQuiz } from '../quizzes/python-django/celery-django-integration';
import { testingDjangoApplicationsQuiz } from '../quizzes/python-django/testing-django-applications';
import { djangoSecurityBestPracticesQuiz } from '../quizzes/python-django/django-security-best-practices';
import { djangoProductionDeploymentQuiz } from '../quizzes/python-django/django-production-deployment';

import { DjangoArchitectureDeepDiveMultipleChoice } from '../multiple-choice/python-django/django-architecture-deep-dive';
import { DjangoOrmAdvancedTechniquesMultipleChoice } from '../multiple-choice/python-django/django-orm-advanced-techniques';
import { CustomManagersQuerysetsMultipleChoice } from '../multiple-choice/python-django/custom-managers-querysets';
import { SignalsHooksMultipleChoice } from '../multiple-choice/python-django/signals-hooks';
import { MiddlewareDevelopmentMultipleChoice } from '../multiple-choice/python-django/middleware-development';
import { CustomDjangoAdminMultipleChoice } from '../multiple-choice/python-django/custom-django-admin';
import { DjangoRestFrameworkFundamentalsMultipleChoice } from '../multiple-choice/python-django/django-rest-framework-fundamentals';
import { DrfSerializersDeepDiveMultipleChoice } from '../multiple-choice/python-django/drf-serializers-deep-dive';
import { DrfViewsetsRoutersMultipleChoice } from '../multiple-choice/python-django/drf-viewsets-routers';
import { AuthenticationPermissionsDrfMultipleChoice } from '../multiple-choice/python-django/authentication-permissions-drf';
import { FilteringSearchingPaginationMultipleChoice } from '../multiple-choice/python-django/filtering-searching-pagination';
import { CachingDjangoMultipleChoice } from '../multiple-choice/python-django/caching-django';
import { CeleryDjangoIntegrationMultipleChoice } from '../multiple-choice/python-django/celery-django-integration';
import { TestingDjangoApplicationsMultipleChoice } from '../multiple-choice/python-django/testing-django-applications';
import { DjangoSecurityBestPracticesMultipleChoice } from '../multiple-choice/python-django/django-security-best-practices';
import { DjangoProductionDeploymentMultipleChoice } from '../multiple-choice/python-django/django-production-deployment';

export const pythonDjangoModule = {
  id: 'python-django',
  title: 'Django Advanced & Django REST Framework',
  description:
    'Master Django for complex web applications and REST APIs. Build scalable web applications with Django ORM, custom managers, signals, middleware, admin customization, and Django REST Framework for production-grade APIs.',
  icon: '🎸',
  keyTakeaways: [
    "Understand Django's MVT architecture and request/response lifecycle",
    'Master advanced ORM techniques including select_related, prefetch_related, and custom SQL',
    'Build custom managers and QuerySets for reusable query logic',
    'Implement signals for decoupled application components',
    'Create custom middleware for authentication, logging, and request processing',
    'Customize Django Admin with actions, filters, and inline editing',
    'Build production REST APIs with Django REST Framework',
    'Master DRF serializers with validation, nested objects, and custom fields',
    'Implement authentication (JWT, OAuth2) and granular permissions',
    'Add filtering, searching, pagination, and versioning to APIs',
    'Optimize performance with caching strategies (Redis, Memcached, database caching)',
    'Integrate Celery for background task processing in Django',
    'Write comprehensive tests for models, views, serializers, and APIs',
    'Implement Django security best practices (CSRF, XSS, SQL injection prevention)',
    'Deploy Django applications to production with Gunicorn, NGINX, and Docker',
    'Master production deployment including database migrations, static files, and monitoring',
  ],
  sections: [
    {
      ...djangoArchitectureDeepDive,
      quiz: djangoArchitectureDeepDiveQuiz,
      multipleChoice: DjangoArchitectureDeepDiveMultipleChoice,
    },
    {
      ...djangoOrmAdvancedTechniques,
      quiz: djangoOrmAdvancedTechniquesQuiz,
      multipleChoice: DjangoOrmAdvancedTechniquesMultipleChoice,
    },
    {
      ...customManagersQuerysets,
      quiz: customManagersQuerysetsQuiz,
      multipleChoice: CustomManagersQuerysetsMultipleChoice,
    },
    {
      ...signalsHooks,
      quiz: signalsHooksQuiz,
      multipleChoice: SignalsHooksMultipleChoice,
    },
    {
      ...middlewareDevelopment,
      quiz: middlewareDevelopmentQuiz,
      multipleChoice: MiddlewareDevelopmentMultipleChoice,
    },
    {
      ...customDjangoAdmin,
      quiz: customDjangoAdminQuiz,
      multipleChoice: CustomDjangoAdminMultipleChoice,
    },
    {
      ...djangoRestFrameworkFundamentals,
      quiz: djangoRestFrameworkFundamentalsQuiz,
      multipleChoice: DjangoRestFrameworkFundamentalsMultipleChoice,
    },
    {
      ...drfSerializersDeepDive,
      quiz: drfSerializersDeepDiveQuiz,
      multipleChoice: DrfSerializersDeepDiveMultipleChoice,
    },
    {
      ...drfViewsetsRouters,
      quiz: drfViewsetsRoutersQuiz,
      multipleChoice: DrfViewsetsRoutersMultipleChoice,
    },
    {
      ...authenticationPermissionsDrf,
      quiz: authenticationPermissionsDrfQuiz,
      multipleChoice: AuthenticationPermissionsDrfMultipleChoice,
    },
    {
      ...filteringSearchingPagination,
      quiz: filteringSearchingPaginationQuiz,
      multipleChoice: FilteringSearchingPaginationMultipleChoice,
    },
    {
      ...cachingDjango,
      quiz: cachingDjangoQuiz,
      multipleChoice: CachingDjangoMultipleChoice,
    },
    {
      ...celeryDjangoIntegration,
      quiz: celeryDjangoIntegrationQuiz,
      multipleChoice: CeleryDjangoIntegrationMultipleChoice,
    },
    {
      ...testingDjangoApplications,
      quiz: testingDjangoApplicationsQuiz,
      multipleChoice: TestingDjangoApplicationsMultipleChoice,
    },
    {
      ...djangoSecurityBestPractices,
      quiz: djangoSecurityBestPracticesQuiz,
      multipleChoice: DjangoSecurityBestPracticesMultipleChoice,
    },
    {
      ...djangoProductionDeployment,
      quiz: djangoProductionDeploymentQuiz,
      multipleChoice: DjangoProductionDeploymentMultipleChoice,
    },
  ],
};
