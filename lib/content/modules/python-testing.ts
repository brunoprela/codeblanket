import { testingFundamentalsPytestBasics } from '../sections/python-testing/testing-fundamentals-pytest-basics';
import { testOrganizationStructure } from '../sections/python-testing/test-organization-structure';
import { fixturesDeepDive } from '../sections/python-testing/fixtures-deep-dive';
import { parametrizedTests } from '../sections/python-testing/parametrized-tests';
import { mockingWithUnittestMock } from '../sections/python-testing/mocking-with-unittest-mock';
import { pytestPluginsExtensions } from '../sections/python-testing/pytest-plugins-extensions';
import { testingDatabasesSqlalchemy } from '../sections/python-testing/testing-databases-sqlalchemy';
import { testingAsyncCode } from '../sections/python-testing/testing-async-code';
import { integrationTesting } from '../sections/python-testing/integration-testing';
import { testCoverage } from '../sections/python-testing/test-coverage';
import { testDrivenDevelopment } from '../sections/python-testing/test-driven-development';
import { propertyBasedTesting } from '../sections/python-testing/property-based-testing';
import { codeQualityTools } from '../sections/python-testing/code-quality-tools';
import { precommitHooksCICD } from '../sections/python-testing/precommit-hooks-cicd';
import { testingBestPractices } from '../sections/python-testing/testing-best-practices';

import { testingFundamentalsPytestBasicsQuiz } from '../quizzes/python-testing/testing-fundamentals-pytest-basics';
import { testOrganizationStructureQuiz } from '../quizzes/python-testing/test-organization-structure';
import { fixturesDeepDiveQuiz } from '../quizzes/python-testing/fixtures-deep-dive';
import { parametrizedTestsQuiz } from '../quizzes/python-testing/parametrized-tests';
import { mockingWithUnittestMockQuiz } from '../quizzes/python-testing/mocking-with-unittest-mock';
import { pytestPluginsExtensionsQuiz } from '../quizzes/python-testing/pytest-plugins-extensions';
import { testingDatabasesSqlalchemyQuiz } from '../quizzes/python-testing/testing-databases-sqlalchemy';
import { testingAsyncCodeQuiz } from '../quizzes/python-testing/testing-async-code';
import { integrationTestingQuiz } from '../quizzes/python-testing/integration-testing';
import { testCoverageQuiz } from '../quizzes/python-testing/test-coverage';
import { testDrivenDevelopmentQuiz } from '../quizzes/python-testing/test-driven-development';
import { propertyBasedTestingQuiz } from '../quizzes/python-testing/property-based-testing';
import { codeQualityToolsQuiz } from '../quizzes/python-testing/code-quality-tools';
import { precommitHooksCICDQuiz } from '../quizzes/python-testing/precommit-hooks-cicd';
import { testingBestPracticesQuiz } from '../quizzes/python-testing/testing-best-practices';

import { testingFundamentalsPytestBasicsMultipleChoice } from '../multiple-choice/python-testing/testing-fundamentals-pytest-basics';
import { testOrganizationStructureMultipleChoice } from '../multiple-choice/python-testing/test-organization-structure';
import { fixturesDeepDiveMultipleChoice } from '../multiple-choice/python-testing/fixtures-deep-dive';
import { parametrizedTestsMultipleChoice } from '../multiple-choice/python-testing/parametrized-tests';
import { mockingWithUnittestMockMultipleChoice } from '../multiple-choice/python-testing/mocking-with-unittest-mock';
import { pytestPluginsExtensionsMultipleChoice } from '../multiple-choice/python-testing/pytest-plugins-extensions';
import { testingDatabasesSqlalchemyMultipleChoice } from '../multiple-choice/python-testing/testing-databases-sqlalchemy';
import { testingAsyncCodeMultipleChoice } from '../multiple-choice/python-testing/testing-async-code';
import { integrationTestingMultipleChoice } from '../multiple-choice/python-testing/integration-testing';
import { testCoverageMultipleChoice } from '../multiple-choice/python-testing/test-coverage';
import { testDrivenDevelopmentMultipleChoice } from '../multiple-choice/python-testing/test-driven-development';
import { propertyBasedTestingMultipleChoice } from '../multiple-choice/python-testing/property-based-testing';
import { codeQualityToolsMultipleChoice } from '../multiple-choice/python-testing/code-quality-tools';
import { precommitHooksCICDMultipleChoice } from '../multiple-choice/python-testing/precommit-hooks-cicd';
import { testingBestPracticesMultipleChoice } from '../multiple-choice/python-testing/testing-best-practices';

export const pythonTestingModule = {
  id: 'python-testing',
  title: 'Testing & Code Quality Mastery',
  description:
    'Master professional testing practices with pytest, from fundamentals to advanced techniques. Build comprehensive test suites with fixtures, mocking, databases, async code, and integration testing. Enforce quality with TDD, coverage, linting, and CI/CD.',
  icon: '🧪',
  keyTakeaways: [
    'Master pytest fundamentals: assertions, fixtures, marks, and configuration',
    'Organize test suites professionally with proper structure and naming',
    'Use fixtures for clean setup/teardown with proper scoping and dependency injection',
    'Write data-driven tests with parametrization for comprehensive coverage',
    'Mock external dependencies effectively with unittest.mock and pytest-mock',
    'Extend pytest capabilities with essential plugins (xdist, cov, benchmark, mock)',
    'Test SQLAlchemy applications with transaction rollback and Factory Boy',
    'Write robust async tests with pytest-asyncio for FastAPI and aiohttp',
    'Build integration tests with real dependencies (Docker, PostgreSQL, Redis)',
    'Measure and enforce code coverage (80-90% target) with pytest-cov',
    'Practice test-driven development (TDD) with Red-Green-Refactor cycle',
    'Use property-based testing with Hypothesis for edge case discovery',
    'Enforce code quality with Black, Ruff, mypy, pylint, and bandit',
    'Automate quality checks with pre-commit hooks and CI/CD pipelines',
    'Apply testing best practices and patterns for maintainable test suites',
  ],
  sections: [
    {
      ...testingFundamentalsPytestBasics,
      quiz: testingFundamentalsPytestBasicsQuiz,
      multipleChoice: testingFundamentalsPytestBasicsMultipleChoice,
    },
    {
      ...testOrganizationStructure,
      quiz: testOrganizationStructureQuiz,
      multipleChoice: testOrganizationStructureMultipleChoice,
    },
    {
      ...fixturesDeepDive,
      quiz: fixturesDeepDiveQuiz,
      multipleChoice: fixturesDeepDiveMultipleChoice,
    },
    {
      ...parametrizedTests,
      quiz: parametrizedTestsQuiz,
      multipleChoice: parametrizedTestsMultipleChoice,
    },
    {
      ...mockingWithUnittestMock,
      quiz: mockingWithUnittestMockQuiz,
      multipleChoice: mockingWithUnittestMockMultipleChoice,
    },
    {
      ...pytestPluginsExtensions,
      quiz: pytestPluginsExtensionsQuiz,
      multipleChoice: pytestPluginsExtensionsMultipleChoice,
    },
    {
      ...testingDatabasesSqlalchemy,
      quiz: testingDatabasesSqlalchemyQuiz,
      multipleChoice: testingDatabasesSqlalchemyMultipleChoice,
    },
    {
      ...testingAsyncCode,
      quiz: testingAsyncCodeQuiz,
      multipleChoice: testingAsyncCodeMultipleChoice,
    },
    {
      ...integrationTesting,
      quiz: integrationTestingQuiz,
      multipleChoice: integrationTestingMultipleChoice,
    },
    {
      ...testCoverage,
      quiz: testCoverageQuiz,
      multipleChoice: testCoverageMultipleChoice,
    },
    {
      ...testDrivenDevelopment,
      quiz: testDrivenDevelopmentQuiz,
      multipleChoice: testDrivenDevelopmentMultipleChoice,
    },
    {
      ...propertyBasedTesting,
      quiz: propertyBasedTestingQuiz,
      multipleChoice: propertyBasedTestingMultipleChoice,
    },
    {
      ...codeQualityTools,
      quiz: codeQualityToolsQuiz,
      multipleChoice: codeQualityToolsMultipleChoice,
    },
    {
      ...precommitHooksCICD,
      quiz: precommitHooksCICDQuiz,
      multipleChoice: precommitHooksCICDMultipleChoice,
    },
    {
      ...testingBestPractices,
      quiz: testingBestPracticesQuiz,
      multipleChoice: testingBestPracticesMultipleChoice,
    },
  ],
};
