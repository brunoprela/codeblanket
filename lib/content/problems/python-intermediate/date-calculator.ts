/**
 * Date Range Calculator
 * Problem ID: intermediate-date-calculator
 * Order: 6
 */

import { Problem } from '../../../types';

export const intermediate_date_calculatorProblem: Problem = {
  id: 'intermediate-date-calculator',
  title: 'Date Range Calculator',
  difficulty: 'Medium',
  description: `Create a utility for common date calculations.

**Functions to Implement:**
1. Calculate age from birthdate
2. Find business days between two dates (exclude weekends)
3. Get all dates in a month
4. Check if date is in range

**Date Format:** YYYY-MM-DD (ISO format)

**Business Days:** Monday-Friday only (ignore holidays)`,
  examples: [
    {
      input: 'calculate_age("1990-05-15")',
      output: '33 (or current age)',
    },
    {
      input: 'business_days_between("2024-01-01", "2024-01-05")',
      output: '4 (excluding weekend)',
    },
  ],
  constraints: [
    'Use datetime module',
    'Handle invalid dates',
    'ISO format (YYYY-MM-DD)',
  ],
  hints: [
    'Use datetime.strptime() to parse',
    'timedelta for date arithmetic',
    'weekday() returns 0-6 (Monday-Sunday)',
  ],
  starterCode: `from datetime import datetime, timedelta, date

def calculate_age(birthdate_str):
    """
    Calculate age from birthdate.
    
    Args:
        birthdate_str: Birthdate in YYYY-MM-DD format
        
    Returns:
        Age in years as integer
        
    Examples:
        >>> calculate_age("1990-05-15")
        33
    """
    pass


def business_days_between(start_str, end_str):
    """
    Count business days between two dates (exclude weekends).
    
    Args:
        start_str: Start date in YYYY-MM-DD format
        end_str: End date in YYYY-MM-DD format
        
    Returns:
        Number of business days
        
    Examples:
        >>> business_days_between("2024-01-01", "2024-01-05")
        4
    """
    pass


def get_month_dates(year, month):
    """
    Get all dates in a given month.
    
    Args:
        year: Year as integer
        month: Month as integer (1-12)
        
    Returns:
        List of date objects for each day in month
        
    Examples:
        >>> len(get_month_dates(2024, 1))
        31
    """
    pass


def is_date_in_range(check_date_str, start_str, end_str):
    """
    Check if date is within range (inclusive).
    
    Args:
        check_date_str: Date to check
        start_str: Range start date
        end_str: Range end date
        
    Returns:
        True if date is in range, False otherwise
    """
    pass


# Test
print(f"Age: {calculate_age('1990-05-15')}")
print(f"Business days: {business_days_between('2024-01-01', '2024-01-10')}")
print(f"Days in Jan 2024: {len(get_month_dates(2024, 1))}")
print(f"In range: {is_date_in_range('2024-06-15', '2024-01-01', '2024-12-31')}")
`,
  testCases: [
    {
      input: ['1990-05-15'],
      expected: 33,
    },
    {
      input: ['2024-01-01', '2024-01-05'],
      expected: 4,
    },
  ],
  solution: `from datetime import datetime, timedelta, date
from calendar import monthrange

def calculate_age(birthdate_str):
    birthdate = datetime.strptime(birthdate_str, "%Y-%m-%d").date()
    today = date.today()
    
    age = today.year - birthdate.year
    # Adjust if birthday hasn't occurred this year
    if (today.month, today.day) < (birthdate.month, birthdate.day):
        age -= 1
    
    return age


def business_days_between(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    
    # Count business days
    count = 0
    current = start
    while current <= end:
        # weekday() returns 0-6 (Monday-Sunday), so 0-4 are weekdays
        if current.weekday() < 5:
            count += 1
        current += timedelta(days=1)
    
    return count


def get_month_dates(year, month):
    # Get number of days in month
    _, num_days = monthrange(year, month)
    
    # Create list of all dates
    dates = []
    for day in range(1, num_days + 1):
        dates.append(date(year, month, day))
    
    return dates


def is_date_in_range(check_date_str, start_str, end_str):
    check_date = datetime.strptime(check_date_str, "%Y-%m-%d").date()
    start = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    
    return start <= check_date <= end`,
  timeComplexity: 'O(d) for business_days_between where d is days between',
  spaceComplexity: 'O(d) for get_month_dates',
  order: 6,
  topic: 'Python Intermediate',
};
