/**
 * Quiz questions for Date and Time section
 */

export const datetimeQuiz = [
  {
    id: 'pi-datetime-q-1',
    question:
      'Explain the difference between datetime, date, and time objects. When should you use each?',
    hint: 'Think about what information each stores and typical use cases.',
    sampleAnswer:
      "datetime stores both date AND time (year, month, day, hour, minute, second, microsecond) - use for timestamps, events, logging. date stores only the date (year, month, day) - use for birthdays, schedules, appointments where time doesn't matter. time stores only time (hour, minute, second) - use for recurring events like \"daily at 9 AM\" without a specific date. Most common is datetime since it's the most complete. Use date when you explicitly don't care about time, and time for time-of-day patterns.",
    keyPoints: [
      'datetime: full timestamp with date and time',
      'date: just year/month/day',
      'time: just hour/minute/second',
      'datetime is most versatile and commonly used',
    ],
  },
  {
    id: 'pi-datetime-q-2',
    question:
      'Why should you always store timestamps in UTC and convert to local timezone only for display?',
    hint: 'Consider daylight saving time, consistency, and timezone conversions.',
    sampleAnswer:
      "Storing in UTC avoids ambiguity: 1) No daylight saving time issues (clock shifts, missing hours), 2) Consistent reference point globally, 3) Easy to convert to any timezone for display, 4) Prevents bugs when users travel or move. Common mistake: storing local time causes problems when DST changes or when comparing times across timezones. Always: store UTC in database, convert to user's timezone only when displaying. This is a critical best practice for any application with users in multiple locations.",
    keyPoints: [
      'UTC avoids DST ambiguity and clock shifts',
      'Consistent global reference point',
      'Easy to convert to any local timezone',
      'Store UTC, display in local timezone',
    ],
  },
  {
    id: 'pi-datetime-q-3',
    question:
      'What is timedelta and how is it used for date arithmetic? Why not just add/subtract integers?',
    hint: 'Think about handling months, leap years, and DST.',
    sampleAnswer:
      'timedelta represents a duration (like "3 days" or "2 hours") and handles date arithmetic correctly - accounting for months with different days, leap years, and DST. You can\'t just add integers because: tomorrow isn\'t always today+1 (DST shifts), next month isn\'t always +30 days (different month lengths), next year isn\'t always +365 days (leap years). Use timedelta for: adding/subtracting time (today + timedelta(days=7)), calculating differences (end_date - start_date), measuring elapsed time. It handles all calendar complexity automatically.',
    keyPoints: [
      'Represents duration, not a point in time',
      'Handles calendar complexity (DST, leap years)',
      'Use for adding/subtracting time periods',
      'Returns from subtracting two datetimes',
    ],
  },
];
