/**
 * Date and Time Section
 */

export const datetimeSection = {
  id: 'datetime',
  title: 'Date and Time',
  content: `# Date and Time

## datetime Module

\`\`\`python
from datetime import datetime, date, time, timedelta

# Current date and time
now = datetime.now()
print(now)  # 2024-01-15 10:30:45.123456

# Current date
today = date.today()
print(today)  # 2024-01-15

# Create specific datetime
dt = datetime(2024, 1, 15, 10, 30, 45)

# Create specific date
d = date(2024, 1, 15)

# Create specific time
t = time(10, 30, 45)
\`\`\`

## Formatting Dates

\`\`\`python
# Convert to string
now = datetime.now()
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print(formatted)  # "2024-01-15 10:30:45"

# Common format codes:
# %Y - Year (4 digits)
# %m - Month (01-12)
# %d - Day (01-31)
# %H - Hour (00-23)
# %M - Minute (00-59)
# %S - Second (00-59)
# %A - Weekday name (Monday)
# %B - Month name (January)

examples = [
    now.strftime("%B %d, %Y"),          # "January 15, 2024"
    now.strftime("%m/%d/%y"),           # "01/15/24"
    now.strftime("%A, %B %d, %Y"),      # "Monday, January 15, 2024"
    now.strftime("%I:%M %p"),           # "10:30 AM"
]
\`\`\`

## Parsing Dates

\`\`\`python
# Parse string to datetime
date_string = "2024-01-15 10:30:45"
dt = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

# Different formats
formats = [
    ("01/15/2024", "%m/%d/%Y"),
    ("15-Jan-2024", "%d-%b-%Y"),
    ("2024-01-15T10:30:45", "%Y-%m-%dT%H:%M:%S"),
]

for date_str, fmt in formats:
    dt = datetime.strptime(date_str, fmt)
\`\`\`

## Date Arithmetic

\`\`\`python
# Add/subtract time
now = datetime.now()
tomorrow = now + timedelta(days=1)
last_week = now - timedelta(weeks=1)
in_2_hours = now + timedelta(hours=2)

# Difference between dates
date1 = datetime(2024, 1, 15)
date2 = datetime(2024, 1, 10)
diff = date1 - date2
print(diff.days)  # 5

# Time components
delta = timedelta(days=5, hours=3, minutes=30)
print(delta.total_seconds())  # Convert to seconds
\`\`\`

## Working with Timestamps

\`\`\`python
import time

# Get current timestamp
timestamp = time.time()  # Seconds since epoch

# Convert timestamp to datetime
dt = datetime.fromtimestamp(timestamp)

# Convert datetime to timestamp
timestamp = dt.timestamp()

# UTC time
utc_now = datetime.utcnow()
\`\`\`

## Timezone Handling

\`\`\`python
from datetime import timezone
import pytz  # Third-party library for better timezone support

# Create timezone-aware datetime
utc_time = datetime.now(timezone.utc)

# With pytz (install with: pip install pytz)
eastern = pytz.timezone('US/Eastern')
eastern_time = datetime.now(eastern)

# Convert between timezones
utc_time = datetime.now(pytz.UTC)
eastern_time = utc_time.astimezone(eastern)
\`\`\`

## Common Operations

\`\`\`python
# Get day of week
today = date.today()
day_of_week = today.strftime("%A")  # "Monday"
weekday_num = today.weekday()       # 0 = Monday

# First day of month
first_day = today.replace(day=1)

# Last day of month
from calendar import monthrange
last_day_num = monthrange(today.year, today.month)[1]
last_day = today.replace(day=last_day_num)

# Age calculation
birthdate = date(1990, 5, 15)
today = date.today()
age = today.year - birthdate.year
if (today.month, today.day) < (birthdate.month, birthdate.day):
    age -= 1
\`\`\`

## Best Practices

1. **Use datetime for timestamps**: More features than time module
2. **Store as UTC**: Convert to local timezone only for display
3. **Use ISO format**: \`%Y-%m-%d %H:%M:%S\` is unambiguous
4. **Handle timezones**: Use pytz for production code
5. **Validate dates**: Check for valid date ranges
6. **Use timedelta**: For date arithmetic`,
  videoUrl: 'https://www.youtube.com/watch?v=eirjjyP2qcQ',
};
