from datetime import date
from datetime import timedelta

cur = date.fromisoformat('1900-01-06')
end = date.fromisoformat('2000-12-31')
week_delta = timedelta(weeks=1)
count = 0

while cur < end:
    if cur.day == 1:
        print(f'{cur}')
        count += 1
    cur += week_delta

print(f'Number of 1st Day Sundays in 20th Century: {count}')