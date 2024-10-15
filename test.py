def decimal_to_minutes(decimal):
    decimal, minutes = str(decimal).split('.')
    minutes = int(minutes) * 60 / 10

    if str(minutes).split('.')[1] != '0':
        seconds = str(minutes).split('.')[1]
        seconds = int(seconds) * 60 / 10

        return f"{int(decimal)}° {int(minutes)}' {int(seconds)}"
    else:
        return f"{int(decimal)}° {int(minutes)}'"

print(decimal_to_minutes(47.2))  # 37° 46' 12''
    