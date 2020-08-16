from datetime import datetime

def log_to_file(filename, message):

    log_message = f'{datetime.now()}::: {message}'

    with open(filename, 'a+') as fl:
        fl.write(log_message + '\n')

    print(log_message)