import smtplib
from email.mime.text import MIMEText

def send_email(subject, body):
    # Email configuration
    sender_email = 'naveensvs.pasala@gmail.com'
    receiver_email = 'naveensvs.pasala@gmail.com'
    password = 'Testqaz@123'
    
    # Compose email
    message = MIMEText(body)
    message['Subject'] = subject
    message['From'] = sender_email
    message['To'] = receiver_email

    # Send email
    with smtplib.SMTP_SSL('smtp.example.com', 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

def notify_success():
    subject = 'MLOps Data Pipeline Success'
    body = 'MLOPS: data pipeline has successfully completed.'
    send_email(subject, body)

def notify_failure(error_message):
    subject = 'MLOps Data Pipeline Failure'
    body = f'MLOPS: data pipeline encountered an error:\n\n{error_message}'
    send_email(subject, body)