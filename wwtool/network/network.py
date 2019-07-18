import smtplib
from email.mime.text import MIMEText


def send_email(title=None, message=None):
    mail_host = "smtp.163.com"          # SMTP server
    mail_user = "jwwangchn"             # username
    mail_pass = "APTX4869"              # passwd

    sender = 'jwwangchn@163.com'        # sender
    receivers = ['877150341@qq.com']    # receiver

    message = MIMEText(message, 'plain', 'utf-8')
    message['From'] = "{}".format(sender)
    message['To'] = ",".join(receivers)
    message['Subject'] = title

    try:
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("mail has been send to {} successfully.".format(receivers[0]))
    except smtplib.SMTPException as error:
        print(error)