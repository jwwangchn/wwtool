import smtplib
from email.mime.text import MIMEText

class Email():
    def __init__(self, host="smtp.163.com", user="jwwangchn", passwd="passwd", sender="jwwangchn@163.com"):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.sender = sender
        
    def send(self, receivers=['877150341@qq.com'], title="Finish to train the model!!!", message="This is the default email message!"):
        message = MIMEText(message, 'plain', 'utf-8')
        message['From'] = "{}".format(self.sender)
        message['To'] = ",".join(receivers)
        message['Subject'] = title
        try:
            smtpObj = smtplib.SMTP_SSL(self.host, 465)
            smtpObj.login(self.user, self.passwd)
            smtpObj.sendmail(self.sender, receivers, message.as_string())
            print("Email has been send to {} successfully.".format(receivers[0]))
        except smtplib.SMTPException as error:
            print("Email is failed to send, the error is: ", error)


