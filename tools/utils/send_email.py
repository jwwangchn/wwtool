import wwtool
import base64

passwd_file = 'tools/utils/base64_passwd.passwd'
with open(passwd_file, 'r') as f:
    passwd_base64 = f.readline()
    passwd = base64.b64decode(passwd_base64).decode()

email = wwtool.Email(passwd = passwd)
email.send()