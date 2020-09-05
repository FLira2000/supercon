import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email import encoders
from os.path import basename

MAIL_CONFIG = { "email": "data.supercon@gmail.com", "password": "jxqbuvfyuztsnlcg" }
targetEmail = "fabioliradev@gmail.com"
ccEmail = "josiasdsj1@gmail.com"

def mailSender(attachmentNameList, bodyMessage):

    #sending results over email
    msg = MIMEMultipart()
    msg['From'] = "data.supercon@gmail.com"
    msg['To'] = targetEmail
    msg['Subject'] = "Prediction data"
    msg['Cc'] = ', '.join([ccEmail])

    body = bodyMessage

    msg.attach(MIMEText(body, 'plain'))

    for attachmentName in attachmentNameList:
        part = MIMEApplication(open(attachmentName, "rb").read(), Name=basename(attachmentName))
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(attachmentName)
        msg.attach(part)

    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(MAIL_CONFIG["email"], MAIL_CONFIG["password"])
    s.send_message(msg)
    s.quit()

    return 
