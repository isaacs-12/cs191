import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import config # contains login info for the email account being used

scans_index = 'Scans_Index.txt'
email_draft = 'Email_Boilerplate.txt'

def send(subject, msg, model=None, model_path, user_email):
	try:
		msg = MIMEMultipart()
    		msg['From'] = config.EMAIL_ADDRESS
    		msg['To'] = (user_email)
		msg['Subject'] = subject

    		msg.attach(MIMEText(text))
		with open(model, "rb") as f:
		    part = MIMEApplication(
			f.read(),
			Name=basename(f)
		    )
		part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
		msg.attach(part)
		
		
		server = smtplib.SMTP('smtp.gmail.com:587')
		server.ehlo()
		server.starttls()
		server.login(config.EMAIL_ADDRESS, config.PASSWORD)
		server.sendmail(config.EMAIL_ADDRESS, user_email, msg.as_string())
		server.quit()
		print ("Success! Email sent.")
	except:
		print ("Email send failure.")

	return

def main()
	name = input("User's Name: ")
	email = input("User's Email: ")
	model_id = input("User's Model ID: ") #filepath
	f = open(email_draft, 'r')
	email_body = f.readlines()
	email_body.replace('<name>', name, 1)
	send("Your TumorViz Model", email_body, model_id, email)
	return()
